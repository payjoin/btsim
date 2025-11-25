use std::collections::HashMap;

use crate::{
    actions::{
        Action, CompositeScorer, CompositeStrategy, PayjoinStrategy, Strategy, UnilateralSpender,
        WalletView,
    },
    blocks::{BroadcastSetHandleMut, BroadcastSetId},
    message::{MessageData, MessageId, MessageType, PayjoinProposal},
    Simulation, TimeStep,
};
use bdk_coin_select::{
    metrics::LowestFee, Candidate, ChangePolicy, CoinSelector, Drain, DrainWeights, Target,
    TargetFee, TargetOutputs, TR_DUST_RELAY_MIN_VALUE, TR_KEYSPEND_TXIN_WEIGHT,
};
use bitcoin::{transaction::InputWeightPrediction, Amount};
use im::{OrdSet, Vector};

use crate::transaction::*;

define_entity_mut_updatable!(
    Wallet,
    {
        pub(crate) id: WalletId,
        pub(crate) addresses: Vec<AddressId>,         // TODO split into internal/external?
        pub(crate) own_transactions: Vec<TxId>,       // transactions originating from this wallet
        pub(crate) last_wallet_info_id: WalletInfoId, // Monotone
        // Monotone index of the last message that was processed by this wallet
        pub(crate) messages_processed: OrdSet<MessageId>,
    },
    {
        pub(crate) broadcast_set_id: BroadcastSetId,
        pub(crate) payment_obligations: OrdSet<PaymentObligationId>,
        pub(crate) expected_payments: OrdSet<PaymentObligationId>,
        pub(crate) broadcast_transactions: Vector<TxId>,
        pub(crate) received_transactions: Vector<TxId>,
        pub(crate) confirmed_utxos: OrdSet<Outpoint>,    // TODO locktimes
        pub(crate) unconfirmed_transactions: OrdSet<TxId>,
        pub(crate) unconfirmed_txos: OrdSet<Outpoint>,  // compute CPFP cost
        pub(crate) unconfirmed_spends: OrdSet<Outpoint>, // RBFable
        pub(crate) payment_obligation_to_payjoin: HashMap<PaymentObligationId, MessageId>,
        /// Map of unconfirmed txos to the cospends that they are in
        // TODO: need something similar for outputs as to prevent double spending to the same payment obligation
        pub(crate) unconfirmed_txos_in_payjoins: HashMap<Outpoint, MessageId>,
        /// Map of txids to the payment obligations that they are associated with
        /// Sim state should refrence this when updating wallet states after confirmation
        pub(crate) txid_to_handle_payment_obligation: HashMap<TxId, PaymentObligationId>,

        /// Set of payment obligations that have been handled
        pub(crate) handled_payment_obligations: OrdSet<PaymentObligationId>,
    }
);

impl<'a> WalletHandle<'a> {
    pub(crate) fn data(&self) -> &'a WalletData {
        &self.sim.wallet_data[self.id.0]
    }

    pub(crate) fn info(&self) -> &'a WalletInfo {
        &self.sim.wallet_info[self.data().last_wallet_info_id.0]
    }

    // TODO: this should take into account liabilties spending unconfirmed UTXOs. For which a CPFP cost model is needed
    // In the future in needs to take as arg the current mempool and somethign to predict the state of the mempool overtime
    pub(crate) fn effective_balance(&self) -> Amount {
        let utxos: Vec<OutputHandle<'a>> = self.unspent_coins().collect();
        let outputs_amounts = utxos.iter().map(|output| output.data().amount).sum();

        outputs_amounts
    }

    // TODO give utxo list as argument so that different variants can be used
    // TODO return change information
    pub(crate) fn select_coins(
        &self,
        target: Target,
        long_term_feerate: bitcoin::FeeRate,
    ) -> (impl Iterator<Item = OutputHandle<'a>>, Drain) {
        // TODO change
        // TODO group by address
        let utxos: Vec<OutputHandle<'a>> = self.unspent_coins().collect();

        let candidates: Vec<Candidate> = utxos
            .iter()
            .enumerate()
            .map(|(_, o)| Candidate {
                value: o.data().amount.to_sat(),
                weight: TR_KEYSPEND_TXIN_WEIGHT,
                input_count: 1,
                is_segwit: true,
            })
            .collect();

        let mut coin_selector = CoinSelector::new(&candidates);
        let drain_weights = DrainWeights::default();

        let dust_limit = TR_DUST_RELAY_MIN_VALUE;

        let long_term_feerate = bdk_coin_select::FeeRate::from_sat_per_wu(
            long_term_feerate.to_sat_per_kwu() as f32 * 1e-3,
        );

        let change_policy = ChangePolicy::min_value_and_waste(
            drain_weights,
            dust_limit,
            target.fee.rate,
            long_term_feerate,
        );

        let metric = LowestFee {
            target,
            long_term_feerate,
            change_policy,
        };

        match coin_selector.run_bnb(metric, 100_000) {
            Err(err) => {
                println!("BNB failed to find a solution: {}", err);

                coin_selector
                    .select_until_target_met(target)
                    .expect("coin selection should always succeed since payments consider budger lower bound");
            }
            Ok(score) => {
                println!("BNB found a solution with score {}", score);
            }
        };

        let selection = coin_selector
            .apply_selection(&utxos)
            .cloned()
            .collect::<Vec<_>>();

        let change = coin_selector.drain(target, change_policy);

        (selection.into_iter(), change)
    }

    fn potentially_spendable_txos(&self) -> impl Iterator<Item = OutputHandle<'a>> + '_ {
        self.info()
            .confirmed_utxos
            .iter()
            .chain(self.info().unconfirmed_txos.iter())
            .map(|outpoint| OutputHandle::new(self.sim, *outpoint))
    }

    fn unspent_coins(&self) -> impl Iterator<Item = OutputHandle<'a>> + '_ {
        self.potentially_spendable_txos().filter(|o| {
            !self.info().unconfirmed_spends.contains(&o.outpoint())
            // TODO Startegies should inform which inputs can be spendable.
            // TODO: these inputs should unlock if the payjoin is expired or the associated payment obligation is due soon (i.e payment anxiety)
            // && !self
            //     .info()
            //     .unconfirmed_txos_in_payjoins
            //     .contains_key(&o.outpoint())
        })
    }

    fn double_spendable_coins(&self) -> impl Iterator<Item = OutputHandle<'a>> + '_ {
        self.potentially_spendable_txos()
            .filter(|o| self.info().unconfirmed_spends.contains(&o.outpoint()))
    }
}

impl<'a> WalletHandleMut<'a> {
    pub(crate) fn data_mut<'b>(&'b mut self) -> &'b mut WalletData {
        &mut self.sim.wallet_data[self.id.0]
    }

    fn info_mut<'b>(&'b mut self) -> &'b mut WalletInfo {
        let last_wallet_info_id = self.data().last_wallet_info_id;
        &mut self.sim.wallet_info[last_wallet_info_id.0]
    }

    pub(crate) fn handle(&self) -> WalletHandle {
        WalletHandle {
            sim: self.sim,
            id: self.data().id,
        }
    }

    pub(crate) fn new_address(&mut self) -> AddressId {
        let id = AddressId(self.sim.address_data.len());
        self.sim.wallet_data[self.id.0].addresses.push(id);
        self.sim
            .address_data
            .push(AddressData { wallet_id: self.id });
        id
    }

    /// Returns the payjoin proposal ids that need to be processed
    fn read_messages(&mut self) -> Vec<(MessageId, PayjoinProposal)> {
        let my_id = self.id;
        let messages_to_process = self
            .sim
            .messages
            .iter()
            .filter(|message| !self.data().messages_processed.contains(&message.id))
            .cloned()
            .collect::<Vec<_>>();
        if messages_to_process.is_empty() {
            return Vec::new();
        }

        let mut payjoin_ids_to_process = Vec::new();

        for message in messages_to_process.iter() {
            if message.from == my_id {
                // Ignore messages from myself
                continue;
            }
            if message.to.is_none() {
                // Ignore broadcast messages for now
                continue;
            }
            if message.to.unwrap() != my_id {
                // Ignore messages not for me
                continue;
            }
            match &message.message {
                MessageType::InitiatePayjoin(payjoin_proposal) => {
                    payjoin_ids_to_process.push((message.id, payjoin_proposal.clone()));
                }
            }
        }

        // Plus one to avoid processing the same message twice
        self.data_mut()
            .messages_processed
            .extend(messages_to_process.iter().map(|message| message.id));
        payjoin_ids_to_process
    }

    /// stateless utility function to construct a transaction for a given payment obligation
    fn construct_transaction_template(
        &self,
        obligation: &PaymentObligationData,
        change_addr: &AddressId,
        to_address: &AddressId,
    ) -> TxData {
        let amount = obligation.amount.to_sat();
        let target = Target {
            fee: TargetFee {
                rate: bdk_coin_select::FeeRate::from_sat_per_vb(1.0),
                replace: None,
            },
            outputs: TargetOutputs {
                value_sum: amount,
                weight_sum: 34, // TODO use payment.to to derive an address, payment.into() ?
                n_outputs: 1,
            },
        };
        let long_term_feerate = bitcoin::FeeRate::from_sat_per_vb(10).expect("valid fee rate");

        let (selected_coins, drain) = self.select_coins(target, long_term_feerate);
        let mut tx = TxData::default();
        tx.inputs = selected_coins
            .map(|o| Input {
                outpoint: o.outpoint,
            })
            .collect();
        tx.outputs = vec![
            Output {
                amount: Amount::from_sat(amount),
                address_id: *to_address,
            },
            Output {
                amount: Amount::from_sat(drain.value),
                address_id: *change_addr,
            },
        ];
        tx
    }

    // TODO: bring this back as as an strategy
    /// Returns the next payment obligation that is not handled
    /// TODO: this should be a priority queue
    // fn next_payment_obligation(&'a self) -> Option<PaymentObligationId> {
    //     self.info()
    //         .payment_obligations
    //         .clone()
    //         .difference(self.info().handled_payment_obligations.clone())
    //         .iter()
    //         .filter(|payment_obligation_id| {
    //             let anxiety_factor = self.deadline_anxiety(
    //                 payment_obligation_id.with(self.sim).data().deadline.0 as i32,
    //             );
    //             // If already in a cospend or not due soon
    //             !self
    //                 .info()
    //                 .payment_obligation_to_cospend
    //                 .contains_key(payment_obligation_id)
    //                 || anxiety_factor <= self.sim.config.payment_obligation_deadline_threshold
    //         })
    //         .next()
    //         .cloned()
    // }

    // fn participate_in_cospend(
    //     &mut self,
    //     message_id: MessageId,
    //     cospend: &PayjoinProposal,
    // ) -> Option<TxId> {
    //     if cospend.valid_till < self.sim.current_timestep {
    //         return None;
    //     }
    //     // if we have a payment obligation then lets batch it with this cospend
    //     if let Some(payment_obligation_id) = self.next_payment_obligation() {
    //         let payment_obligation = payment_obligation_id.with(self.sim).data().clone();
    //         let change_addr = self.new_address();
    //         let to_address = payment_obligation.to.with_mut(self.sim).new_address();
    //         let mut tx_template =
    //             self.construct_transaction_template(&payment_obligation, &change_addr, &to_address);
    //         // "Lock" The inputs to this cospend. These inputs can be spent if the cospend is expired and our payment is due soon
    //         for input in tx_template.inputs.iter() {
    //             self.info_mut()
    //                 .unconfirmed_txos_in_cospends
    //                 .insert(input.outpoint, message_id);
    //         }

    //         self.ack_transaction(&mut tx_template);
    //         tx_template.inputs.extend(cospend.tx.inputs.iter().cloned());
    //         tx_template
    //             .outputs
    //             .extend(cospend.tx.outputs.iter().cloned());
    //         tx_template
    //             .wallet_acks
    //             .extend(cospend.tx.wallet_acks.iter().cloned());

    //         debug_assert!(tx_template.wallet_acks.contains(&self.id));

    //         let tx_id = self.spend_tx(tx_template);

    //         // Keep an index of what payment obligations are being handled in which cospends
    //         self.info_mut()
    //             .payment_obligation_to_cospend
    //             .insert(payment_obligation_id, message_id);
    //         self.info_mut()
    //             .txid_to_handle_payment_obligation
    //             .insert(tx_id, payment_obligation_id);

    //         return Some(tx_id);
    //     }

    //     None
    // }

    fn create_payjoin(
        &mut self,
        // Message id is serving as a proxy for payjoin id
        payment_obligation: &PaymentObligationId,
    ) -> MessageData {
        let payment_obligation = payment_obligation.with(self.sim).data().clone();
        let message_id = MessageId(self.sim.messages.len());
        let change_addr = self.new_address();
        let to_address = payment_obligation.to.with_mut(self.sim).new_address();
        let mut tx_template =
            self.construct_transaction_template(&payment_obligation, &change_addr, &to_address);
        self.ack_transaction(&mut tx_template);
        debug_assert!(tx_template.wallet_acks.contains(&self.id));
        let payjoin_proposal = PayjoinProposal {
            tx: tx_template,
            valid_till: payment_obligation.deadline,
        };
        // "Lock" The inputs to this cospend. These inputs can be spent if the cospend is expired and our payment is due soon
        for input in payjoin_proposal.tx.inputs.iter() {
            self.info_mut()
                .unconfirmed_txos_in_payjoins
                .insert(input.outpoint, message_id);
        }
        self.info_mut()
            .payment_obligation_to_payjoin
            .insert(payment_obligation.id, message_id);

        MessageData {
            id: message_id,
            from: self.id,
            to: Some(payment_obligation.to),
            message: MessageType::InitiatePayjoin(payjoin_proposal),
        }
    }

    pub(crate) fn wallet_view(&self) -> WalletView {
        let mut payjoins = Vec::new();
        let messages = self
            .sim
            .messages
            .iter()
            .filter(|message| !self.data().messages_processed.contains(&message.id))
            .cloned()
            .collect::<Vec<_>>();
        for message in messages.iter() {
            match &message.message {
                MessageType::InitiatePayjoin(payjoin_proposal) => {
                    payjoins.push(payjoin_proposal.clone());
                }
            }
        }
        let payment_obligations = self
            .info()
            .payment_obligations
            .clone()
            .difference(self.info().handled_payment_obligations.clone())
            .iter()
            .map(|po| po.with(self.sim).data().clone())
            .collect::<Vec<_>>();
        WalletView::new(payment_obligations, messages, self.sim.current_timestep)
    }

    pub(crate) fn do_action(&'a mut self, action: &Action) {
        match action {
            Action::Wait => {}
            Action::UnilateralSpend(po) => {
                let _ = self.handle_payment_obligation(po);
            }
            Action::InitiatePayjoin(po) => {
                let message = self.create_payjoin(po);
                self.sim.broadcast_message(message);
            }
            _ => unimplemented!(),
        }
    }

    pub(crate) fn wake_up(&'a mut self) {
        let scorer = CompositeScorer {
            payjoin_utility_factor: 0.01,
            payment_obligation_utility_factor: 0.01,
        };
        let strategy = CompositeStrategy {
            strategies: vec![Box::new(UnilateralSpender), Box::new(PayjoinStrategy)],
        };
        let wallet_view = self.wallet_view();
        let action = strategy
            .enumerate_candidate_actions(&wallet_view)
            .into_iter()
            .min_by_key(|action| scorer.score_action(action, self))
            .unwrap_or(Action::Wait);
        self.do_action(&action);

        // TODO: bring all the payjoin logic back as a strategy
        // let cospend_to_process = self.read_messages();
        // let mut txs_to_broadcast = Vec::new();
        // for (message_id, cospend) in cospend_to_process {
        //     if let Some(tx_id) = self.participate_in_cospend(message_id, &cospend) {
        //         txs_to_broadcast.push(tx_id);
        //     }
        // }

        // // If I have any payment obligations I should try to spend them if they are due soon
        // // Other wise I should register my inputs and look for others to collaborate with
        // if let Some(payment_obligation_id) = self.next_payment_obligation() {
        //     let payment_obligation = payment_obligation_id.with(self.sim).data().clone();

        //     // TODO: this should be configurable
        //     // Right now the wallets are patient for the most part
        //     if self.deadline_anxiety(payment_obligation.deadline.0 as i32)
        //         <= self.sim.config.payment_obligation_deadline_threshold
        //     {
        //         self.handle_payment_obligations(&payment_obligation);
        //         return;
        //     }

        //     // If its not due soon lets batch the payment
        //     let message_id = MessageId(self.sim.messages.len());
        //     let cospend = self.create_cospend(message_id, &payment_obligation);
        //     let message = MessageData {
        //         id: message_id,
        //         from: self.id,
        //         to: Some(payment_obligation.to),
        //         message: MessageType::RegisterCospend(cospend),
        //     };
        //     self.sim.broadcast_message(message.clone());

        //     self.broadcast(txs_to_broadcast);
        // }
    }

    fn handle_payment_obligation(
        &'a mut self,
        payment_obligation_id: &PaymentObligationId,
    ) -> TxId {
        let payment_obligation = payment_obligation_id.with(&self.sim).data().clone();
        let change_addr = self.new_address();
        let to_wallet = payment_obligation.to;
        let to_address = to_wallet.with_mut(self.sim).new_address();
        let mut tx_template =
            self.construct_transaction_template(&payment_obligation, &change_addr, &to_address);
        self.ack_transaction(&mut tx_template);

        let tx_id = self.spend_tx(tx_template);
        self.info_mut()
            .txid_to_handle_payment_obligation
            .insert(tx_id, *payment_obligation_id);
        self.broadcast(vec![tx_id]);
        tx_id
    }

    fn ack_transaction(&self, tx: &mut TxData) {
        tx.wallet_acks.push(self.id);
    }

    // TODO: refactor this? Do we event need this?
    fn spend_tx(&mut self, txdata: TxData) -> TxId {
        // TODO: assert this is my obligation
        let spend = self
            .new_tx(|tx, _| {
                tx.inputs = txdata.inputs;
                tx.outputs = txdata.outputs;
                tx.wallet_acks = txdata.wallet_acks;
            })
            .id;

        spend
    }

    pub(crate) fn new_tx<F>(&mut self, build: F) -> TxHandle
    where
        F: FnOnce(&mut TxData, &Simulation),
    {
        let id = self.sim.new_tx(build);
        self.data_mut().own_transactions.push(id);
        TxHandle::new(self.sim, id)
    }

    pub(crate) fn broadcast(
        &'a mut self,
        txs: impl IntoIterator<Item = TxId>,
    ) -> BroadcastSetHandleMut<'a> {
        let mut wallet_info = self.info().clone();

        let txs = Vector::from_iter(txs);

        wallet_info.broadcast_transactions.append(txs.clone());

        // TODO refactor boilerplate for updating wallet ID
        let id = WalletInfoId(self.sim.wallet_info.len());
        self.sim.wallet_info.push(wallet_info);
        let data = self.data_mut();
        data.last_wallet_info_id = id;

        self.sim.broadcast(txs)
    }
}

define_entity!(
    PaymentObligation,
    {
        pub(crate) id: PaymentObligationId,
        pub(crate) deadline: TimeStep,
        pub(crate) amount: Amount,
        pub(crate) from: WalletId,
        pub(crate) to: WalletId,
    },
    {
        // TODO coin selection strategy agnostic (pessimal?) spendable balance lower
        // bound
    }
);

impl<'a> PaymentObligationHandle<'a> {
    pub(crate) fn data(&self) -> &'a PaymentObligationData {
        &self.sim.payment_data[self.id.0]
    }
}

define_entity!(Address, {
    pub(crate) wallet_id: WalletId,
    // TODO script_type
    // TODO internal
    // TODO silent payments
}, {});

impl From<AddressData> for InputWeightPrediction {
    fn from(_: AddressData) -> Self {
        // TODO match on script_type
        bitcoin::transaction::InputWeightPrediction::P2TR_KEY_DEFAULT_SIGHASH
    }
}

impl From<AddressHandle<'_>> for InputWeightPrediction {
    fn from(address: AddressHandle<'_>) -> Self {
        Self::from(address.data().clone())
    }
}

// TODO traits?
impl<'a> AddressHandle<'a> {
    pub(crate) fn data(&'a self) -> &'a AddressData {
        &self.sim.address_data[self.id.0]
    }

    pub(crate) fn wallet(&self) -> WalletHandle<'a> {
        self.data().wallet_id.with(self.sim)
    }
}
