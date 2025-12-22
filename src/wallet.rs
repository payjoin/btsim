use crate::{
    actions::{Action, CompositeScorer, CompositeStrategy, WalletView},
    blocks::{BroadcastSetHandleMut, BroadcastSetId},
    bulletin_board::{BroadcastMessageType, BulletinBoardId},
    message::{MessageData, MessageId, MessageType, PayjoinProposal},
    Simulation, TimeStep,
};
use bdk_coin_select::{
    metrics::LowestFee, Candidate, ChangePolicy, CoinSelector, Drain, DrainWeights, Target,
    TargetFee, TargetOutputs, TR_DUST_RELAY_MIN_VALUE, TR_KEYSPEND_TXIN_WEIGHT,
};
use bitcoin::{transaction::InputWeightPrediction, Amount};
use im::{HashMap, OrdSet, Vector};
use log::{info, warn};

use crate::transaction::*;

define_entity_id_and_handle!(Wallet);
define_entity_handle_mut!(Wallet);
define_entity_info_id!(Wallet);
define_entity_data!(Wallet, {
    pub(crate) id: WalletId,
    pub(crate) addresses: Vec<AddressId>,         // TODO split into internal/external?
    pub(crate) own_transactions: Vec<TxId>,       // transactions originating from this wallet
    pub(crate) last_wallet_info_id: WalletInfoId, // Monotone
    // Monotone index of the last message that was processed by this wallet
    pub(crate) messages_processed: OrdSet<MessageId>,
    pub(crate) strategies: CompositeStrategy,
    pub(crate) scorer: CompositeScorer,
}, skip_eq_clone);
define_entity_info!(Wallet, {
        pub(crate) broadcast_set_id: BroadcastSetId,
        pub(crate) payment_obligations: OrdSet<PaymentObligationId>,
        pub(crate) expected_payments: OrdSet<PaymentObligationId>,
        pub(crate) broadcast_transactions: Vector<TxId>,
        pub(crate) received_transactions: Vector<TxId>,
        pub(crate) confirmed_utxos: OrdSet<Outpoint>,    // TODO locktimes
        pub(crate) unconfirmed_transactions: OrdSet<TxId>,
        pub(crate) unconfirmed_txos: OrdSet<Outpoint>,  // compute CPFP cost
        pub(crate) unconfirmed_spends: OrdSet<Outpoint>, // RBFable
        /// Payjoins that I sent to other wallets
        pub(crate) initiated_payjoins: HashMap<PaymentObligationId, BulletinBoardId>,
        /// Payjoins that I received from other wallets. A mapping of what payment obligation was used in this payjoin
        pub(crate) received_payjoins: HashMap<PaymentObligationId, BulletinBoardId>,
        /// Map of unconfirmed txos to the cospends that they are in
        // TODO: need something similar for outputs as to prevent double spending to the same payment obligation
        // TODO: generalize this to other type of interactive protocols
        pub(crate) unconfirmed_txos_in_payjoins: HashMap<Outpoint, BulletinBoardId>,
        /// Map of txids to the payment obligations that they are associated with
        /// Sim state should refrence this when updating wallet states after confirmation
        pub(crate) txid_to_payment_obligation_ids: HashMap<TxId, Vec<PaymentObligationId>>,

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
                // TODO: should be a error log
                warn!("BNB failed to find a solution: {}", err);

                coin_selector
                    .select_until_target_met(target)
                    .expect("coin selection should always succeed since payments consider budger lower bound");
            }
            Ok(_) => (),
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
            && !self
                .info()
                .unconfirmed_txos_in_payjoins
                .contains_key(&o.outpoint())
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

    /// stateless utility function to construct a transaction for a given payment obligation
    fn construct_transaction_template(
        &mut self,
        payment_obligation_ids: &[PaymentObligationId],
        change_addr: &AddressId,
    ) -> TxData {
        let mut amount_and_destination = vec![];
        for payment_obligation_id in payment_obligation_ids.iter() {
            let payment_obligation = payment_obligation_id.with(self.sim).data().clone();
            let to_wallet = payment_obligation.to;
            let to_address = to_wallet.with_mut(self.sim).new_address();
            amount_and_destination.push((payment_obligation.amount, to_address));
        }

        let amount = amount_and_destination
            .iter()
            .map(|(amount, _)| amount.to_sat())
            .sum();
        let target = Target {
            fee: TargetFee {
                rate: bdk_coin_select::FeeRate::from_sat_per_vb(1.0),
                replace: None,
            },
            outputs: TargetOutputs {
                value_sum: amount,
                weight_sum: 34, // TODO use payment.to to derive an address, payment.into() ?
                n_outputs: amount_and_destination.len(),
            },
        };
        let long_term_feerate = bitcoin::FeeRate::from_sat_per_vb(10).expect("valid fee rate");

        let (selected_coins, drain) = self.handle().select_coins(target, long_term_feerate);
        let mut tx = TxData::default();
        let mut outputs = vec![];
        for (amount, address_id) in amount_and_destination.iter() {
            outputs.push(Output {
                amount: *amount,
                address_id: *address_id,
            });
        }
        outputs.push(Output {
            amount: Amount::from_sat(drain.value),
            address_id: *change_addr,
        });
        tx.inputs = selected_coins
            .map(|o| Input {
                outpoint: o.outpoint,
            })
            .collect();
        tx.outputs = outputs;
        tx
    }

    fn participate_in_payjoin(
        &mut self,
        message_id: &MessageId,
        bulletin_board_id: &BulletinBoardId,
        payjoin: &PayjoinProposal,
        payment_obligation_id: &PaymentObligationId,
    ) -> TxId {
        // if we have a payment obligation then lets batch it with this payjoin
        let change_addr = self.new_address();
        let mut tx_template =
            self.construct_transaction_template(&[*payment_obligation_id], &change_addr);
        // "Lock" The inputs to this payjoin. These inputs can be spent if the payjoin is expired and our payment is due soon
        for input in tx_template.inputs.iter() {
            self.info_mut()
                .unconfirmed_txos_in_payjoins
                .insert(input.outpoint, *bulletin_board_id);
        }

        self.ack_transaction(&mut tx_template);
        tx_template.inputs.extend(payjoin.tx.inputs.iter().cloned());
        tx_template
            .outputs
            .extend(payjoin.tx.outputs.iter().cloned());
        tx_template
            .wallet_acks
            .extend(payjoin.tx.wallet_acks.iter().cloned());
        debug_assert!(tx_template.wallet_acks.contains(&self.id));

        let tx_id = self.spend_tx(tx_template);

        // Keep an index of what payment obligations are being handled in which payjoins
        self.info_mut()
            .received_payjoins
            .insert(*payment_obligation_id, *bulletin_board_id);

        // Mark the message as processed
        self.data_mut().messages_processed.insert(*message_id);

        tx_id
    }

    fn create_payjoin(
        &mut self,
        // Message id is serving as a proxy for payjoin id
        payment_obligation: &PaymentObligationId,
    ) -> MessageData {
        let payment_obligation_data = payment_obligation.with(self.sim).data().clone();
        let message_id = MessageId(self.sim.messages.len());
        let change_addr = self.new_address();
        let mut tx_template =
            self.construct_transaction_template(&[*payment_obligation], &change_addr);
        self.ack_transaction(&mut tx_template);
        debug_assert!(tx_template.wallet_acks.contains(&self.id));
        let payjoin_proposal = PayjoinProposal {
            tx: tx_template,
            valid_till: payment_obligation_data.deadline,
        };
        // "Lock" The inputs to this cospend. These inputs can be spent if the cospend is expired and our payment is due soon
        let bulletin_board_id = self.sim.create_bulletin_board();
        for input in payjoin_proposal.tx.inputs.iter() {
            self.info_mut()
                .unconfirmed_txos_in_payjoins
                .insert(input.outpoint, bulletin_board_id);
        }
        self.info_mut()
            .initiated_payjoins
            .insert(payment_obligation_data.id, bulletin_board_id);

        self.sim.add_message_to_bulletin_board(
            bulletin_board_id,
            BroadcastMessageType::InitiatePayjoin(payjoin_proposal),
        );
        MessageData {
            id: message_id,
            from: self.id,
            to: payment_obligation_data.to,
            message: MessageType::InitiatePayjoin(bulletin_board_id),
        }
    }

    pub(crate) fn wallet_view(&self) -> WalletView {
        let messages = self
            .sim
            .messages
            .iter()
            .filter(|message| !self.data().messages_processed.contains(&message.id))
            .filter(|message| message.from != self.id && message.to == self.id)
            .cloned()
            .collect::<Vec<_>>();
        let wallet_info = self.info();
        // Extract just the payjoin proposals we have not processed yet
        let payjoin_proposals = messages
            .iter()
            .filter_map(|message| match &message.message {
                MessageType::InitiatePayjoin(bulletin_board_id) => {
                    let payjoin_proposal = bulletin_board_id
                        .with(self.sim)
                        .data()
                        .messages
                        .iter()
                        .find_map(|message| match message {
                            BroadcastMessageType::InitiatePayjoin(payjoin_proposal) => {
                                Some(payjoin_proposal.clone())
                            }
                            _ => None,
                        });
                    if let Some(payjoin_proposal) = payjoin_proposal {
                        Some((message.id, *bulletin_board_id, payjoin_proposal.clone()))
                    } else {
                        None
                    }
                }
            })
            .collect::<Vec<_>>();

        // Filter out payment obligations that are already handled, or have an initiated/received payjoin
        // TODO: in the future where we want to support fallbacks we should not filter out initiated payjoins
        // Rather they are considered a seperate action.
        let initiated_po_ids: OrdSet<PaymentObligationId> =
            wallet_info.initiated_payjoins.keys().cloned().collect();
        let received_po_ids: OrdSet<PaymentObligationId> =
            wallet_info.received_payjoins.keys().cloned().collect();

        let payment_obligations = wallet_info
            .payment_obligations
            .clone()
            .iter()
            .filter(|po_id| {
                !wallet_info.handled_payment_obligations.contains(po_id)
                    && !initiated_po_ids.contains(po_id)
                    && !received_po_ids.contains(po_id)
            })
            .filter(|po| po.with(self.sim).data().reveal_time <= self.sim.current_timestep)
            .map(|po| po.with(self.sim).data().clone())
            .collect::<Vec<_>>();
        WalletView::new(
            payment_obligations,
            payjoin_proposals,
            self.sim.current_timestep,
        )
    }

    pub(crate) fn do_action(&'a mut self, action: &Action) {
        match action {
            Action::Wait => {}
            Action::UnilateralSpend(po) => {
                self.handle_payment_obligations(&[*po]);
            }
            Action::BatchSpend(po_ids) => {
                self.handle_payment_obligations(po_ids);
            }
            Action::InitiatePayjoin(po) => {
                let message = self.create_payjoin(po);
                self.sim.broadcast_message(message);
            }
            Action::RespondToPayjoin(payjoin_proposal, po, bulletin_board_id, message_id) => {
                let tx_id = self.participate_in_payjoin(
                    message_id,
                    bulletin_board_id,
                    payjoin_proposal,
                    po,
                );
                self.broadcast(vec![tx_id]);
            }
        }
    }

    pub(crate) fn wake_up(&'a mut self) {
        let scorer = &self.data().scorer;
        let wallet_view = self.wallet_view();
        let mut all_actions = Vec::new();
        for strategy in self.data().strategies.strategies.iter() {
            all_actions.extend(strategy.enumerate_candidate_actions(&wallet_view));
        }

        let action = all_actions
            .into_iter()
            .max_by_key(|action| scorer.score_action(action, self))
            .unwrap_or(Action::Wait);
        info!("Wallet id: {:?} chose action: {:?}", self.id, action);
        self.do_action(&action);
    }

    fn handle_payment_obligations(&'a mut self, payment_obligation_ids: &[PaymentObligationId]) {
        let change_addr = self.new_address();
        let mut tx_template =
            self.construct_transaction_template(payment_obligation_ids, &change_addr);
        self.ack_transaction(&mut tx_template);

        let tx_id = self.spend_tx(tx_template);
        self.info_mut()
            .txid_to_payment_obligation_ids
            .insert(tx_id, payment_obligation_ids.to_vec());
        self.broadcast(vec![tx_id]);
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
        pub(crate) reveal_time: TimeStep,
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
