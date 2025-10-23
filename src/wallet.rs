use crate::{
    blocks::{BroadcastSetHandleMut, BroadcastSetId},
    message::{MessageData, MessageId, MessageType},
    Epoch, Simulation,
};
use bdk_coin_select::{
    metrics::LowestFee, Candidate, ChangePolicy, CoinSelector, Drain, DrainWeights, Target,
    TargetFee, TargetOutputs, TR_DUST_RELAY_MIN_VALUE, TR_KEYSPEND_TXIN_WEIGHT,
};
use bitcoin::{transaction::InputWeightPrediction, Amount};
use im::{OrdSet, Vector};

use crate::transaction::*;

#[derive(Debug, PartialEq, Eq, Clone)]
pub(crate) struct CollabroationSpend {
    pub(crate) payment_obligation_id: PaymentObligationId,
    pub(crate) messages_sent: Vec<MessageId>,
    pub(crate) messages_received: Vec<MessageId>,
    pub(crate) tx_template: TxData,
    pub(crate) counter_party: WalletId,
}

define_entity_mut_updatable!(
    Wallet,
    {
        pub(crate) id: WalletId,
        pub(crate) addresses: Vec<AddressId>,         // TODO split into internal/external?
        pub(crate) own_transactions: Vec<TxId>,       // transactions originating from this wallet
        pub(crate) last_wallet_info_id: WalletInfoId, // Monotone
        pub(crate) seen_messages: OrdSet<MessageId>,
        pub(crate) handled_payment_obligations: OrdSet<PaymentObligationId>,
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
        pub(crate) collabroation_spends: Vec<CollabroationSpend>,
    }
);

impl<'a> WalletHandle<'a> {
    pub(crate) fn data(&self) -> &'a WalletData {
        &self.sim.wallet_data[self.id.0]
    }

    pub(crate) fn info(&self) -> &'a WalletInfo {
        &self.sim.wallet_info[self.data().last_wallet_info_id.0]
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
            .map(|(i, o)| Candidate {
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
        self.potentially_spendable_txos()
            .filter(|o| !self.info().unconfirmed_spends.contains(&o.outpoint()))
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

    pub(crate) fn new_address(&mut self) -> AddressId {
        let id = AddressId(self.sim.address_data.len());
        self.sim.wallet_data[self.id.0].addresses.push(id);
        self.sim
            .address_data
            .push(AddressData { wallet_id: self.id });
        id
    }

    pub(crate) fn handle_message(&'a mut self, message: MessageData) {
        if message.to == self.id {
            if self.sim.wallet_data[self.id.0]
                .seen_messages
                .contains(&message.id)
            {
                return;
            }
            self.sim.wallet_data[self.id.0]
                .seen_messages
                .insert(message.id);

            match message.message {
                MessageType::RegisterInputs(_) => {
                    let last_wallet_info_id = self.data().last_wallet_info_id;
                    let last_wallet_info = &mut self.sim.wallet_info[last_wallet_info_id.0];
                    // Check if our counter party is responding to our input registration
                    if let Some(collabroation_spend) = last_wallet_info
                        .collabroation_spends
                        .iter_mut()
                        .find(|c| c.counter_party == message.from)
                    {
                        // Add this message to the collabroation spend
                        collabroation_spend.messages_received.push(message.id);
                        // Register out outputs
                        let message = MessageData {
                            id: MessageId(self.sim.messages.len()),
                            message: MessageType::RegisterOutputs(
                                collabroation_spend.tx_template.outputs.clone(),
                            ),
                            from: self.id,
                            to: message.from,
                            previous_message: Some(message.id),
                        };
                        collabroation_spend.messages_sent.push(message.id);
                        self.sim.broadcast_message(message.clone());
                        return;
                    }

                    // TODO check that these are not my inputs

                    // Handling payment obligations will register our inputs if we have a payment obligation that is not due
                    // TODO: Employ some strategy to determine the best payment obligation to collaborate on
                    self.handle_payment_obligations();
                }
                MessageType::RegisterOutputs(_) => {
                    let last_wallet_info_id = self.data().last_wallet_info_id;
                    let last_wallet_info = &mut self.sim.wallet_info[last_wallet_info_id.0];
                    // Check if our counter party is responding to our input registration
                    if let Some(collabroation_spend) = last_wallet_info
                        .collabroation_spends
                        .iter_mut()
                        .find(|c| c.counter_party == message.from)
                    {
                        // Add this message to the collabroation spend
                        collabroation_spend.messages_received.push(message.id);
                        // If we have received 2 messages and sent 2 then we can join everything together and broadcast the transaction
                        if collabroation_spend.messages_sent.len() == 2
                            && collabroation_spend.messages_received.len() == 2
                        {
                            let mut tx_template = collabroation_spend.tx_template.clone();
                            for messages_received in collabroation_spend.messages_received.iter() {
                                let message =
                                    self.sim.messages[messages_received.0].message.clone();
                                match message {
                                    MessageType::RegisterOutputs(outs) => {
                                        tx_template.outputs.extend(outs.iter());
                                    }
                                    MessageType::RegisterInputs(ins) => {
                                        tx_template.inputs.extend(ins.iter());
                                    }
                                }
                            }

                            let txid = self.spend_tx(tx_template);
                            self.broadcast(std::iter::once(txid));
                            return;
                        }

                        // Ack the register outputs message by registering our outputs
                        let message = MessageData {
                            id: MessageId(self.sim.messages.len()),
                            message: MessageType::RegisterOutputs(
                                collabroation_spend.tx_template.outputs.clone(),
                            ),
                            from: self.id,
                            to: message.from,
                            previous_message: Some(message.id),
                        };
                        collabroation_spend.messages_sent.push(message.id);
                        self.sim.broadcast_message(message.clone());
                        return;
                    }
                }
            }
        }
        // TODO: else panic? something is wrong with the simulation?
    }

    /// stateless utility function to construct a transaction for a given payment obligation
    fn construct_payment_transaction(
        &self,
        obligation: PaymentObligationData,
        change_addr: AddressId,
        to_address: AddressId,
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
                address_id: to_address,
            },
            Output {
                amount: Amount::from_sat(drain.value),
                address_id: change_addr,
            },
        ];
        tx
    }

    fn effective_payment_obligations(&self) -> OrdSet<PaymentObligationId> {
        self.info()
            .payment_obligations
            .clone()
            .difference(self.data().handled_payment_obligations.clone())
    }

    pub(crate) fn handle_payment_obligations(&mut self) -> Option<TxId> {
        let payment_obligations = self.effective_payment_obligations();

        if payment_obligations.is_empty() {
            return None;
        }
        let payment_obligation_id = payment_obligations
            .iter()
            .next()
            .expect("payment obligations should not be empty");
        let payment_obligation = payment_obligation_id.with(self.sim).data().clone();
        let change_addr = self.new_address();
        let to_wallet = payment_obligation.to;
        let to_address = to_wallet.with_mut(self.sim).new_address();
        let tx_template =
            self.construct_payment_transaction(payment_obligation.clone(), change_addr, to_address);

        let time_left = payment_obligation.deadline.0 as i32 - self.sim.current_epoch.0 as i32;
        // TODO: this should be configurable. Right now the wallets are very impatient
        if time_left <= 2 {
            let tx_id = self.spend_tx(tx_template);
            self.data_mut()
                .handled_payment_obligations
                .insert(*payment_obligation_id);
            return Some(tx_id);
        }
        // Try to register our inputs if we already have not
        let last_wallet_info_id = self.data().last_wallet_info_id;
        let mut last_wallet_info = self.sim.wallet_info[last_wallet_info_id.0].clone();
        // TODO: refactor to use a map?
        // TODO: currently we are only collaborating once per sim run
        if last_wallet_info.collabroation_spends.is_empty() {
            // For now lets reach out to recepient to batch this payment
            // Later we can reach out to the entire peer graph to form a coalition
            let message = MessageData {
                id: MessageId(self.sim.messages.len()),
                message: MessageType::RegisterInputs(tx_template.inputs.clone()),
                from: self.id,
                to: to_wallet,
                previous_message: None,
            };
            self.sim.broadcast_message(message.clone());
            let collabroation_spend = CollabroationSpend {
                payment_obligation_id: *payment_obligation_id,
                messages_sent: vec![message.id],
                messages_received: vec![],
                counter_party: to_wallet,
                tx_template: tx_template.clone(),
            };
            last_wallet_info
                .collabroation_spends
                .push(collabroation_spend);
            self.sim.wallet_info[last_wallet_info_id.0] = last_wallet_info;
        }

        None
    }

    // TODO: refactor this? Do we event need this?
    pub(crate) fn spend_tx(&mut self, txdata: TxData) -> TxId {
        // TODO: assert this is my obligation
        let spend = self
            .new_tx(|tx, _| {
                tx.inputs = txdata.inputs;
                tx.outputs = txdata.outputs;
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
        pub(crate) deadline: Epoch,
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
