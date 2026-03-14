use crate::{
    actions::{Action, CompositeScorer, CompositeStrategy, WalletView},
    blocks::BroadcastSetId,
    bulletin_board::{BroadcastMessageType, BulletinBoardId},
    message::{MessageId, MessageType, PayjoinProposal},
    script_type::ScriptType,
    tx_contruction::{
        MultiPartyPayjoinSession, SentBulletinBoardId, SentInputs, SentOutputs, SentReadyToSign,
        TxConstructionState,
    },
    Simulation, TimeStep,
};
use bdk_coin_select::{
    metrics::LowestFee, Candidate, ChangePolicy, CoinSelector, Drain, DrainWeights, Target,
    TargetFee, TargetOutputs, TR_DUST_RELAY_MIN_VALUE,
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
    pub(crate) script_type: ScriptType,
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
        /// Set of multi-party payjoin sessions that this wallet is participating in
        pub(crate) active_multi_party_payjoins: HashMap<BulletinBoardId, MultiPartyPayjoinSession>,
        /// UTXOs currently committed to interactive protocols (payjoins, multi-party sessions).
        pub(crate) used_utxos: OrdSet<Outpoint>,
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
        let locked = OrdSet::new();
        let utxos: Vec<OutputHandle<'a>> = self.unspent_coins(&locked).collect();
        let outputs_amounts = utxos.iter().map(|output| output.data().amount).sum();

        outputs_amounts
    }

    // TODO give utxo list as argument so that different variants can be used
    // TODO return change information
    pub(crate) fn select_coins(
        &self,
        target: Target,
        long_term_feerate: bitcoin::FeeRate,
        locked_inputs: &OrdSet<Outpoint>,
    ) -> (impl Iterator<Item = OutputHandle<'a>>, Drain) {
        // TODO change
        // TODO group by address
        let utxos: Vec<OutputHandle<'a>> = self.unspent_coins(locked_inputs).collect();

        let candidates: Vec<Candidate> = utxos
            .iter()
            .enumerate()
            .map(|(_, o)| Candidate {
                value: o.data().amount.to_sat(),
                weight: o.address().data().script_type.input_weight_wu(),
                input_count: 1,
                is_segwit: o.address().data().script_type.is_segwit(),
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

    fn unspent_coins<'s>(
        &'s self,
        locked_inputs: &'s OrdSet<Outpoint>,
    ) -> impl Iterator<Item = OutputHandle<'a>> + 's {
        let info = self.info();
        self.potentially_spendable_txos().filter(move |o| {
            !info.unconfirmed_spends.contains(&o.outpoint())
                && !locked_inputs.contains(&o.outpoint())
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
        self.sim.address_data.push(AddressData {
            wallet_id: self.id,
            script_type: self.data().script_type,
        });
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
        let output_weight_sum: u32 = amount_and_destination
            .iter()
            .map(|(_, address_id)| {
                address_id
                    .with(self.sim)
                    .data()
                    .script_type
                    .output_weight_wu()
            })
            .sum();
        let target = Target {
            fee: TargetFee {
                rate: bdk_coin_select::FeeRate::from_sat_per_vb(1.0),
                replace: None,
            },
            outputs: TargetOutputs {
                value_sum: amount,
                weight_sum: output_weight_sum,
                n_outputs: amount_and_destination.len(),
            },
        };
        let long_term_feerate = bitcoin::FeeRate::from_sat_per_vb(10).expect("valid fee rate");

        let locked_inputs = self.info().used_utxos.clone();

        let (selected_coins, drain) =
            self.handle()
                .select_coins(target, long_term_feerate, &locked_inputs);
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

    fn abandon_payjoin(&mut self, payment_obligation_id: &PaymentObligationId) {
        // Check initiated payjoins first, then received payjoins
        let bulletin_board_id = self
            .info()
            .initiated_payjoins
            .get(payment_obligation_id)
            .or_else(|| self.info().received_payjoins.get(payment_obligation_id))
            .cloned();

        if let Some(bb_id) = bulletin_board_id {
            // Remove UTXOs committed to this payjoin from used_utxos
            let outpoints_to_free: Vec<Outpoint> = self
                .info()
                .unconfirmed_txos_in_payjoins
                .iter()
                .filter(|(_, bid)| **bid == bb_id)
                .map(|(outpoint, _)| *outpoint)
                .collect();

            for outpoint in outpoints_to_free {
                self.info_mut().used_utxos.remove(&outpoint);
                self.info_mut()
                    .unconfirmed_txos_in_payjoins
                    .remove(&outpoint);
            }

            // Remove from initiated or received payjoins map
            self.info_mut()
                .initiated_payjoins
                .remove(payment_obligation_id);
            self.info_mut()
                .received_payjoins
                .remove(payment_obligation_id);
            return;
        }

        // Check active multi-party payjoin sessions
        let mp_bb_id = self
            .info()
            .active_multi_party_payjoins
            .iter()
            .find(|(_, session)| {
                session
                    .payment_obligation_ids
                    .contains(payment_obligation_id)
            })
            .map(|(bb_id, _)| *bb_id);

        if let Some(bb_id) = mp_bb_id {
            // Remove UTXOs committed to this session from used_utxos
            let outpoints_to_free: Vec<Outpoint> = self
                .info()
                .unconfirmed_txos_in_payjoins
                .iter()
                .filter(|(_, bid)| **bid == bb_id)
                .map(|(outpoint, _)| *outpoint)
                .collect();

            for outpoint in outpoints_to_free {
                self.info_mut().used_utxos.remove(&outpoint);
                self.info_mut()
                    .unconfirmed_txos_in_payjoins
                    .remove(&outpoint);
            }

            // Remove the entire multi-party session
            self.info_mut().active_multi_party_payjoins.remove(&bb_id);
        }
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
        // Mark UTXOs as used in this payjoin
        for input in tx_template.inputs.iter() {
            self.info_mut()
                .unconfirmed_txos_in_payjoins
                .insert(input.outpoint, *bulletin_board_id);
            self.info_mut().used_utxos.insert(input.outpoint);
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
    ) -> BulletinBoardId {
        let payment_obligation_data = payment_obligation.with(self.sim).data().clone();
        let change_addr = self.new_address();
        let mut tx_template =
            self.construct_transaction_template(&[payment_obligation_data.id], &change_addr);
        self.ack_transaction(&mut tx_template);
        debug_assert!(tx_template.wallet_acks.contains(&self.id));
        let payjoin_proposal = PayjoinProposal {
            tx: tx_template,
            valid_till: payment_obligation_data.deadline,
        };
        // Mark UTXOs as used in this cospend
        let bulletin_board_id = self.sim.create_bulletin_board();
        for input in payjoin_proposal.tx.inputs.iter() {
            self.info_mut()
                .unconfirmed_txos_in_payjoins
                .insert(input.outpoint, bulletin_board_id);
            self.info_mut().used_utxos.insert(input.outpoint);
        }
        self.info_mut()
            .initiated_payjoins
            .insert(payment_obligation_data.id, bulletin_board_id);

        self.sim.add_message_to_bulletin_board(
            bulletin_board_id,
            BroadcastMessageType::InitiatePayjoin(payjoin_proposal),
        );
        bulletin_board_id
    }

    fn create_multi_party_payjoin_session(&mut self, po_ids: &Vec<PaymentObligationId>) {
        if self.id.0 != 0 {
            return; // TODO For now the first wallet is the leader
        }
        // First we create the bulletin board
        let bulletin_board_id = self.sim.create_bulletin_board();
        // Then we invite all members to join
        for po_id in po_ids.iter() {
            let recv = po_id.with(&self.sim).data().to;
            self.sim.broadcast_message(
                recv,
                self.id,
                MessageType::InitiateMultiPartyPayjoin(bulletin_board_id),
            );
        }
        let change_addr = self.new_address();
        let tx_template = self.construct_transaction_template(po_ids, &change_addr);
        // Mark UTXOs as used in this multi-party payjoin session.
        for input in tx_template.inputs.iter() {
            self.info_mut()
                .unconfirmed_txos_in_payjoins
                .insert(input.outpoint, bulletin_board_id);
            self.info_mut().used_utxos.insert(input.outpoint);
        }
        let session = SentBulletinBoardId::new(self.sim, bulletin_board_id, tx_template.clone());

        session.send_inputs();
        info!("Sent inputs for multi party payjoin session");

        let session = MultiPartyPayjoinSession {
            payment_obligation_ids: po_ids.clone(),
            tx_template,
            state: TxConstructionState::SentInputs,
        };
        self.info_mut()
            .active_multi_party_payjoins
            .insert(bulletin_board_id, session);
    }

    fn participate_in_multi_party_payjoin(&mut self, bulletin_board_id: &BulletinBoardId) {
        let session = self
            .info()
            .active_multi_party_payjoins
            .get(bulletin_board_id)
            .unwrap();
        // TODO: construct tx template and contribute inputs / locking the po's and utxos to this session
        let state = session.state.clone();
        log::info!(
            "wallet id: {:?} participating in multi party payjoin session with state: {:?}",
            self.id,
            state
        );
        match state {
            TxConstructionState::SentBulletinBoardId => {
                let t = SentBulletinBoardId::new(
                    self.sim,
                    *bulletin_board_id,
                    session.tx_template.clone(),
                );
                t.send_inputs();
                let mut updated_session = session.clone();
                updated_session.state = TxConstructionState::SentInputs;
                self.info_mut()
                    .active_multi_party_payjoins
                    .insert(*bulletin_board_id, updated_session);
                log::info!(
                    "Sent inputs for multi party payjoin session with bulletin board id: {:?}",
                    bulletin_board_id
                );
                return;
            }
            TxConstructionState::SentInputs => {
                let t = SentInputs::new(self.sim, *bulletin_board_id, session.tx_template.clone());
                let res = t.have_enough_inputs();
                if let Some(_) = res {
                    let mut updated_session = session.clone();
                    updated_session.state = TxConstructionState::SentOutputs;
                    self.info_mut()
                        .active_multi_party_payjoins
                        .insert(*bulletin_board_id, updated_session);
                    log::info!(
                        "Sent outputs for multi party payjoin session with bulletin board id: {:?}",
                        bulletin_board_id
                    );
                }
                return;
            }
            TxConstructionState::SentOutputs => {
                let t = SentOutputs::new(self.sim, *bulletin_board_id, session.tx_template.clone());
                let res = t.have_enough_outputs();
                if let Some(_) = res {
                    let mut updated_session = session.clone();
                    updated_session.state = TxConstructionState::SentReadyToSign;
                    self.info_mut()
                        .active_multi_party_payjoins
                        .insert(*bulletin_board_id, updated_session);
                    log::info!(
                        "Sent ready to sign for multi party payjoin session with bulletin board id: {:?}",
                        bulletin_board_id
                    );
                }
                return;
            }
            TxConstructionState::SentReadyToSign => {
                let t = SentReadyToSign::new(self.sim, *bulletin_board_id);
                let res = t.have_enough_ready_to_sign();
                if let Some(tx) = res {
                    // TODO: only the leader should broadcast the tx right now
                    if self.id.0 != 0 {
                        return;
                    }
                    println!("tx: {:?}", tx);
                    let tx_id = self.spend_tx(tx);
                    log::info!(
                        "Multi party payjoin session successful with bulletin board id: {:?}",
                        bulletin_board_id
                    );
                    self.broadcast(std::iter::once(tx_id));
                    // Update session state to success
                    let mut updated_session = session.clone();
                    updated_session.state = TxConstructionState::Success(tx_id);
                    self.info_mut()
                        .active_multi_party_payjoins
                        .insert(*bulletin_board_id, updated_session);
                    log::info!(
                        "Multi party payjoin session successful with bulletin board id: {:?}",
                        bulletin_board_id
                    );
                }
                return;
            }
            TxConstructionState::Success(tx_id) => {
                log::info!("Multi party payjoin session successful: {:?}", tx_id);
                return;
            }
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
                _ => None,
            })
            .collect::<Vec<_>>();
        // New multi party payjoin sessions
        let new_multi_party_payjoins = messages
            .iter()
            .filter_map(|message| match &message.message {
                MessageType::InitiateMultiPartyPayjoin(bulletin_board_id) => {
                    Some((*bulletin_board_id, message.id))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        // Already active multi party payjoin sessions
        let active_mp_pj_sessions = wallet_info
            .active_multi_party_payjoins
            .iter()
            .filter_map(|(bulletin_board_id, session)| match &session.state {
                TxConstructionState::SentBulletinBoardId
                | TxConstructionState::SentInputs
                | TxConstructionState::SentOutputs
                | TxConstructionState::SentReadyToSign => Some(*bulletin_board_id),
                _ => None,
            })
            .collect::<Vec<_>>();

        // Filter out payment obligations that are already handled, or have an initiated/received payjoin.
        // The scorer evaluates all UTXOs (including used ones) by clearing used_utxos
        // in the simulation clone, so even UTXOs committed to payjoins can be picked for
        // more valuable actions when scoring.
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

        // POs currently in pending payjoins candidates for AbandonPayjoin
        let multi_party_po_ids: OrdSet<PaymentObligationId> = wallet_info
            .active_multi_party_payjoins
            .values()
            .flat_map(|session| session.payment_obligation_ids.iter().cloned())
            .collect();

        let payjoin_pending_pos = wallet_info
            .payment_obligations
            .iter()
            .filter(|po_id| {
                !wallet_info.handled_payment_obligations.contains(po_id)
                    && (initiated_po_ids.contains(po_id)
                        || received_po_ids.contains(po_id)
                        || multi_party_po_ids.contains(po_id))
            })
            .filter(|po| po.with(self.sim).data().reveal_time <= self.sim.current_timestep)
            .map(|po| po.with(self.sim).data().clone())
            .collect::<Vec<_>>();

        WalletView::new(
            payment_obligations,
            payjoin_proposals,
            new_multi_party_payjoins,
            active_mp_pj_sessions,
            self.sim.current_timestep,
            self.id,
            wallet_info.used_utxos.clone(),
            payjoin_pending_pos,
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
                let bulletin_board_id = self.create_payjoin(po);
                self.sim.broadcast_message(
                    po.with(&self.sim).data().to,
                    self.id,
                    MessageType::InitiatePayjoin(bulletin_board_id),
                );
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
            Action::InitiateMultiPartyPayjoin(po_ids) => {
                self.create_multi_party_payjoin_session(po_ids);
            }
            Action::ParticipateMultiPartyPayjoin((
                message_id,
                bulletin_board_id,
                payment_obligation_id,
            )) => {
                // Create new session -- assuming this doesnt exist already
                let change_addr = self.new_address();
                let tx_template =
                    self.construct_transaction_template(&[*payment_obligation_id], &change_addr);
                // Mark UTXOs as used in this multi-party payjoin session.
                for input in tx_template.inputs.iter() {
                    self.info_mut()
                        .unconfirmed_txos_in_payjoins
                        .insert(input.outpoint, *bulletin_board_id);
                    self.info_mut().used_utxos.insert(input.outpoint);
                }
                self.info_mut().active_multi_party_payjoins.insert(
                    *bulletin_board_id,
                    MultiPartyPayjoinSession {
                        payment_obligation_ids: vec![*payment_obligation_id],
                        tx_template,
                        // TODO: better state for someone who has not started the session yet
                        state: TxConstructionState::SentBulletinBoardId,
                    },
                );
                // Mark message as processed
                self.data_mut().messages_processed.insert(*message_id);
                self.participate_in_multi_party_payjoin(bulletin_board_id);
            }
            Action::ContinueParticipateMultiPartyPayjoin(bulletin_board_id) => {
                self.participate_in_multi_party_payjoin(bulletin_board_id);
            }
            Action::AbandonPayjoin(po_id) => {
                self.abandon_payjoin(po_id);
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

    pub(crate) fn broadcast(&mut self, txs: impl IntoIterator<Item = TxId>) -> BroadcastSetId {
        let mut wallet_info = self.info().clone();

        let txs = Vector::from_iter(txs);

        wallet_info.broadcast_transactions.append(txs.clone());

        // TODO refactor boilerplate for updating wallet ID
        let id = WalletInfoId(self.sim.wallet_info.len());
        self.sim.wallet_info.push(wallet_info);
        let data = self.data_mut();
        data.last_wallet_info_id = id;

        let res = self.sim.broadcast(txs);
        res.id
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
    pub(crate) script_type: ScriptType,
    // TODO internal
    // TODO silent payments
}, {});

impl From<AddressData> for InputWeightPrediction {
    fn from(data: AddressData) -> Self {
        data.script_type.input_weight_prediction()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SimulationBuilder;

    fn create_test_sim() -> Simulation {
        use crate::config::{ScorerConfig, WalletTypeConfig};
        SimulationBuilder::new(
            42,
            vec![WalletTypeConfig {
                name: "test".to_string(),
                count: 2,
                strategies: vec!["UnilateralSpender".to_string()],
                scorer: ScorerConfig {
                    initiate_payjoin_utility_factor: 1.0,
                    respond_to_payjoin_utility_factor: 1.0,
                    payment_obligation_utility_factor: 1.0,
                    multi_party_payjoin_utility_factor: 0.0,
                },
            }],
            100,
            1,
            0,
        )
        .build()
    }

    #[test]
    fn test_used_utxos_populated_on_payjoin() {
        let mut sim = create_test_sim();
        sim.build_universe();

        // Create a payment obligation from wallet 0 to wallet 1
        let po_id = PaymentObligationId(sim.payment_data.len());
        sim.payment_data.push(PaymentObligationData {
            id: po_id,
            amount: Amount::from_sat(100_000),
            from: WalletId(0),
            to: WalletId(1),
            deadline: TimeStep(50),
            reveal_time: TimeStep(0),
        });
        {
            let mut w = WalletId(0).with_mut(&mut sim);
            w.info_mut().payment_obligations.insert(po_id);
        }

        // Before creating a payjoin, used_utxos should be empty
        assert!(
            WalletId(0).with(&sim).info().used_utxos.is_empty(),
            "used_utxos should start empty"
        );

        // Create a payjoin — this should populate used_utxos
        let bb_id = {
            let mut w = WalletId(0).with_mut(&mut sim);
            w.create_payjoin(&po_id)
        };

        let info = WalletId(0).with(&sim).info();
        assert!(
            !info.used_utxos.is_empty(),
            "used_utxos should be populated after creating a payjoin"
        );
        // Every input committed to the payjoin should be in used_utxos
        for (outpoint, _) in info.unconfirmed_txos_in_payjoins.iter() {
            assert!(
                info.used_utxos.contains(outpoint),
                "outpoint {:?} should be in used_utxos",
                outpoint
            );
        }
    }

    #[test]
    fn test_coin_selection_excludes_used_utxos() {
        let mut sim = create_test_sim();
        sim.build_universe();

        // Mark all confirmed utxos as used
        let all_utxos: Vec<Outpoint> = WalletId(0)
            .with(&sim)
            .info()
            .confirmed_utxos
            .iter()
            .cloned()
            .collect();
        assert!(!all_utxos.is_empty(), "wallet should have UTXOs");

        let locked: OrdSet<Outpoint> = all_utxos.iter().cloned().collect();

        // With all UTXOs locked, unspent_coins should return nothing
        let available: Vec<_> = WalletId(0).with(&sim).unspent_coins(&locked).collect();
        assert!(
            available.is_empty(),
            "no coins should be available when all are locked"
        );

        // With empty lock set, all should be available
        let empty = OrdSet::new();
        let available: Vec<_> = WalletId(0).with(&sim).unspent_coins(&empty).collect();
        assert!(
            !available.is_empty(),
            "all coins should be available with empty lock set"
        );
    }

    #[test]
    fn test_abandon_payjoin_frees_utxos() {
        let mut sim = create_test_sim();
        sim.build_universe();

        // Create a payment obligation
        let po_id = PaymentObligationId(sim.payment_data.len());
        sim.payment_data.push(PaymentObligationData {
            id: po_id,
            amount: Amount::from_sat(100_000),
            from: WalletId(0),
            to: WalletId(1),
            deadline: TimeStep(50),
            reveal_time: TimeStep(0),
        });
        {
            let mut w = WalletId(0).with_mut(&mut sim);
            w.info_mut().payment_obligations.insert(po_id);
        }

        // Create a payjoin commits UTXOs
        let _bb_id = {
            let mut w = WalletId(0).with_mut(&mut sim);
            w.create_payjoin(&po_id)
        };

        // Verify UTXOs are locked
        let used_before: Vec<Outpoint> = WalletId(0)
            .with(&sim)
            .info()
            .used_utxos
            .iter()
            .cloned()
            .collect();
        assert!(
            !used_before.is_empty(),
            "UTXOs should be locked after payjoin"
        );

        // Abandon the payjoin
        {
            let mut w = WalletId(0).with_mut(&mut sim);
            w.abandon_payjoin(&po_id);
        }

        // Verify UTXOs are freed
        let info = WalletId(0).with(&sim).info();
        assert!(
            info.used_utxos.is_empty(),
            "used_utxos should be empty after abandoning payjoin"
        );
        assert!(
            !info.initiated_payjoins.contains_key(&po_id),
            "initiated_payjoins should not contain the abandoned PO"
        );
    }
}
