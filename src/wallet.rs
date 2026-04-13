use crate::{
    actions::{Action, CompositeScorer, CompositeStrategy, WalletView},
    blocks::BroadcastSetId,
    bulletin_board::BulletinBoardId,
    cospend::UtxoWithMetadata,
    message::{MessageId, MessageType},
    script_type::ScriptType,
    tx_contruction::{MultiPartyPayjoinSession, SentOutputs, SentReadyToSign, TxConstructionState},
    CoinSelectionStrategy, Simulation, TimeStep,
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
        /// Map of txids to the payment obligations that they are associated with
        /// Sim state should refrence this when updating wallet states after confirmation
        pub(crate) txid_to_payment_obligation_ids: HashMap<TxId, Vec<PaymentObligationId>>,

        /// Set of payment obligations that have been handled
        pub(crate) handled_payment_obligations: OrdSet<PaymentObligationId>,
        /// Set of multi-party payjoin sessions that this wallet is participating in
        pub(crate) active_multi_party_payjoins: HashMap<BulletinBoardId, MultiPartyPayjoinSession>,
        /// UTXOs registered in the order book by this wallet
        // TODO: this should be moved to wallet data
        pub(crate) registered_inputs: OrdSet<Outpoint>,
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
        select_all: bool,
        required_inputs: Option<&[Outpoint]>,
    ) -> (impl Iterator<Item = OutputHandle<'a>>, Drain) {
        // TODO change
        // TODO group by address
        let utxos: Vec<OutputHandle<'a>> = match required_inputs {
            Some(required) => self
                .unspent_coins()
                .filter(|o| required.contains(&o.outpoint()))
                .collect(),
            None => self.unspent_coins().collect(),
        };

        let candidates: Vec<Candidate> = utxos
            .iter()
            .map(|o| Candidate {
                value: o.data().amount.to_sat(),
                weight: o.address().data().script_type.input_weight_wu(),
                input_count: 1,
                is_segwit: o.address().data().script_type.is_segwit(),
            })
            .collect();

        let mut coin_selector = CoinSelector::new(&candidates);
        if select_all {
            coin_selector.select_all();
        }
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

        if let Err(err) = coin_selector.run_bnb(metric, 100_000) {
            // TODO: should be a error log
            warn!("BNB failed to find a solution: {}", err);

            coin_selector.select_until_target_met(target).expect(
                "coin selection should always succeed since payments consider budger lower bound",
            );
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
    pub(crate) fn data_mut(&mut self) -> &mut WalletData {
        &mut self.sim.wallet_data[self.id.0]
    }

    fn info_mut(&mut self) -> &mut WalletInfo {
        let last_wallet_info_id = self.data().last_wallet_info_id;
        &mut self.sim.wallet_info[last_wallet_info_id.0]
    }

    pub(crate) fn handle(&self) -> WalletHandle<'_> {
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
        select_all: bool,
        required_inputs: Option<&[Outpoint]>,
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

        let (selected_coins, drain) =
            self.handle()
                .select_coins(target, long_term_feerate, select_all, required_inputs);
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
            TxConstructionState::AcceptedProposal => {
                // Outputs are contributed via ContributeOutputsToSession; nothing to do here.
                log::info!(
                    "wallet id: {:?} in AcceptedProposal state for bb {:?}, waiting for ContributeOutputsToSession",
                    self.id,
                    bulletin_board_id
                );
            }
            TxConstructionState::SentOutputs => {
                let inputs = session.inputs.clone();
                let t = SentOutputs::new(
                    self.sim,
                    *bulletin_board_id,
                    TxData {
                        inputs,
                        outputs: vec![],
                    },
                );
                let res = t.have_enough_outputs();
                if res.is_some() {
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
            }
            TxConstructionState::SentReadyToSign => {
                let t = SentReadyToSign::new(self.sim, *bulletin_board_id);
                let res = t.have_enough_ready_to_sign();
                if let Some(tx) = res {
                    // Only the participant with the lowest wallet ID broadcasts to avoid
                    // duplicate transactions (new_tx deduplicates by content, so all
                    // participants would get the same TxId, violating broadcast/received invariants).
                    let min_participant_id = self.sim.bulletin_boards[bulletin_board_id.0]
                        .messages
                        .iter()
                        .filter_map(|msg| match msg {
                            crate::bulletin_board::BroadcastMessageType::ContributeInputs(op) => {
                                Some(op.with(self.sim).wallet().id)
                            }
                            _ => None,
                        })
                        .min();
                    let is_broadcaster = min_participant_id == Some(self.id);

                    let tx_id = if is_broadcaster {
                        let id = self.spend_tx(tx);
                        self.broadcast(std::iter::once(id));
                        let po_ids = session.payment_obligation_ids.clone();
                        self.info_mut()
                            .txid_to_payment_obligation_ids
                            .insert(id, po_ids);
                        Some(id)
                    } else {
                        None
                    };
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
            }
            TxConstructionState::Success(tx_id) => {
                log::info!("Multi party payjoin session successful: {:?}", tx_id);
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
        // New cospend proposals
        let new_cospend_proposals = messages
            .iter()
            .filter_map(|message| match &message.message {
                MessageType::ProposeCoSpend(bulletin_board_id) => {
                    Some((*bulletin_board_id, message.id))
                }
                MessageType::RegisterWalletInput(_) => None,
            })
            .collect::<Vec<_>>();
        // Already active multi party payjoin sessions (AcceptedProposal handled via ContributeOutputsToSession)
        let active_mp_pj_sessions = wallet_info
            .active_multi_party_payjoins
            .iter()
            .filter_map(|(bulletin_board_id, session)| match &session.state {
                TxConstructionState::SentOutputs | TxConstructionState::SentReadyToSign => {
                    Some(*bulletin_board_id)
                }
                _ => None,
            })
            .collect::<Vec<_>>();

        let payment_obligations = wallet_info
            .payment_obligations
            .clone()
            .iter()
            .filter(|po_id| !wallet_info.handled_payment_obligations.contains(po_id))
            // Do not offer paying again while a tx for this PO is already in the mempool;
            // handled_payment_obligations only updates on confirm, so without this the wallet
            // could build another tx reusing the same inputs (double-spend in `spends`).
            .filter(|po_id| {
                !wallet_info
                    .txid_to_payment_obligation_ids
                    .iter()
                    .any(|(txid, po_ids)| {
                        wallet_info.unconfirmed_transactions.contains(txid)
                            && po_ids.contains(po_id)
                    })
            })
            // Filter out POs that are not revealed yet
            .filter(|po| po.with(self.sim).data().reveal_time <= self.sim.current_timestep)
            .map(|po| po.with(self.sim).data().clone())
            .collect::<Vec<_>>();

        let utxos = self
            .handle()
            .unspent_coins()
            .map(|o| UtxoWithMetadata {
                outpoint: o.outpoint(),
                amount: o.data().amount,
                owner: self.id,
            })
            .collect::<Vec<_>>();
        let registered_inputs = self
            .info()
            .registered_inputs
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let orderbook_utxos = self.sim.get_orderbook_utxos();
        let pending_interests = self.sim.cospend_interests.clone();

        WalletView::new(
            payment_obligations,
            new_cospend_proposals,
            active_mp_pj_sessions,
            utxos,
            registered_inputs,
            orderbook_utxos,
            pending_interests,
        )
    }

    fn register_input(&mut self, outpoint: &Outpoint) {
        if self.info().registered_inputs.contains(outpoint) {
            return;
        }
        self.info_mut().registered_inputs.insert(*outpoint);
        info!(
            "Wallet {:?} registered input {:?} in order book",
            self.id, outpoint
        );
    }

    pub(crate) fn do_action(&'a mut self, action: &Action) {
        match action {
            Action::Wait => {}
            // TODO: the next 3 actions can be folded into one spend action, param'd off # of po's and coin selection strategy. All of them are unilateral
            Action::UnilateralPayments(po_ids, coin_selection_strategy) => {
                self.handle_payment_obligations(
                    po_ids,
                    matches!(coin_selection_strategy, CoinSelectionStrategy::SpendAll),
                );
            }
            Action::AcceptCospendProposal((message_id, bulletin_board_id)) => {
                // Aggregator already pre-filled all inputs on the bulletin board.
                // Find our own inputs from the bulletin board's ContributeInputs messages.
                use crate::bulletin_board::BroadcastMessageType;
                let my_inputs: Vec<Input> = self.sim.bulletin_boards[bulletin_board_id.0]
                    .messages
                    .iter()
                    .filter_map(|msg| match msg {
                        BroadcastMessageType::ContributeInputs(op) => Some(op),
                        _ => None,
                    })
                    .filter(|op| {
                        self.info().confirmed_utxos.contains(op)
                            && !self.info().unconfirmed_spends.contains(op)
                    })
                    .map(|op| Input { outpoint: *op })
                    .collect();
                self.info_mut().active_multi_party_payjoins.insert(
                    *bulletin_board_id,
                    MultiPartyPayjoinSession {
                        payment_obligation_ids: vec![],
                        inputs: my_inputs,
                        state: TxConstructionState::AcceptedProposal,
                    },
                );
                self.data_mut().messages_processed.insert(*message_id);
            }
            Action::ProposeCospend(interests) => {
                for interest in interests {
                    self.sim.cospend_interests.push(interest.clone());
                }
            }
            Action::CreateAggregateProposal(interests) => {
                let bb_id = self.sim.create_bulletin_board();
                // Collect unique (outpoint, owner) pairs to avoid double-spending
                // when multiple interests share the same UTXO (e.g. taker proposes
                // the same UTXO against multiple makers).
                let mut seen_outpoints = std::collections::HashSet::new();
                let unique_utxos: Vec<_> = interests
                    .iter()
                    .flat_map(|i| i.utxos.iter())
                    .filter(|u| seen_outpoints.insert(u.outpoint))
                    // Skip UTXOs that have been spent since the interest was recorded.
                    // Interests are non-committal and may go stale between proposal and
                    // aggregation (e.g. the owner spent the coin unilaterally in the same
                    // tick before the aggregator ran).
                    .filter(|u| {
                        let info = &self.sim.wallet_info
                            [self.sim.wallet_data[u.owner.0].last_wallet_info_id.0];
                        info.confirmed_utxos.contains(&u.outpoint)
                            && !info.unconfirmed_spends.contains(&u.outpoint)
                    })
                    .collect();
                // Pre-fill all unique inputs on the bulletin board
                for u in &unique_utxos {
                    self.sim.add_message_to_bulletin_board(
                        bb_id,
                        crate::bulletin_board::BroadcastMessageType::ContributeInputs(u.outpoint),
                    );
                }
                // Invite each unique participant once
                let mut invited = std::collections::HashSet::new();
                for u in &unique_utxos {
                    if invited.insert(u.owner) {
                        self.sim.broadcast_message(
                            u.owner,
                            self.id,
                            MessageType::ProposeCoSpend(bb_id),
                        );
                    }
                }
                // Clear the handled interests
                self.sim
                    .cospend_interests
                    .retain(|i| !interests.contains(i));
            }
            Action::ContributeOutputsToSession(bulletin_board_id, po_ids) => {
                let session_inputs = self
                    .info()
                    .active_multi_party_payjoins
                    .get(bulletin_board_id)
                    .unwrap()
                    .inputs
                    .clone();
                let input_outpoints: Vec<Outpoint> =
                    session_inputs.iter().map(|i| i.outpoint).collect();
                let required = if input_outpoints.is_empty() {
                    None
                } else {
                    Some(input_outpoints.as_slice())
                };
                let change_addr = self.new_address();
                let full_template =
                    self.construct_transaction_template(po_ids, &change_addr, false, required);
                // Inputs are already pre-filled by the aggregator; broadcast our outputs directly.
                use crate::bulletin_board::BroadcastMessageType;
                for output in full_template.outputs.iter() {
                    self.sim.add_message_to_bulletin_board(
                        *bulletin_board_id,
                        BroadcastMessageType::ContributeOutputs(output.clone()),
                    );
                }
                let session = self
                    .info_mut()
                    .active_multi_party_payjoins
                    .get_mut(bulletin_board_id)
                    .unwrap();
                session.payment_obligation_ids = po_ids.clone();
                session.state = TxConstructionState::SentOutputs;
            }
            Action::ContinueParticipateInCospend(bulletin_board_id) => {
                self.participate_in_multi_party_payjoin(bulletin_board_id);
            }
            Action::RegisterInput(outpoint) => {
                self.register_input(outpoint);
            }
        }
    }

    pub(crate) fn wake_up(&'a mut self) {
        let scorer = &self.data().scorer;
        let wallet_view = self.wallet_view();
        // Clone strategies to allow passing &self to enumerate_candidate_actions
        // without conflicting with the borrow on strategies.strategies
        let strategies = self.data().strategies.clone();
        let mut all_actions = Vec::new();
        for strategy in strategies.strategies.iter() {
            all_actions.extend(strategy.enumerate_candidate_actions(&wallet_view, self));
        }

        let action = all_actions
            .into_iter()
            .min_by_key(|action| scorer.action_cost(action, self))
            .unwrap_or(Action::Wait);
        info!("Wallet id: {:?} chose action: {:?}", self.id, action);
        self.do_action(&action);
    }

    fn handle_payment_obligations(
        &'a mut self,
        payment_obligation_ids: &[PaymentObligationId],
        select_all_utxos: bool,
    ) {
        let change_addr = self.new_address();
        let tx_template = self.construct_transaction_template(
            payment_obligation_ids,
            &change_addr,
            select_all_utxos,
            None,
        );
        let tx_id = self.spend_tx(tx_template);
        self.info_mut()
            .txid_to_payment_obligation_ids
            .insert(tx_id, payment_obligation_ids.to_vec());
        self.broadcast(vec![tx_id]);
    }

    // TODO: refactor this? Do we event need this?
    fn spend_tx(&mut self, txdata: TxData) -> TxId {
        // TODO: assert this is my obligation
        let spend = self
            .new_tx(|tx, _| {
                tx.inputs = txdata.inputs;
                tx.outputs = txdata.outputs;
            })
            .id;

        spend
    }

    pub(crate) fn new_tx<F>(&mut self, build: F) -> TxHandle<'_>
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
