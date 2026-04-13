use std::{iter::Sum, ops::Add};

use log::debug;

use crate::{
    bulletin_board::BulletinBoardId,
    cospend::{CospendInterest, UtxoWithMetadata},
    message::MessageId,
    transaction::Outpoint,
    tx_contruction::TxConstructionState,
    wallet::{PaymentObligationData, PaymentObligationId, WalletHandleMut},
    CoinSelectionStrategy, Simulation, TimeStep,
};

fn piecewise_linear(x: f64, points: &[(f64, f64)]) -> f64 {
    assert!(points.len() >= 2, "need at least two points");

    // Clamp on either end of the points
    if x <= points[0].0 {
        return points[0].1;
    }

    let last = points.len() - 1;
    if x >= points[last].0 {
        return points[last].1;
    }

    // Find segment [x_i, x_{i+1}] containing x
    for window in points.windows(2) {
        let (x0, y0) = window[0];
        let (x1, y1) = window[1];

        if x >= x0 && x <= x1 {
            let t = (x - x0) / (x1 - x0);
            return y0 + t * (y1 - y0);
        }
    }

    unreachable!("x did not fall into any segment; are points sorted?");
}

/// An Action a wallet can perform
#[derive(Debug)]
pub(crate) enum Action {
    /// Spend a payment obligation unilaterally
    UnilateralPayments(Vec<PaymentObligationId>, CoinSelectionStrategy),
    /// Accept a cospend invitation
    AcceptCospendProposal((MessageId, BulletinBoardId)),
    /// Contribute outputs to a cospend session that is waiting for them
    ContributeOutputsToSession(BulletinBoardId, Vec<PaymentObligationId>),
    /// Continue to participate in a multi-party payjoin
    ContinueParticipateInCospend(BulletinBoardId),
    /// Taker records non-committal interest in cospending with each orderbook UTXO
    ProposeCospend(Vec<CospendInterest>),
    /// Aggregator creates an aggregate session from pending interests
    CreateAggregateProposal(Vec<CospendInterest>),
    /// Register a single UTXO in the order book (maker action)
    RegisterInput(Vec<Outpoint>),
    /// Do nothing. There may be better oppurtunities to spend a payment obligation or participate in a payjoin.
    Wait,
}

/// Predicted diff of wallet state resulting from taking an action.
/// Each field corresponds to one category of state change.
#[derive(Debug, Default)]
pub(crate) struct PredictedOutcome {
    /// Payment obligations fulfilled by this action
    pub(crate) payment_obligations_handled: Vec<PaymentObligationId>,
    /// Payment obligations missed/abandoned
    pub(crate) payment_obligations_abandoned: Vec<PaymentObligationId>,
    /// UTXOs consumed as inputs
    pub(crate) utxos_spent: Vec<Outpoint>,
    /// Outpoints of outputs created. TODO: and what feerate interval they were created in?
    pub(crate) outputs_created: Vec<Outpoint>,
    /// UTXOs newly published on the order book
    pub(crate) order_book_registrations: Vec<Outpoint>,
    /// New multi-party cospend sessions added for this wallet
    pub(crate) cospend_proposals_accepted: Vec<BulletinBoardId>,
    /// Session state machine transitions
    pub(crate) session_state_updates: Vec<(BulletinBoardId, TxConstructionState)>,
}

impl PredictedOutcome {
    pub(crate) fn cost(
        &self,
        scorer: &CompositeScorer,
        sim: &Simulation,
        current_timestep: TimeStep,
    ) -> ActionCost {
        let mut cost = ActionCost(INHERENT_ACTION_COST);
        for po_id in &self.payment_obligations_handled {
            let po = po_id.with(sim).data();
            let time_left = po.deadline.0 as i32 - current_timestep.0 as i32;
            // Utility of 2*weight at deadline easily
            // exceeds the base cost, making near-deadline payments cheaper than waiting.
            let base_cost = po.amount.to_float_in(bitcoin::Denomination::Bitcoin);
            let points = [
                (0.0, 2.0 * scorer.payment_obligation_weight),
                (2.0, scorer.payment_obligation_weight),
                (5.0, 0.0),
            ];
            let utility = piecewise_linear(time_left as f64, &points);
            debug!(
                "PaymentObligationHandled cost: base={} utility={}",
                base_cost, utility
            );
            cost = cost + ActionCost(base_cost - utility);
        }
        // TODO: cost from payment_obligations_abandoned (missed deadline penalty)
        // TODO: cost from utxos_spent (UTXO fragmentation / privacy loss)
        // TODO: cost from outputs_created (fee rate efficiency)
        // TODO: cost from order_book_registrations (coordination value)
        // TODO: cost from cospend_proposals_accepted / session_state_updates (privacy gain)
        cost
    }
}

/// State of the wallet that can be used to potential enumerate actions
#[derive(Debug)]
pub(crate) struct WalletView {
    payment_obligations: Vec<PaymentObligationData>,
    active_cospends: Vec<BulletinBoardId>,
    cospend_proposals: Vec<(BulletinBoardId, MessageId)>,
    utxos: Vec<UtxoWithMetadata>,
    registered_inputs: Vec<Outpoint>,
    /// UTXOs currently registered on the order book (for taker to propose to)
    orderbook_utxos: Vec<UtxoWithMetadata>,
    /// Pending cospend interests (for aggregator to batch into sessions)
    pending_interests: Vec<CospendInterest>,
}

impl WalletView {
    pub(crate) fn new(
        payment_obligations: Vec<PaymentObligationData>,
        cospend_proposals: Vec<(BulletinBoardId, MessageId)>,
        active_cospends: Vec<BulletinBoardId>,
        utxos: Vec<UtxoWithMetadata>,
        registered_inputs: Vec<Outpoint>,
        orderbook_utxos: Vec<UtxoWithMetadata>,
        pending_interests: Vec<CospendInterest>,
    ) -> Self {
        Self {
            payment_obligations,
            active_cospends,
            cospend_proposals,
            utxos,
            registered_inputs,
            orderbook_utxos,
            pending_interests,
        }
    }
}
fn simulate_one_action(wallet_handle: &WalletHandleMut, action: &Action) -> PredictedOutcome {
    let old_info = wallet_handle.info().clone();

    let wallet_id = wallet_handle.data().id;
    let mut sim = wallet_handle.sim.clone();
    wallet_id.with_mut(&mut sim).do_action(action);
    let new_info = wallet_id.with(&sim).info().clone();

    // POs handled: derived from action since confirmation is deferred to block
    let payment_obligations_handled: Vec<PaymentObligationId> = match action {
        Action::UnilateralPayments(po_ids, _) => po_ids.clone(),
        Action::ContributeOutputsToSession(_, po_ids) => po_ids.clone(),
        _ => vec![],
    };

    // UTXOs spent: new entries in unconfirmed_spends
    let utxos_spent: Vec<Outpoint> = new_info
        .unconfirmed_spends
        .iter()
        .filter(|op| !old_info.unconfirmed_spends.contains(op))
        .copied()
        .collect();

    // Outputs created: new entries in unconfirmed_txos
    let outputs_created: Vec<Outpoint> = new_info
        .unconfirmed_txos
        .iter()
        .filter(|op| !old_info.unconfirmed_txos.contains(op))
        .copied()
        .collect();

    // Order book registrations: new entries in registered_inputs
    let order_book_registrations: Vec<Outpoint> = new_info
        .registered_inputs
        .iter()
        .filter(|op| !old_info.registered_inputs.contains(op))
        .copied()
        .collect();

    // Cospend proposals accepted: new sessions where this wallet is a maker
    let cospend_proposals_accepted: Vec<BulletinBoardId> = new_info
        .active_multi_party_payjoins
        .iter()
        .filter(|(bb, _session)| !old_info.active_multi_party_payjoins.contains_key(bb))
        .map(|(bb, _)| *bb)
        .collect();

    // Session state transitions: boards present in both old and new with a different state
    let session_state_updates: Vec<(BulletinBoardId, TxConstructionState)> = new_info
        .active_multi_party_payjoins
        .iter()
        .filter_map(|(bb, new_session)| {
            old_info
                .active_multi_party_payjoins
                .get(bb)
                .and_then(|old_session| {
                    if old_session.state != new_session.state {
                        Some((*bb, new_session.state.clone()))
                    } else {
                        None
                    }
                })
        })
        .collect();

    PredictedOutcome {
        payment_obligations_handled,
        payment_obligations_abandoned: vec![],
        utxos_spent,
        outputs_created,
        order_book_registrations,
        cospend_proposals_accepted,
        session_state_updates,
    }
}

/// Strategies will pick one action to minimize their cost
/// TODO: Strategies should be composible. They should enform the action decision space scoring and doing actions should be handling by something else that has composed multiple strategies.
pub(crate) trait Strategy: std::fmt::Debug {
    fn enumerate_candidate_actions(
        &self,
        state: &WalletView,
        wallet: &WalletHandleMut,
    ) -> Vec<Action>;
    fn clone_box(&self) -> Box<dyn Strategy>;
}

#[derive(Debug, PartialEq, PartialOrd)]
// TODO: this should just be bitcoin::Amount
pub(crate) struct ActionCost(f64);

// Flat base cost applied to any action, including waiting.
const INHERENT_ACTION_COST: f64 = 0.0;

impl Sum for ActionCost {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        Self(iter.map(|s| s.0).sum())
    }
}

impl Eq for ActionCost {}

impl Ord for ActionCost {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        assert!(!self.0.is_nan() && !other.0.is_nan());
        self.0.partial_cmp(&other.0).expect("Checked for NaNs")
    }
}

impl Add for ActionCost {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct UnilateralSpender;

impl Strategy for UnilateralSpender {
    /// The decision space of the unilateral spender is the set of all payment obligations
    fn enumerate_candidate_actions(
        &self,
        state: &WalletView,
        _wallet: &WalletHandleMut,
    ) -> Vec<Action> {
        if state.payment_obligations.is_empty() {
            return vec![Action::Wait];
        }
        let mut actions = vec![];
        for po in state.payment_obligations.iter() {
            actions.push(Action::UnilateralPayments(
                vec![po.id],
                CoinSelectionStrategy::BNB,
            ));
        }
        actions
    }

    fn clone_box(&self) -> Box<dyn Strategy> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Consolidator;

impl Strategy for Consolidator {
    /// Always uses SpendAll when paying.
    /// trade-off. Fee savings from reducing UTXO fragmentation are captured when fee_savings_weight > 0.
    fn enumerate_candidate_actions(
        &self,
        state: &WalletView,
        _wallet: &WalletHandleMut,
    ) -> Vec<Action> {
        let mut actions = Vec::new();
        for po in state.payment_obligations.iter() {
            actions.push(Action::UnilateralPayments(
                vec![po.id],
                CoinSelectionStrategy::SpendAll,
            ));
        }
        actions.push(Action::Wait);
        actions
    }

    fn clone_box(&self) -> Box<dyn Strategy> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct BatchSpender;

impl Strategy for BatchSpender {
    fn enumerate_candidate_actions(
        &self,
        state: &WalletView,
        _wallet: &WalletHandleMut,
    ) -> Vec<Action> {
        if state.payment_obligations.is_empty() {
            return vec![Action::Wait];
        }
        // TODO: we may need to consider different partitioning strategies for the batch spend
        let payment_obligation_ids: Vec<PaymentObligationId> =
            state.payment_obligations.iter().map(|po| po.id).collect();
        vec![Action::UnilateralPayments(
            payment_obligation_ids,
            CoinSelectionStrategy::BNB,
        )]
    }

    fn clone_box(&self) -> Box<dyn Strategy> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MakerStrategy;

impl Strategy for MakerStrategy {
    fn enumerate_candidate_actions(
        &self,
        state: &WalletView,
        wallet: &WalletHandleMut,
    ) -> Vec<Action> {
        let mut actions = vec![];

        // Continue to participate in active sessions
        for bulletin_board_id in state.active_cospends.iter() {
            actions.push(Action::ContinueParticipateInCospend(*bulletin_board_id));
        }

        // Accept new invitations. TODO: in the future makers will be have certain preferences for which invitations to accept.
        if let Some((bulletin_board_id, message_id)) = state.cospend_proposals.iter().next() {
            if state.active_cospends.is_empty() {
                actions.push(Action::AcceptCospendProposal((
                    *message_id,
                    *bulletin_board_id,
                )));
            }
        }

        // Contribute outputs to sessions that are waiting for them (SentInputs state)
        for (bb_id, session) in wallet.info().active_multi_party_payjoins.iter() {
            if session.state == TxConstructionState::AcceptedProposal {
                for po in state.payment_obligations.iter() {
                    actions.push(Action::ContributeOutputsToSession(*bb_id, vec![po.id]));
                }
            }
        }

        // Only figure out what to register when truly idle: no pending invitations, no active
        // sessions (except completed ones). The wallet's session map is the authoritative source.
        let has_active_sessions = !state.active_cospends.is_empty()
            || wallet
                .info()
                .active_multi_party_payjoins
                .values()
                .any(|s| !matches!(s.state, TxConstructionState::Success(_)));
        let unilateral_actions = if state.cospend_proposals.is_empty() && !has_active_sessions {
            UnilateralSpender.enumerate_candidate_actions(state, wallet)
        } else {
            vec![]
        };
        let per_action_spent: Vec<std::collections::HashSet<Outpoint>> = unilateral_actions
            .iter()
            .filter(|a| matches!(a, Action::UnilateralPayments(_, _)))
            .map(|action| {
                simulate_one_action(wallet, action)
                    .utxos_spent
                    .into_iter()
                    .collect()
            })
            .collect();
        let common_inputs: Vec<Outpoint> = per_action_spent
            .iter()
            .skip(1)
            .fold(
                per_action_spent.first().cloned().unwrap_or_default(),
                |acc, s| acc.intersection(s).copied().collect(),
            )
            .iter()
            .filter(|o| !state.registered_inputs.contains(o))
            .copied()
            .collect();
        if !common_inputs.is_empty() {
            actions.push(Action::RegisterInput(common_inputs));
        }
        if actions.is_empty() {
            actions.push(Action::Wait);
        }
        actions
    }

    fn clone_box(&self) -> Box<dyn Strategy> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct TakerStrategy;

impl Strategy for TakerStrategy {
    fn enumerate_candidate_actions(
        &self,
        state: &WalletView,
        wallet: &WalletHandleMut,
    ) -> Vec<Action> {
        let mut actions = vec![];

        // Contribute outputs to sessions awaiting them (SentInputs state)
        for (bb_id, session) in wallet.info().active_multi_party_payjoins.iter() {
            if session.state == TxConstructionState::AcceptedProposal {
                for po in state.payment_obligations.iter() {
                    actions.push(Action::ContributeOutputsToSession(*bb_id, vec![po.id]));
                }
            }
        }

        // Continue active sessions in later states
        for bulletin_board_id in state.active_cospends.iter() {
            actions.push(Action::ContinueParticipateInCospend(*bulletin_board_id));
        }

        if !actions.is_empty() {
            return actions;
        }

        // Accept any pending invitations from the aggregator before proposing new ones.
        if let Some((bulletin_board_id, message_id)) = state.cospend_proposals.iter().next() {
            return vec![Action::AcceptCospendProposal((
                *message_id,
                *bulletin_board_id,
            ))];
        }

        if state.payment_obligations.is_empty() {
            return vec![Action::Wait];
        }

        // Propose to each orderbook UTXO (non-committal): one interest per peer coin.
        let own_utxo = match state.utxos.first() {
            Some(u) => u.clone(),
            None => return vec![Action::Wait],
        };
        let interests: Vec<CospendInterest> = state
            .orderbook_utxos
            .iter()
            .map(|peer_utxo| CospendInterest {
                utxos: vec![own_utxo.clone(), peer_utxo.clone()],
            })
            .collect();

        if interests.is_empty() {
            return vec![Action::Wait];
        }
        vec![Action::ProposeCospend(interests)]
    }

    fn clone_box(&self) -> Box<dyn Strategy> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct AggregatorStrategy;

impl Strategy for AggregatorStrategy {
    fn enumerate_candidate_actions(
        &self,
        state: &WalletView,
        _wallet: &WalletHandleMut,
    ) -> Vec<Action> {
        if state.pending_interests.is_empty() {
            return vec![Action::Wait];
        }
        vec![Action::CreateAggregateProposal(
            state.pending_interests.clone(),
        )]
    }

    fn clone_box(&self) -> Box<dyn Strategy> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CompositeStrategy {
    pub(crate) strategies: Vec<Box<dyn Strategy>>,
}

impl Strategy for CompositeStrategy {
    fn enumerate_candidate_actions(
        &self,
        state: &WalletView,
        wallet: &WalletHandleMut,
    ) -> Vec<Action> {
        let mut actions = vec![];
        for strategy in self.strategies.iter() {
            actions.extend(strategy.enumerate_candidate_actions(state, wallet));
        }
        actions
    }

    fn clone_box(&self) -> Box<dyn Strategy> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Strategy> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

// TODO: this should be a trait once we have different scoring strategies
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CompositeScorer {
    /// Weight applied to fee savings in sats
    pub(crate) fee_savings_weight: f64,
    /// Weight applied to privacy utility
    pub(crate) privacy_weight: f64,
    /// Weight applied to deadline urgency for payment obligations
    pub(crate) payment_obligation_weight: f64,
    /// Weight applied to multi-party coordination value
    pub(crate) coordination_weight: f64,
}

impl CompositeScorer {
    pub(crate) fn action_cost(
        &self,
        action: &Action,
        wallet_handle: &WalletHandleMut,
    ) -> ActionCost {
        let current_timestep = wallet_handle.sim.current_timestep;
        let outcome = simulate_one_action(wallet_handle, action);
        outcome.cost(self, wallet_handle.sim, current_timestep)
    }
}

/// Creates a strategy instance from its name string
pub(crate) fn create_strategy(name: &str) -> Result<Box<dyn Strategy>, String> {
    match name {
        "UnilateralSpender" => Ok(Box::new(UnilateralSpender)),
        "Consolidator" => Ok(Box::new(Consolidator)),
        "BatchSpender" => Ok(Box::new(BatchSpender)),
        "TakerStrategy" => Ok(Box::new(TakerStrategy)),
        "MakerStrategy" => Ok(Box::new(MakerStrategy)),
        "AggregatorStrategy" => Ok(Box::new(AggregatorStrategy)),
        _ => Err(format!("Unknown strategy: {}", name)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        wallet::{PaymentObligationData, WalletId},
        TimeStep,
    };
    use bitcoin::Amount;

    fn create_test_wallet_view(payment_obligations: Vec<PaymentObligationData>) -> WalletView {
        WalletView::new(
            payment_obligations,
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
        )
    }

    fn test_sim() -> crate::Simulation {
        use crate::{
            config::{ScorerConfig, WalletTypeConfig},
            script_type::ScriptType,
            SimulationBuilder,
        };
        SimulationBuilder::new(
            42,
            vec![WalletTypeConfig {
                name: "test".to_string(),
                count: 2,
                strategies: vec!["UnilateralSpender".to_string()],
                scorer: ScorerConfig {
                    fee_savings_weight: 0.0,
                    privacy_weight: 0.0,
                    payment_obligation_weight: 0.0,
                    coordination_weight: 0.0,
                },
                script_type: ScriptType::P2tr,
            }],
            10,
            1,
            0,
        )
        .build()
    }

    #[test]
    fn test_unilateral_spender() {
        let mut sim = test_sim();
        let wallet = WalletId(0).with_mut(&mut sim);
        let strategy = UnilateralSpender;
        let po = PaymentObligationData {
            id: PaymentObligationId(0),
            deadline: TimeStep(100),
            reveal_time: TimeStep(0),
            amount: Amount::from_sat(1000),
            from: WalletId(0),
            to: WalletId(1),
        };
        let view = create_test_wallet_view(vec![po]);

        let actions = strategy.enumerate_candidate_actions(&view, &wallet);

        assert_eq!(actions.len(), 1);
        assert!(actions
            .iter()
            .any(|a| matches!(a, Action::UnilateralPayments(ids, CoinSelectionStrategy::BNB) if ids.len() == 1)));
    }

    #[test]
    fn test_unilateral_consolidate_spender() {
        let mut sim = test_sim();
        let wallet = WalletId(0).with_mut(&mut sim);
        let strategy = Consolidator;
        let po = PaymentObligationData {
            id: PaymentObligationId(0),
            deadline: TimeStep(100),
            reveal_time: TimeStep(0),
            amount: Amount::from_sat(1000),
            from: WalletId(0),
            to: WalletId(1),
        };
        let view = WalletView::new(vec![po], vec![], vec![], vec![], vec![], vec![], vec![]);

        let actions = strategy.enumerate_candidate_actions(&view, &wallet);

        assert!(actions.iter().any(|a| matches!(a, Action::Wait)));
        // Consolidator always uses SpendAll (strategy commitment, not cost trade-off)
        assert!(actions
            .iter()
            .any(|a| matches!(a, Action::UnilateralPayments(ids, CoinSelectionStrategy::SpendAll) if ids.len() == 1)));
        assert!(!actions
            .iter()
            .any(|a| matches!(a, Action::UnilateralPayments(_, CoinSelectionStrategy::BNB))));
    }

    #[test]
    fn test_batch_spender_creates_batches() {
        let mut sim = test_sim();
        let wallet = WalletId(0).with_mut(&mut sim);
        let strategy = BatchSpender;
        let po1 = PaymentObligationData {
            id: PaymentObligationId(0),
            deadline: TimeStep(100),
            reveal_time: TimeStep(0),
            amount: Amount::from_sat(1000),
            from: WalletId(0),
            to: WalletId(1),
        };
        let po2 = PaymentObligationData {
            id: PaymentObligationId(1),
            deadline: TimeStep(100),
            reveal_time: TimeStep(0),
            amount: Amount::from_sat(2000),
            from: WalletId(0),
            to: WalletId(2),
        };
        let view = create_test_wallet_view(vec![po1, po2]);

        let actions = strategy.enumerate_candidate_actions(&view, &wallet);

        // BatchSpender creates a single batch with all obligations
        assert_eq!(actions.len(), 1);
        assert!(actions
            .iter()
            .any(|a| matches!(a, Action::UnilateralPayments(ids, _) if ids.len() == 2)));
    }

    #[test]
    fn test_composite_strategy_combines_actions() {
        let mut sim = test_sim();
        let wallet = WalletId(0).with_mut(&mut sim);
        let composite = CompositeStrategy {
            strategies: vec![Box::new(UnilateralSpender), Box::new(BatchSpender)],
        };

        let po1 = PaymentObligationData {
            id: PaymentObligationId(0),
            deadline: TimeStep(100),
            reveal_time: TimeStep(0),
            amount: Amount::from_sat(1000),
            from: WalletId(0),
            to: WalletId(1),
        };
        let po2 = PaymentObligationData {
            id: PaymentObligationId(1),
            deadline: TimeStep(100),
            reveal_time: TimeStep(0),
            amount: Amount::from_sat(2000),
            from: WalletId(0),
            to: WalletId(2),
        };
        let view = create_test_wallet_view(vec![po1, po2]);

        let actions = composite.enumerate_candidate_actions(&view, &wallet);

        // Should include actions from both strategies
        // UnilateralSpender: 2 actions (one per obligation, single-PO each)
        // BatchSpender: 1 action (all obligations in one tx)
        assert_eq!(actions.len(), 3);

        let single_po_count = actions
            .iter()
            .filter(|a| matches!(a, Action::UnilateralPayments(ids, CoinSelectionStrategy::BNB) if ids.len() == 1))
            .count();
        assert_eq!(single_po_count, 2);

        let batch_count = actions
            .iter()
            .filter(|a| matches!(a, Action::UnilateralPayments(ids, _) if ids.len() == 2))
            .count();
        assert_eq!(batch_count, 1);
    }

    #[test]
    fn test_taker_waits_with_no_orderbook_utxos() {
        let mut sim = test_sim();
        let wallet = WalletId(0).with_mut(&mut sim);
        let strategy = TakerStrategy;

        let po = PaymentObligationData {
            id: PaymentObligationId(0),
            deadline: TimeStep(100),
            reveal_time: TimeStep(0),
            amount: Amount::from_sat(1000),
            from: WalletId(0),
            to: WalletId(1),
        };
        // No orderbook UTXOs — taker has nothing to propose to
        let view = create_test_wallet_view(vec![po]);

        let actions = strategy.enumerate_candidate_actions(&view, &wallet);

        assert_eq!(actions.len(), 1);
        assert!(matches!(actions[0], Action::Wait));
    }

    #[test]
    fn test_taker_proposes_cospend_with_orderbook_utxos() {
        use crate::{cospend::UtxoWithMetadata, transaction::Outpoint, transaction::TxId};
        let mut sim = test_sim();
        let wallet = WalletId(0).with_mut(&mut sim);
        let strategy = TakerStrategy;

        let po = PaymentObligationData {
            id: PaymentObligationId(0),
            deadline: TimeStep(100),
            reveal_time: TimeStep(0),
            amount: Amount::from_sat(1000),
            from: WalletId(0),
            to: WalletId(1),
        };
        let peer_utxo = UtxoWithMetadata {
            outpoint: Outpoint {
                txid: TxId(99),
                index: 0,
            },
            amount: Amount::from_sat(5000),
            owner: WalletId(1),
        };
        let own_utxo = UtxoWithMetadata {
            outpoint: Outpoint {
                txid: TxId(42),
                index: 0,
            },
            amount: Amount::from_sat(3000),
            owner: WalletId(0),
        };
        let view = WalletView::new(
            vec![po],
            vec![],
            vec![],
            vec![own_utxo],
            vec![],
            vec![peer_utxo],
            vec![],
        );

        let actions = strategy.enumerate_candidate_actions(&view, &wallet);

        assert_eq!(actions.len(), 1);
        assert!(actions
            .iter()
            .any(|a| matches!(a, Action::ProposeCospend(interests) if interests.len() == 1)));
    }

    #[test]
    fn test_taker_continues_active_session() {
        let mut sim = test_sim();
        let wallet = WalletId(0).with_mut(&mut sim);
        let strategy = TakerStrategy;

        let po = PaymentObligationData {
            id: PaymentObligationId(0),
            deadline: TimeStep(100),
            reveal_time: TimeStep(0),
            amount: Amount::from_sat(1000),
            from: WalletId(0),
            to: WalletId(1),
        };

        let view = WalletView::new(
            vec![po],
            vec![],
            vec![BulletinBoardId(0)], // Active session (SentOutputs or later)
            vec![],
            vec![],
            vec![],
            vec![],
        );

        let actions = strategy.enumerate_candidate_actions(&view, &wallet);

        assert_eq!(actions.len(), 1);
        assert!(actions
            .iter()
            .any(|a| matches!(a, Action::ContinueParticipateInCospend(_))));
    }

    #[test]
    fn test_maker_with_new_invitation() {
        let mut sim = test_sim();
        let wallet = WalletId(0).with_mut(&mut sim);
        let strategy = MakerStrategy;

        let po = PaymentObligationData {
            id: PaymentObligationId(0),
            deadline: TimeStep(100),
            reveal_time: TimeStep(0),
            amount: Amount::from_sat(1000),
            from: WalletId(0),
            to: WalletId(1),
        };

        let view = WalletView::new(
            vec![po],
            vec![(BulletinBoardId(0), MessageId(0))], // New invitation
            vec![],                                   // No active sessions yet
            vec![],
            vec![],
            vec![],
            vec![],
        );

        let actions = strategy.enumerate_candidate_actions(&view, &wallet);

        assert_eq!(actions.len(), 1);
        assert!(actions
            .iter()
            .any(|a| matches!(a, Action::AcceptCospendProposal((_, BulletinBoardId(0))))));
    }

    #[test]
    fn test_maker_contributes_outputs_when_session_awaiting() {
        let mut sim = test_sim();

        // Create a bulletin board and accept an invitation, advancing session to SentInputs
        let bb_id = sim.create_bulletin_board();
        let msg_id = MessageId(0);
        WalletId(0)
            .with_mut(&mut sim)
            .do_action(&Action::AcceptCospendProposal((msg_id, bb_id)));

        let po = PaymentObligationData {
            id: PaymentObligationId(0),
            deadline: TimeStep(100),
            reveal_time: TimeStep(0),
            amount: Amount::from_sat(1000),
            from: WalletId(0),
            to: WalletId(1),
        };

        let strategy = MakerStrategy;
        let wallet = WalletId(0).with_mut(&mut sim);

        // Verify the session is in SentInputs state
        let session = wallet
            .info()
            .active_multi_party_payjoins
            .get(&bb_id)
            .unwrap();
        assert_eq!(
            session.state,
            crate::tx_contruction::TxConstructionState::AcceptedProposal
        );
        let view = WalletView::new(vec![po], vec![], vec![], vec![], vec![], vec![], vec![]);
        let actions = strategy.enumerate_candidate_actions(&view, &wallet);

        assert!(actions
            .iter()
            .any(|a| matches!(a, Action::ContributeOutputsToSession(id, ids) if *id == bb_id && ids.len() == 1)));
        // Should NOT emit ContinueParticipateInCospend for this session
        assert!(!actions
            .iter()
            .any(|a| matches!(a, Action::ContinueParticipateInCospend(id) if *id == bb_id)));
    }

    #[test]
    fn test_maker_with_active_session() {
        let mut sim = test_sim();
        let wallet = WalletId(0).with_mut(&mut sim);
        let strategy = MakerStrategy;

        let po = PaymentObligationData {
            id: PaymentObligationId(0),
            deadline: TimeStep(100),
            reveal_time: TimeStep(0),
            amount: Amount::from_sat(1000),
            from: WalletId(0),
            to: WalletId(1),
        };

        let view = WalletView::new(
            vec![po],
            vec![],
            vec![BulletinBoardId(0)], // Active session
            vec![],
            vec![],
            vec![],
            vec![],
        );

        let actions = strategy.enumerate_candidate_actions(&view, &wallet);

        assert_eq!(actions.len(), 1);
        assert!(actions
            .iter()
            .any(|a| matches!(a, Action::ContinueParticipateInCospend(_))));
    }

    #[test]
    fn test_maker_prefers_continue_when_invite_and_active() {
        let mut sim = test_sim();
        let wallet = WalletId(0).with_mut(&mut sim);
        let strategy = MakerStrategy;

        let po = PaymentObligationData {
            id: PaymentObligationId(0),
            deadline: TimeStep(100),
            reveal_time: TimeStep(0),
            amount: Amount::from_sat(1000),
            from: WalletId(0),
            to: WalletId(1),
        };

        // With both a new invitation and active session, strategy should continue active session.
        let view = WalletView::new(
            vec![po],
            vec![(BulletinBoardId(0), MessageId(0))],
            vec![BulletinBoardId(1)],
            vec![],
            vec![],
            vec![],
            vec![],
        );

        let actions = strategy.enumerate_candidate_actions(&view, &wallet);

        assert_eq!(actions.len(), 1);
        assert!(matches!(
            actions[0],
            Action::ContinueParticipateInCospend(BulletinBoardId(1))
        ));
    }
}
