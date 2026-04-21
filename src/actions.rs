use std::{iter::Sum, ops::Add};

use bdk_coin_select::{Target, TargetFee, TargetOutputs};
use bitcoin::Amount;
use log::debug;

use crate::{
    bulletin_board::BulletinBoardId,
    coin_selection::{select_all, select_bnb},
    cospend::CospendInterest,
    message::MessageId,
    transaction::Outpoint,
    tx_contruction::TxConstructionState,
    wallet::{PaymentObligationData, PaymentObligationId, WalletHandle},
    Simulation, TimeStep,
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
    /// Spend a payment obligation unilaterally with pre-selected inputs and pre-computed change
    UnilateralPayments(Vec<PaymentObligationId>, Vec<Outpoint>, Vec<Amount>),
    /// Accept a cospend invitation
    AcceptCospendProposal((MessageId, BulletinBoardId)),
    /// Contribute outputs to a cospend session that is waiting for them, with pre-computed change
    ContributeOutputsToSession(BulletinBoardId, Vec<PaymentObligationId>, Vec<Amount>),
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
#[allow(dead_code)]
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

fn simulate_one_action(wallet_handle: &WalletHandle, action: &Action) -> PredictedOutcome {
    let old_info = wallet_handle.info().clone();

    let wallet_id = wallet_handle.data().id;
    let mut sim = wallet_handle.sim.clone();
    wallet_id.with_mut(&mut sim).do_action(action);
    let new_info = wallet_id.with(&sim).info().clone();

    // POs handled: derived from action since confirmation is deferred to block
    let payment_obligations_handled: Vec<PaymentObligationId> = match action {
        Action::UnilateralPayments(po_ids, _, _) => po_ids.clone(),
        Action::ContributeOutputsToSession(_, po_ids, _) => po_ids.clone(),
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
    fn enumerate_candidate_actions(&self, wallet: &WalletHandle) -> Vec<Action>;
    fn clone_box(&self) -> Box<dyn Strategy>;
}

#[derive(Debug, PartialEq)]
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

impl PartialOrd for ActionCost {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Add for ActionCost {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

/// Build a BDK Target for a slice of payment obligations, estimating output weight
/// from each recipient wallet's script type.
fn target_for_obligations(pos: &[PaymentObligationData], wallet: &WalletHandle) -> Target {
    let value_sum: u64 = pos.iter().map(|po| po.amount.to_sat()).sum();
    let weight_sum: u32 = pos
        .iter()
        .map(|po| po.to.with(wallet.sim).data().script_type.output_weight_wu())
        .sum();
    Target {
        fee: TargetFee {
            rate: bdk_coin_select::FeeRate::from_sat_per_vb(1.0),
            replace: None,
        },
        outputs: TargetOutputs {
            value_sum,
            weight_sum,
            n_outputs: pos.len(),
        },
    }
}

/// Compute pre-selected change outputs for a `ContributeOutputsToSession` action.
/// If the session has pre-selected inputs (from the aggregator), uses those exactly.
/// Otherwise falls back to full BNB / spend-all selection over all wallet UTXOs.
fn change_for_session_contribution(
    bb_id: &BulletinBoardId,
    pos: &[PaymentObligationData],
    wallet: &WalletHandle,
) -> Vec<Amount> {
    let session = wallet
        .info()
        .active_multi_party_payjoins
        .get(bb_id)
        .unwrap();
    let session_input_outpoints: Vec<Outpoint> =
        session.inputs.iter().map(|i| i.outpoint).collect();
    let target = target_for_obligations(pos, wallet);
    if session_input_outpoints.is_empty() {
        let candidates = wallet.coin_candidates();
        if let Some((_, change)) = select_bnb(&candidates, target) {
            return change;
        }
        select_all(&candidates, target).1
    } else {
        let candidates = wallet.coin_candidates_for(&session_input_outpoints);
        select_all(&candidates, target).1
    }
}

#[derive(Debug, Clone)]
pub(crate) struct UnilateralSpender;

impl Strategy for UnilateralSpender {
    /// The decision space of the unilateral spender is the set of all payment obligations.
    /// For each obligation, enumerate both BNB and spend-all coin selections so the cost
    /// function can pick the cheaper input set.
    fn enumerate_candidate_actions(&self, wallet: &WalletHandle) -> Vec<Action> {
        let payment_obligations = wallet.unhandled_payment_obligations();
        if payment_obligations.is_empty() {
            return vec![Action::Wait];
        }
        let candidates = wallet.coin_candidates();
        let mut actions = vec![];
        for po in payment_obligations.iter() {
            let target = target_for_obligations(std::slice::from_ref(po), wallet);
            if let Some((inputs, change)) = select_bnb(&candidates, target) {
                actions.push(Action::UnilateralPayments(vec![po.id], inputs, change));
            }
            let (all_inputs, change) = select_all(&candidates, target);
            if !all_inputs.is_empty() {
                actions.push(Action::UnilateralPayments(vec![po.id], all_inputs, change));
            }
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
pub(crate) struct Consolidator;

impl Strategy for Consolidator {
    /// Always uses spend-all when paying — forces consolidation regardless of fee efficiency.
    /// Fee savings from reducing UTXO fragmentation are captured when fee_savings_weight > 0.
    fn enumerate_candidate_actions(&self, wallet: &WalletHandle) -> Vec<Action> {
        let candidates = wallet.coin_candidates();
        let mut actions = Vec::new();
        for po in wallet.unhandled_payment_obligations().iter() {
            let target = target_for_obligations(std::slice::from_ref(po), wallet);
            let (all_inputs, change) = select_all(&candidates, target);
            if !all_inputs.is_empty() {
                actions.push(Action::UnilateralPayments(vec![po.id], all_inputs, change));
            }
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
    fn enumerate_candidate_actions(&self, wallet: &WalletHandle) -> Vec<Action> {
        let payment_obligations = wallet.unhandled_payment_obligations();
        if payment_obligations.is_empty() {
            return vec![Action::Wait];
        }
        // TODO: we may need to consider different partitioning strategies for the batch spend
        let po_ids: Vec<PaymentObligationId> = payment_obligations.iter().map(|po| po.id).collect();
        let target = target_for_obligations(&payment_obligations, wallet);
        let candidates = wallet.coin_candidates();
        let mut actions = vec![];
        if let Some((inputs, change)) = select_bnb(&candidates, target) {
            actions.push(Action::UnilateralPayments(po_ids.clone(), inputs, change));
        }
        let (all_inputs, change) = select_all(&candidates, target);
        if !all_inputs.is_empty() {
            actions.push(Action::UnilateralPayments(po_ids, all_inputs, change));
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
pub(crate) struct MakerStrategy;

impl Strategy for MakerStrategy {
    fn enumerate_candidate_actions(&self, wallet: &WalletHandle) -> Vec<Action> {
        let mut actions = vec![];

        let active_cospends = wallet.active_cospend_sessions();
        let cospend_proposals = wallet.pending_cospend_proposals();
        let payment_obligations = wallet.unhandled_payment_obligations();
        let registered_inputs = wallet.registered_input_outpoints();

        // Continue to participate in active sessions
        for bulletin_board_id in active_cospends.iter() {
            actions.push(Action::ContinueParticipateInCospend(*bulletin_board_id));
        }

        // Accept new invitations. TODO: in the future makers will be have certain preferences for which invitations to accept.
        if let Some((bulletin_board_id, message_id)) = cospend_proposals.first() {
            if active_cospends.is_empty() {
                actions.push(Action::AcceptCospendProposal((
                    *message_id,
                    *bulletin_board_id,
                )));
            }
        }

        // Contribute outputs to sessions that are waiting for them (AcceptedProposal state)
        for (bb_id, session) in wallet.info().active_multi_party_payjoins.iter() {
            if session.state == TxConstructionState::AcceptedProposal {
                for po in payment_obligations.iter() {
                    let change =
                        change_for_session_contribution(bb_id, std::slice::from_ref(po), wallet);
                    actions.push(Action::ContributeOutputsToSession(
                        *bb_id,
                        vec![po.id],
                        change,
                    ));
                }
            }
        }

        // Only figure out what to register when truly idle: no pending invitations, no active
        // sessions (except completed ones). The wallet's session map is the authoritative source.
        let has_active_sessions = !active_cospends.is_empty()
            || wallet
                .info()
                .active_multi_party_payjoins
                .values()
                .any(|s| !matches!(s.state, TxConstructionState::Success(_)));
        let unilateral_actions = if cospend_proposals.is_empty() && !has_active_sessions {
            UnilateralSpender.enumerate_candidate_actions(wallet)
        } else {
            vec![]
        };
        // Selected inputs are already embedded in each action i.e no simulation needed.
        let per_action_inputs: Vec<std::collections::HashSet<Outpoint>> = unilateral_actions
            .iter()
            .filter_map(|a| match a {
                Action::UnilateralPayments(_, inputs, _) => Some(inputs.iter().copied().collect()),
                _ => None,
            })
            .collect();
        let common_inputs: Vec<Outpoint> = per_action_inputs
            .iter()
            .skip(1)
            .fold(
                per_action_inputs.first().cloned().unwrap_or_default(),
                |acc, s| acc.intersection(s).copied().collect(),
            )
            .iter()
            .filter(|o| !registered_inputs.contains(o))
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
    fn enumerate_candidate_actions(&self, wallet: &WalletHandle) -> Vec<Action> {
        let mut actions = vec![];

        let active_cospends = wallet.active_cospend_sessions();
        let cospend_proposals = wallet.pending_cospend_proposals();
        let payment_obligations = wallet.unhandled_payment_obligations();

        // Contribute outputs to sessions awaiting them (AcceptedProposal state)
        for (bb_id, session) in wallet.info().active_multi_party_payjoins.iter() {
            if session.state == TxConstructionState::AcceptedProposal {
                for po in payment_obligations.iter() {
                    let change =
                        change_for_session_contribution(bb_id, std::slice::from_ref(po), wallet);
                    actions.push(Action::ContributeOutputsToSession(
                        *bb_id,
                        vec![po.id],
                        change,
                    ));
                }
            }
        }

        // Continue active sessions in later states
        for bulletin_board_id in active_cospends.iter() {
            actions.push(Action::ContinueParticipateInCospend(*bulletin_board_id));
        }

        if !actions.is_empty() {
            return actions;
        }

        // Accept any pending invitations from the aggregator before proposing new ones.
        if let Some((bulletin_board_id, message_id)) = cospend_proposals.first() {
            return vec![Action::AcceptCospendProposal((
                *message_id,
                *bulletin_board_id,
            ))];
        }

        if payment_obligations.is_empty() {
            return vec![Action::Wait];
        }

        // Propose to each orderbook UTXO (non-committal): one interest per peer coin.
        let own_utxo = match wallet.spendable_utxos().into_iter().next() {
            Some(u) => u,
            None => return vec![Action::Wait],
        };
        let interests: Vec<CospendInterest> = wallet
            .orderbook_utxos()
            .into_iter()
            .map(|peer_utxo| CospendInterest {
                utxos: vec![own_utxo.clone(), peer_utxo],
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
    fn enumerate_candidate_actions(&self, wallet: &WalletHandle) -> Vec<Action> {
        let pending_interests = wallet.pending_interests();
        if pending_interests.is_empty() {
            return vec![Action::Wait];
        }
        vec![Action::CreateAggregateProposal(pending_interests)]
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
    fn enumerate_candidate_actions(&self, wallet: &WalletHandle) -> Vec<Action> {
        let mut actions = vec![];
        for strategy in self.strategies.iter() {
            actions.extend(strategy.enumerate_candidate_actions(wallet));
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
    pub(crate) fn action_cost(&self, action: &Action, wallet_handle: &WalletHandle) -> ActionCost {
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
        bulletin_board::BulletinBoardId,
        message::{MessageData, MessageId, MessageType},
        tx_contruction::{MultiPartyPayjoinSession, TxConstructionState},
        wallet::{PaymentObligationData, WalletId},
        TimeStep,
    };
    use bitcoin::Amount;

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

    fn add_payment_obligation(sim: &mut crate::Simulation, po: PaymentObligationData) {
        let id = po.id;
        let from = po.from;
        sim.payment_data.push(po);
        let last_id = sim.wallet_data[from.0].last_wallet_info_id;
        sim.wallet_info[last_id.0].payment_obligations.insert(id);
    }

    fn inject_active_session(
        sim: &mut crate::Simulation,
        wallet_id: WalletId,
        bb_id: BulletinBoardId,
        state: TxConstructionState,
    ) {
        let mut wallet = wallet_id.with_mut(sim);
        let mut info = wallet.info().clone();
        info.active_multi_party_payjoins.insert(
            bb_id,
            MultiPartyPayjoinSession {
                state,
                inputs: vec![],
                payment_obligation_ids: vec![],
            },
        );
        wallet.update_info(info);
    }

    fn add_cospend_proposal_message(
        sim: &mut crate::Simulation,
        to: WalletId,
        from: WalletId,
        bb_id: BulletinBoardId,
    ) {
        let id = MessageId(sim.messages.len());
        sim.messages.push(MessageData {
            id,
            message: MessageType::ProposeCoSpend(bb_id),
            from,
            to,
        });
    }

    #[test]
    fn test_unilateral_spender_no_utxos() {
        let mut sim = test_sim();
        let po = PaymentObligationData {
            id: PaymentObligationId(0),
            deadline: TimeStep(100),
            reveal_time: TimeStep(0),
            amount: Amount::from_sat(1000),
            from: WalletId(0),
            to: WalletId(1),
        };
        add_payment_obligation(&mut sim, po);
        let wallet = WalletId(0).with_mut(&mut sim);
        let strategy = UnilateralSpender;

        let actions = strategy.enumerate_candidate_actions(&wallet);

        // Wallet has no UTXOs, coin selection produces nothing falls back to Wait.
        assert_eq!(actions.len(), 1);
        assert!(matches!(actions[0], Action::Wait));
    }

    #[test]
    fn test_unilateral_consolidate_spender_no_utxos() {
        let mut sim = test_sim();
        let po = PaymentObligationData {
            id: PaymentObligationId(0),
            deadline: TimeStep(100),
            reveal_time: TimeStep(0),
            amount: Amount::from_sat(1000),
            from: WalletId(0),
            to: WalletId(1),
        };
        add_payment_obligation(&mut sim, po);
        let wallet = WalletId(0).with_mut(&mut sim);
        let strategy = Consolidator;

        let actions = strategy.enumerate_candidate_actions(&wallet);

        // Consolidator always emits Wait, and skips UnilateralPayments when no UTXOs exist.
        assert!(actions.iter().any(|a| matches!(a, Action::Wait)));
        assert!(!actions
            .iter()
            .any(|a| matches!(a, Action::UnilateralPayments(_, _, _))));
    }

    #[test]
    fn test_batch_spender_no_utxos() {
        let mut sim = test_sim();
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
            to: WalletId(1),
        };
        add_payment_obligation(&mut sim, po1);
        add_payment_obligation(&mut sim, po2);
        let wallet = WalletId(0).with_mut(&mut sim);
        let strategy = BatchSpender;

        let actions = strategy.enumerate_candidate_actions(&wallet);

        // No UTXOs coin selection produces nothing, falls back to Wait.
        assert_eq!(actions.len(), 1);
        assert!(matches!(actions[0], Action::Wait));
    }

    #[test]
    fn test_composite_strategy_combines_actions() {
        // TODO: this test is kinda useless, we need to add UTXOs to the sim and test the composite strategy.
        // Otherwise we are just testing that both strategies fall back to Wait when there are no UTXOs.
        // This is bc coin selection uses `wallet.handle().coin_candidates();` not `state.utxos`.
        let mut sim = test_sim();
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
            to: WalletId(1),
        };
        add_payment_obligation(&mut sim, po1);
        add_payment_obligation(&mut sim, po2);
        let wallet = WalletId(0).with_mut(&mut sim);
        let composite = CompositeStrategy {
            strategies: vec![Box::new(UnilateralSpender), Box::new(BatchSpender)],
        };

        let actions = composite.enumerate_candidate_actions(&wallet);

        // Wallet has no UTXOs in the sim, both strategies fall back to Wait.
        // Composite collects one Wait from each strategy.
        assert_eq!(actions.len(), 2);
        assert!(actions.iter().all(|a| matches!(a, Action::Wait)));
    }

    #[test]
    fn test_taker_waits_with_no_orderbook_utxos() {
        let mut sim = test_sim();
        let po = PaymentObligationData {
            id: PaymentObligationId(0),
            deadline: TimeStep(100),
            reveal_time: TimeStep(0),
            amount: Amount::from_sat(1000),
            from: WalletId(0),
            to: WalletId(1),
        };
        add_payment_obligation(&mut sim, po);
        // No UTXOs or orderbook entries — taker has nothing to propose with or to
        let wallet = WalletId(0).with_mut(&mut sim);
        let strategy = TakerStrategy;

        let actions = strategy.enumerate_candidate_actions(&wallet);

        assert_eq!(actions.len(), 1);
        assert!(matches!(actions[0], Action::Wait));
    }

    #[test]
    fn test_taker_proposes_cospend_with_orderbook_utxos() {
        let mut sim = test_sim();
        // Give wallets real UTXOs via build_universe
        sim.build_universe();

        let po = PaymentObligationData {
            id: PaymentObligationId(sim.payment_data.len()),
            deadline: TimeStep(100),
            reveal_time: TimeStep(0),
            amount: Amount::from_sat(1000),
            from: WalletId(0),
            to: WalletId(1),
        };
        add_payment_obligation(&mut sim, po);

        // Register exactly one of wallet 1's UTXOs in the orderbook
        let first_peer_utxo = WalletId(1)
            .with_mut(&mut sim)
            .info()
            .confirmed_utxos
            .iter()
            .next()
            .cloned()
            .unwrap();
        WalletId(1)
            .with_mut(&mut sim)
            .do_action(&Action::RegisterInput(vec![first_peer_utxo]));

        let strategy = TakerStrategy;
        let wallet = WalletId(0).with_mut(&mut sim);

        let actions = strategy.enumerate_candidate_actions(&wallet);

        assert_eq!(actions.len(), 1);
        assert!(actions
            .iter()
            .any(|a| matches!(a, Action::ProposeCospend(interests) if interests.len() == 1)));
    }

    #[test]
    fn test_taker_continues_active_session() {
        let mut sim = test_sim();
        let po = PaymentObligationData {
            id: PaymentObligationId(0),
            deadline: TimeStep(100),
            reveal_time: TimeStep(0),
            amount: Amount::from_sat(1000),
            from: WalletId(0),
            to: WalletId(1),
        };
        add_payment_obligation(&mut sim, po);
        let bb_id = sim.create_bulletin_board();
        inject_active_session(
            &mut sim,
            WalletId(0),
            bb_id,
            TxConstructionState::SentOutputs,
        );

        let wallet = WalletId(0).with_mut(&mut sim);
        let strategy = TakerStrategy;

        let actions = strategy.enumerate_candidate_actions(&wallet);

        assert_eq!(actions.len(), 1);
        assert!(actions
            .iter()
            .any(|a| matches!(a, Action::ContinueParticipateInCospend(_))));
    }

    #[test]
    fn test_maker_with_new_invitation() {
        let mut sim = test_sim();
        let po = PaymentObligationData {
            id: PaymentObligationId(0),
            deadline: TimeStep(100),
            reveal_time: TimeStep(0),
            amount: Amount::from_sat(1000),
            from: WalletId(0),
            to: WalletId(1),
        };
        add_payment_obligation(&mut sim, po);
        let bb_id = sim.create_bulletin_board();
        // Send a ProposeCoSpend message from wallet 1 to wallet 0
        add_cospend_proposal_message(&mut sim, WalletId(0), WalletId(1), bb_id);

        let wallet = WalletId(0).with_mut(&mut sim);
        let strategy = MakerStrategy;

        let actions = strategy.enumerate_candidate_actions(&wallet);

        assert_eq!(actions.len(), 1);
        assert!(actions
            .iter()
            .any(|a| matches!(a, Action::AcceptCospendProposal((_, id)) if *id == bb_id)));
    }

    #[test]
    fn test_maker_contributes_outputs_when_session_awaiting() {
        let mut sim = test_sim();

        // Create a bulletin board and accept an invitation, advancing session to AcceptedProposal
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
        add_payment_obligation(&mut sim, po);

        let strategy = MakerStrategy;
        let wallet = WalletId(0).with_mut(&mut sim);

        // Verify the session is in AcceptedProposal state
        let session = wallet
            .info()
            .active_multi_party_payjoins
            .get(&bb_id)
            .unwrap();
        assert_eq!(session.state, TxConstructionState::AcceptedProposal);

        let actions = strategy.enumerate_candidate_actions(&wallet);

        assert!(actions
            .iter()
            .any(|a| matches!(a, Action::ContributeOutputsToSession(id, ids, _) if *id == bb_id && ids.len() == 1)));
        // Should NOT emit ContinueParticipateInCospend for this session
        assert!(!actions
            .iter()
            .any(|a| matches!(a, Action::ContinueParticipateInCospend(id) if *id == bb_id)));
    }

    #[test]
    fn test_maker_with_active_session() {
        let mut sim = test_sim();
        let po = PaymentObligationData {
            id: PaymentObligationId(0),
            deadline: TimeStep(100),
            reveal_time: TimeStep(0),
            amount: Amount::from_sat(1000),
            from: WalletId(0),
            to: WalletId(1),
        };
        add_payment_obligation(&mut sim, po);
        let bb_id = sim.create_bulletin_board();
        inject_active_session(
            &mut sim,
            WalletId(0),
            bb_id,
            TxConstructionState::SentOutputs,
        );

        let wallet = WalletId(0).with_mut(&mut sim);
        let strategy = MakerStrategy;

        let actions = strategy.enumerate_candidate_actions(&wallet);

        assert_eq!(actions.len(), 1);
        assert!(actions
            .iter()
            .any(|a| matches!(a, Action::ContinueParticipateInCospend(_))));
    }

    #[test]
    fn test_maker_prefers_continue_when_invite_and_active() {
        let mut sim = test_sim();
        let po = PaymentObligationData {
            id: PaymentObligationId(0),
            deadline: TimeStep(100),
            reveal_time: TimeStep(0),
            amount: Amount::from_sat(1000),
            from: WalletId(0),
            to: WalletId(1),
        };
        add_payment_obligation(&mut sim, po);
        let bb_invite = sim.create_bulletin_board();
        let bb_active = sim.create_bulletin_board();
        // Pending invitation for bb_invite
        add_cospend_proposal_message(&mut sim, WalletId(0), WalletId(1), bb_invite);
        // Active session for bb_active (SentOutputs state)
        inject_active_session(
            &mut sim,
            WalletId(0),
            bb_active,
            TxConstructionState::SentOutputs,
        );

        let wallet = WalletId(0).with_mut(&mut sim);
        let strategy = MakerStrategy;

        let actions = strategy.enumerate_candidate_actions(&wallet);

        // With both a new invitation and active session, strategy should continue active session.
        assert_eq!(actions.len(), 1);
        assert!(matches!(
            actions[0],
            Action::ContinueParticipateInCospend(id) if id == bb_active
        ));
    }
}
