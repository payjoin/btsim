use std::{collections::HashSet, iter::Sum, ops::Add};

use log::debug;

use crate::{
    bulletin_board::BulletinBoardId,
    cospend::UtxoWithAmount,
    message::MessageId,
    transaction::{Outpoint, TxId},
    wallet::{PaymentObligationData, PaymentObligationId, WalletHandleMut, WalletId},
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
    /// Spend a payment obligation unilaterally
    UnilateralSpend(PaymentObligationId),
    /// Batch spend multiple payment obligations
    BatchSpend(Vec<PaymentObligationId>),
    /// Pay a payment obligation while consolidating all UTXOs into change
    ConsolidateSelf(PaymentObligationId),
    /// Participate in Multiparty payjoin
    AcceptCospendProposal((MessageId, BulletinBoardId, PaymentObligationId)),
    /// Continue to participate in a multi-party payjoin
    ContinueParticipateInCospend(BulletinBoardId),
    /// Create a cospend proposal: batch payment obligations and pair with order book UTXOs
    CreateCospendProposal(Vec<PaymentObligationId>),
    /// Register a single UTXO in the order book (maker action)
    RegisterInput(Outpoint),
    /// Do nothing. There may be better oppurtunities to spend a payment obligation or participate in a payjoin.
    Wait,
}

/// Hypothetical outcomes of an action
#[derive(Debug)]
pub(crate) enum PredictedOutcome {
    PaymentObligationsHandled(Vec<PaymentObligationHandledOutcome>),
    AcceptCospendProposal(AcceptCospendOutcome),
    CreateCospendProposal(CreateCospendProposalOutcome),
    Consolidation(ConsolidationOutcome),
    RegisterInput(RegisterInputOutcome),
}

#[derive(Debug)]
pub(crate) struct PaymentObligationHandledOutcome {
    /// Base cost: fee_paid + amount handled. In sats
    base_cost: f64,
    /// Time left on the payment obligation
    time_left: i32,
}

impl PaymentObligationHandledOutcome {
    fn cost(&self, payment_obligation_weight: f64) -> ActionCost {
        let points = [
            (0.0, 2.0 * payment_obligation_weight),
            (2.0, payment_obligation_weight),
            (5.0, 0.0), // There may be other oppurtunities if waited a longer
        ];
        let utility = piecewise_linear(self.time_left as f64, &points); // Also in denominated in sats.
        let cost = self.base_cost - utility;
        debug!("PaymentObligationHandledEvent cost: {:?}", cost);
        ActionCost(cost)
    }
}

#[derive(Debug)]
pub(crate) struct ParticipateMultiPartyPayjoinOutcome {
    /// Time left on the payment obligation
    time_left: i32,
    /// Base cost: fee_paid + amount handled. In sats
    base_cost: f64,
}

 #[derive(Debug)]
 pub(crate) struct RespondToPayjoinOutcome {
     /// Base cost: fee_paid + amount handled. In sats
     base_cost: f64,
 }
 
 impl RespondToPayjoinOutcome {
     fn cost(&self) -> ActionCost {
         // Responding to a payjoin should always be better than unilaterally spending at this point
         // As there is no interaction cost. TODO in the future we will want to model the cost of doing
         // the last round of interaction with the counterparty as a function of rounds remaining.
         ActionCost(0.0)
     }
 }
 
 #[derive(Debug)]
      // TODO: model the participation utility as a linear function of the progression of the session
pub(crate) struct AcceptCospendOutcome;

impl AcceptCospendOutcome {
    fn cost(&self) -> ActionCost {
        // TODO: model the participation utility as a linear function of the oppurtunity cost of participating in other cospends
        // The payoff assigned to participating
        // And the deadline for the payment obligation that this maker may have
        // For now this costs nothing as testing scaffolding.
        ActionCost(0.0)
    }
}

#[derive(Debug)]
pub(crate) struct ConsolidationOutcome {
    /// Base cost: fee_paid in sats.
    #[allow(dead_code)]
    base_cost: f64,
}

impl ConsolidationOutcome {
    fn cost(&self) -> ActionCost {
        debug!("ConsolidationEvent cost: 0");
        ActionCost(0.0)
    }
}

#[derive(Debug)]
pub(crate) struct CreateCospendProposalOutcome {
    /// Time left on the earliest-deadline payment obligation
    time_left: i32,
    /// Base cost: fee_paid + total amount handled. In sats
    base_cost: f64,
}

impl CreateCospendProposalOutcome {
    fn cost(&self, privacy_weight: f64) -> ActionCost {
        // Similar shape to InitiatePayjoinOutcome -- creating a proposal is
        // speculative, so the utility increases with time remaining.
        let points = [
            (0.0, 0.0),
            (2.0, privacy_weight),
            (5.0, 5.0 * privacy_weight),
        ];
        let utility = piecewise_linear(self.time_left as f64, &points);
        ActionCost(self.base_cost - utility)
    }
}

#[derive(Debug)]
pub(crate) struct RegisterInputOutcome {
    /// Number of payment obligations the wallet currently has
    num_payment_obligations: usize,
    /// Number of inputs already registered (including this one)
    num_registered_inputs: usize,
}

impl RegisterInputOutcome {
    fn cost(&self, coordination_weight: f64) -> ActionCost {
        // More payment obligations = higher cost (wallet should be spending, not registering)
        // TODO: this should be the cost of missing those payments bc we dont have enought inputs to spend
        let obligation_pressure = self.num_payment_obligations as f64 * coordination_weight;
        // Registering additional inputs is increasingly costly
        let multi_registration_cost =
            (self.num_registered_inputs as f64).powi(2) * coordination_weight;
        ActionCost(obligation_pressure + multi_registration_cost)
    }
}

/// State of the wallet that can be used to potential enumerate actions
#[derive(Debug)]
pub(crate) struct WalletView {
    payment_obligations: Vec<PaymentObligationData>,
    active_cospends: Vec<BulletinBoardId>,
    cospend_proposals: Vec<(BulletinBoardId, MessageId)>,
    current_timestep: TimeStep,
    wallet_id: WalletId,
    utxos: Vec<UtxoWithAmount>,
    registered_inputs: Vec<Outpoint>,
}

impl WalletView {
    pub(crate) fn new(
        payment_obligations: Vec<PaymentObligationData>,
        cospend_proposals: Vec<(BulletinBoardId, MessageId)>,
        active_cospends: Vec<BulletinBoardId>,
        current_timestep: TimeStep,
        wallet_id: WalletId,
        utxos: Vec<UtxoWithAmount>,
        registered_inputs: Vec<Outpoint>,
    ) -> Self {
        Self {
            payment_obligations,
            active_cospends,
            cospend_proposals,
            current_timestep,
            wallet_id,
            utxos,
            registered_inputs,
        }
    }
}
fn get_payment_obligation_handled_outcome(
    payment_obligation_id: &PaymentObligationId,
    sim: &Simulation,
    current_timestep: TimeStep,
    fee_paid: f64,
) -> PaymentObligationHandledOutcome {
    let payment_obligation = payment_obligation_id.with(&sim).data();
    let deadline = payment_obligation.deadline;
    PaymentObligationHandledOutcome {
        base_cost: fee_paid
            + payment_obligation
                .amount
                .to_float_in(bitcoin::Denomination::Satoshi),
        time_left: deadline.0 as i32 - current_timestep.0 as i32,
    }
}

fn simulate_one_action(wallet_handle: &WalletHandleMut, action: &Action) -> Vec<PredictedOutcome> {
    let wallet_view = wallet_handle.wallet_view();
    let mut events = vec![];
    let old_info = wallet_handle.info().clone();

    // Deep clone the simulation and run the action
    let wallet_id = wallet_handle.data().id;
    let mut sim = wallet_handle.sim.clone();
    let mut new_wallet_handle = wallet_handle.data().id.with_mut(&mut sim);
    new_wallet_handle.do_action(action);
    let new_wallet_handle = wallet_id.with(&sim);
    let new_info = new_wallet_handle.info();
    let fee_paid_total = action_fee_paid_sats(&old_info, new_info, &sim);

    if let Action::UnilateralSpend(payment_obligation_id) = action {
        let payment_obligation = payment_obligation_id.with(&sim).data();
        let deadline = payment_obligation.deadline;
        events.push(PredictedOutcome::PaymentObligationsHandled(vec![
            PaymentObligationHandledOutcome {
                base_cost: fee_paid_total
                    + payment_obligation
                        .amount
                        .to_float_in(bitcoin::Denomination::Satoshi),
                time_left: deadline.0 as i32 - wallet_view.current_timestep.0 as i32,
            },
        ]));
    }

    if let Action::BatchSpend(payment_obligation_ids) = action {
        let mut outcomes = vec![];
        let total_amount_handled: f64 = payment_obligation_ids
            .iter()
            .map(|payment_obligation_id| {
                payment_obligation_id
                    .with(&sim)
                    .data()
                    .amount
                    .to_float_in(bitcoin::Denomination::Satoshi)
            })
            .sum();
        for payment_obligation_id in payment_obligation_ids.iter() {
            let amount_handled = payment_obligation_id
                .with(&sim)
                .data()
                .amount
                .to_float_in(bitcoin::Denomination::Satoshi);
            let fee_paid = if total_amount_handled > 0.0 {
                fee_paid_total * (amount_handled / total_amount_handled)
            } else {
                0.0
            };
            outcomes.push(get_payment_obligation_handled_outcome(
                payment_obligation_id,
                &sim,
                wallet_view.current_timestep,
                fee_paid,
            ));
        }
        events.push(PredictedOutcome::PaymentObligationsHandled(outcomes));
    }

    if matches!(action, Action::ConsolidateSelf(_)) {
        events.push(PredictedOutcome::Consolidation(ConsolidationOutcome {
            base_cost: fee_paid_total,
        }));
    }

    if let Action::CreateCospendProposal(po_ids) = action {
        let earliest_deadline = po_ids
            .iter()
            .map(|id| {
                id.with(&sim).data().deadline.0 as i32 - wallet_view.current_timestep.0 as i32
            })
            .min()
            .unwrap_or(0);
        events.push(PredictedOutcome::CreateCospendProposal(
            CreateCospendProposalOutcome {
                time_left: earliest_deadline,
                base_cost: fee_paid_total,
            },
        ));
    }

    if let Action::AcceptCospendProposal((_, _, _)) = action {
        events.push(PredictedOutcome::AcceptCospendProposal(
            AcceptCospendOutcome,
        ));
    }

    if matches!(action, Action::RegisterInput(_)) {
        let num_registered = wallet_view.registered_inputs.len() + 1;
        events.push(PredictedOutcome::RegisterInput(RegisterInputOutcome {
            num_payment_obligations: wallet_view.payment_obligations.len(),
            num_registered_inputs: num_registered,
        }));
    }

    events
}

fn action_fee_paid_sats(
    old_info: &crate::wallet::WalletInfo,
    new_info: &crate::wallet::WalletInfo,
    sim: &Simulation,
) -> f64 {
    let old_txs: HashSet<TxId> = old_info.broadcast_transactions.iter().copied().collect();
    new_info
        .broadcast_transactions
        .iter()
        .filter(|txid| !old_txs.contains(txid))
        .map(|txid| txid.with(sim).info().fee.to_sat() as f64)
        .sum()
}

/// Strategies will pick one action to minimize their cost
/// TODO: Strategies should be composible. They should enform the action decision space scoring and doing actions should be handling by something else that has composed multiple strategies.
pub(crate) trait Strategy: std::fmt::Debug {
    fn enumerate_candidate_actions(&self, state: &WalletView) -> Vec<Action>;
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
    fn enumerate_candidate_actions(&self, state: &WalletView) -> Vec<Action> {
        if state.payment_obligations.is_empty() {
            return vec![Action::Wait];
        }
        let mut actions = vec![];
        for po in state.payment_obligations.iter() {
            // For every payment obligation, we can spend it unilaterally
            actions.push(Action::UnilateralSpend(po.id));
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
    fn enumerate_candidate_actions(&self, state: &WalletView) -> Vec<Action> {
        let mut actions = Vec::new();
        for po in state.payment_obligations.iter() {
            actions.push(Action::UnilateralSpend(po.id));
            actions.push(Action::ConsolidateSelf(po.id));
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
    fn enumerate_candidate_actions(&self, state: &WalletView) -> Vec<Action> {
        if state.payment_obligations.is_empty() {
            return vec![Action::Wait];
        }
        // TODO: we may need to consider differnt partitioning strategies for the batch spend
        // Maybe some partitions have higher utility if batched than others that have longer deadlines
        let mut payment_obligation_ids: Vec<PaymentObligationId> = vec![];
        for po in state.payment_obligations.iter() {
            payment_obligation_ids.push(po.id);
        }
        vec![Action::BatchSpend(payment_obligation_ids)]
    }

    fn clone_box(&self) -> Box<dyn Strategy> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
 pub(crate) struct MultipartyPayjoinInitiatorStrategy;
 
 impl Strategy for MultipartyPayjoinInitiatorStrategy {
     fn enumerate_candidate_actions(&self, state: &WalletView) -> Vec<Action> {
         if state.payment_obligations.is_empty() {
             return vec![Action::Wait];
         }
         // TODO: if the sesion is on going do not intiate a new one
         // TODO: this is scaffolding for now, peers in the future will evaluate if they should initiate a multi-party payjoin given the number of payment obligations
         if state.wallet_id != WalletId(0) {
             return vec![Action::Wait];
         }
         let receivers = state
             .payment_obligations
             .iter()
             .map(|po| po.to)
             .collect::<HashSet<_>>();
         if receivers.len() < 2 {
             return vec![Action::Wait];
         }
 
         // TODO: only one multi-party payjoin session can be active at a time FOR NOW
         let mut actions = vec![];
        if !state.active_cospends.is_empty() {
             // If we have an active session we should actively participate in it
            debug_assert!(state.active_cospends.len() <= 1);
            for bulletin_board_id in state.active_cospends.iter() {
                actions.push(Action::ContinueParticipateInCospend(*bulletin_board_id));
             }
             return actions;
         }
 
        actions.push(Action::CreateCospendProposal(
             state.payment_obligations.iter().map(|po| po.id).collect(),
         ));
 
         actions
     }
 
     fn clone_box(&self) -> Box<dyn Strategy> {
         Box::new(self.clone())
     }
 }
 
 #[derive(Debug, Clone)]
pub(crate) struct MakerStrategy;

impl Strategy for MakerStrategy {
    fn enumerate_candidate_actions(&self, state: &WalletView) -> Vec<Action> {
        let mut actions = vec![];

        // Continue to participate in active sessions
        for bulletin_board_id in state.active_cospends.iter() {
            actions.push(Action::ContinueParticipateInCospend(*bulletin_board_id));
        }

        // Accept new invitations
        if let Some((bulletin_board_id, message_id)) = state.cospend_proposals.iter().next() {
            if state.active_cospends.is_empty() {
                for po in state.payment_obligations.iter() {
                    actions.push(Action::AcceptCospendProposal((
                        *message_id,
                        *bulletin_board_id,
                        po.id,
                    )));
                }
            }
        }

        // Register unregistered UTXOs in the order book (one action per UTXO)
        for utxo in state.utxos.iter() {
            if !state.registered_inputs.contains(&utxo.outpoint) {
                actions.push(Action::RegisterInput(utxo.outpoint));
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
pub(crate) struct TakerStrategy;

impl Strategy for TakerStrategy {
    fn enumerate_candidate_actions(&self, state: &WalletView) -> Vec<Action> {
        if state.payment_obligations.is_empty() {
            return vec![Action::Wait];
        }
        let po_ids: Vec<PaymentObligationId> =
            state.payment_obligations.iter().map(|po| po.id).collect();
        // If we have an active session, continue participating in it
        if !state.active_cospends.is_empty() {
            let mut actions = vec![];
            for bulletin_board_id in state.active_cospends.iter() {
                actions.push(Action::ContinueParticipateInCospend(*bulletin_board_id));
            }
            return actions;
        }

        let receivers = state
            .payment_obligations
            .iter()
            .map(|po| po.to)
            .collect::<HashSet<_>>();
        if receivers.len() < 2 {
            return vec![Action::Wait];
        }

        vec![Action::CreateCospendProposal(po_ids)]
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
    fn enumerate_candidate_actions(&self, state: &WalletView) -> Vec<Action> {
        let mut actions = vec![];
        for strategy in self.strategies.iter() {
            actions.extend(strategy.enumerate_candidate_actions(state));
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
        let events = simulate_one_action(wallet_handle, action);
        // For now each action should only result in one event or none if we are waiting
        debug_assert!(events.len() <= 1);
        let mut cost = ActionCost(INHERENT_ACTION_COST);
        for event in events {
            match event {
                PredictedOutcome::PaymentObligationsHandled(outcomes) => {
                    for outcome in outcomes.iter() {
                        cost = cost + outcome.cost(self.payment_obligation_weight);
                    }
                }
                PredictedOutcome::AcceptCospendProposal(event) => {
                    cost = cost + event.cost();
                }
                PredictedOutcome::Consolidation(event) => {
                    cost = cost + event.cost();
                }
                PredictedOutcome::CreateCospendProposal(event) => {
                    cost = cost + event.cost(self.privacy_weight);
                }
                PredictedOutcome::RegisterInput(event) => {
                    cost = cost + event.cost(self.coordination_weight);
                }
            }
        }
        cost
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
        _ => Err(format!("Unknown strategy: {}", name)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{wallet::PaymentObligationData, TimeStep};
    use bitcoin::Amount;

    fn create_test_wallet_view(payment_obligations: Vec<PaymentObligationData>) -> WalletView {
        WalletView::new(
            payment_obligations,
            vec![],
            vec![],
            TimeStep(0),
            WalletId(0),
            vec![],
            vec![],
        )
    }

    #[test]
    fn test_unilateral_spender() {
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

        let actions = strategy.enumerate_candidate_actions(&view);

        assert_eq!(actions.len(), 1);
        assert!(actions
            .iter()
            .any(|a| matches!(a, Action::UnilateralSpend(_))));
    }

    #[test]
    fn test_unilateral_consolidate_spender() {
        let strategy = Consolidator;
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
            vec![],
            TimeStep(0),
            WalletId(0),
            vec![],
            vec![],
        );

        let actions = strategy.enumerate_candidate_actions(&view);

        assert!(actions
            .iter()
            .any(|a| matches!(a, Action::ConsolidateSelf(_))));
        assert!(actions.iter().any(|a| matches!(a, Action::Wait)));
        assert!(actions
            .iter()
            .any(|a| matches!(a, Action::UnilateralSpend(_))));
    }

    #[test]
    fn test_batch_spender_creates_batches() {
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

        let actions = strategy.enumerate_candidate_actions(&view);

        // BatchSpender creates a single batch with all obligations
        assert_eq!(actions.len(), 1);
        assert!(actions
            .iter()
            .any(|a| matches!(a, Action::BatchSpend(ids) if ids.len() == 2)));
    }

    #[test]
    fn test_composite_strategy_combines_actions() {
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

        let actions = composite.enumerate_candidate_actions(&view);

        // Should include actions from both strategies
        // UnilateralSpender: 2 actions (one per obligation)
        // BatchSpender: 1 action (batch of both)
        assert_eq!(actions.len(), 3);

        let unilateral_count = actions
            .iter()
            .filter(|a| matches!(a, Action::UnilateralSpend(_)))
            .count();
        assert_eq!(unilateral_count, 2);

        let batch_count = actions
            .iter()
            .filter(|a| matches!(a, Action::BatchSpend(_)))
            .count();
        assert_eq!(batch_count, 1);
    }

    #[test]
    fn test_multiparty_initiator_with_insufficient_receivers() {
        let strategy = TakerStrategy;

        // Only 1 unique receiver - not enough for multi-party
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
            to: WalletId(1), // Same receiver
        };
        let view = create_test_wallet_view(vec![po1, po2]);

        let actions = strategy.enumerate_candidate_actions(&view);

        // Should return Wait because we need at least 2 different receivers
        assert_eq!(actions.len(), 1);
        assert!(matches!(actions[0], Action::Wait));
    }

    #[test]
    fn test_multiparty_initiator_with_multiple_receivers() {
        let strategy = TakerStrategy;

        // 2 different receivers - enough for multi-party
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

        let actions = strategy.enumerate_candidate_actions(&view);

        assert_eq!(actions.len(), 1);
        assert!(actions
            .iter()
            .any(|a| matches!(a, Action::CreateCospendProposal(ids) if ids.len() == 2)));
    }

    #[test]
    fn test_multiparty_initiator_only_wallet_0() {
        let strategy = TakerStrategy;

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

        // Create view with wallet_id = 1 (not 0)
        let view = create_test_wallet_view(vec![po1, po2]);

        let actions = strategy.enumerate_candidate_actions(&view);

        assert_eq!(actions.len(), 1);
        assert!(actions
            .iter()
            .any(|a| matches!(a, Action::CreateCospendProposal(ids) if ids.len() == 2)));
    }

    #[test]
    fn test_taker_continues_active_session() {
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
            vec![BulletinBoardId(0)], // Active session
            TimeStep(0),
            WalletId(0),
            vec![],
            vec![],
        );

        let actions = strategy.enumerate_candidate_actions(&view);

        assert_eq!(actions.len(), 1);
        assert!(actions
            .iter()
            .any(|a| matches!(a, Action::ContinueParticipateInCospend(_))));
    }

    #[test]
    fn test_maker_with_new_invitation() {
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
            TimeStep(0),
            WalletId(1),
            vec![],
            vec![],
        );

        let actions = strategy.enumerate_candidate_actions(&view);

        assert_eq!(actions.len(), 1);
        assert!(actions
            .iter()
            .any(|a| matches!(a, Action::AcceptCospendProposal(_))));
    }

    #[test]
    fn test_maker_with_active_session() {
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
            TimeStep(0),
            WalletId(1),
            vec![],
            vec![],
        );

        let actions = strategy.enumerate_candidate_actions(&view);

        assert_eq!(actions.len(), 1);
        assert!(actions
            .iter()
            .any(|a| matches!(a, Action::ContinueParticipateInCospend(_))));
    }

    #[test]
    fn test_maker_prefers_continue_when_invite_and_active() {
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
            TimeStep(0),
            WalletId(1),
            vec![],
            vec![],
        );

        let actions = strategy.enumerate_candidate_actions(&view);

        assert_eq!(actions.len(), 1);
        assert!(matches!(
            actions[0],
            Action::ContinueParticipateInCospend(BulletinBoardId(1))
        ));
    }
}
