use std::{collections::HashSet, iter::Sum, ops::Add};

use log::debug;

use crate::{
    bulletin_board::BulletinBoardId,
    message::{MessageId, PayjoinProposal},
    transaction::TxId,
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
    /// Initiate a payjoin with a counterparty
    InitiatePayjoin(PaymentObligationId),
    /// respond to a payjoin proposal
    RespondToPayjoin(
        PayjoinProposal,
        PaymentObligationId,
        BulletinBoardId,
        MessageId,
    ),
    InitiateMultiPartyPayjoin(Vec<PaymentObligationId>),
    /// Participate in Multiparty payjoin
    ParticipateMultiPartyPayjoin((MessageId, BulletinBoardId, PaymentObligationId)),
    /// Continue to participate in a multi-party payjoin
    ContinueParticipateMultiPartyPayjoin(BulletinBoardId),
    /// Do nothing. There may be better oppurtunities to spend a payment obligation or participate in a payjoin.
    Wait,
}

/// Hypothetical outcomes of an action
#[derive(Debug)]
pub(crate) enum PredictedOutcome {
    PaymentObligationsHandled(Vec<PaymentObligationHandledOutcome>),
    InitiatePayjoin(InitiatePayjoinOutcome),
    RespondToPayjoin(RespondToPayjoinOutcome),
    InitiateMultiPartyPayjoin(InitiateMultiPartyPayjoinOutcome),
    ParticipateMultiPartyPayjoin(ParticipateMultiPartyPayjoinOutcome),
    Consolidation(ConsolidationOutcome),
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
pub(crate) struct InitiatePayjoinOutcome {
    /// Time left on the payment obligation
    time_left: i32,
    /// Base cost: fee_paid + amount handled. In sats
    base_cost: f64,
}

impl InitiatePayjoinOutcome {
    /// Batching anxiety should increase and payjoin utility should decrease the closer the deadline is.
    /// This can be modeled as a inverse cubic function of the time left.
    /// privacy are evaluated independently via per-dimension weights.
    /// TODO: This privacy term is being mis used here. Its just capaturing you value doing payjoins not privacy. A better way to compare the value of different outcomes is just
    /// fee savings. Which should be reflected in the base cost anways.
    fn cost(&self, privacy_weight: f64) -> ActionCost {
        let points = [
            (0.0, 0.0),
            (2.0, privacy_weight),
            (5.0, 5.0 * privacy_weight),
        ];
        let utility = piecewise_linear(self.time_left as f64, &points);
        let cost = self.base_cost - utility;
        debug!("InitiatePayjoinEvent cost: {:?}", cost);
        ActionCost(cost)
    }
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
pub(crate) struct InitiateMultiPartyPayjoinOutcome {
    /// Time left on the payment obligation
    time_left: i32,
    /// Base cost: fee_paid + amount handled. In sats
    base_cost: f64,
    /// Upper bound on the number of participants in the multi-party payjoin
    max_participants: u32,
}

impl InitiateMultiPartyPayjoinOutcome {
    fn cost(&self) -> ActionCost {
        // TODO This should have a similar "shape" as the initiate payjoin but with a different utility function.
        // taking into accounts the number of participants, their inputs.
        // For now this costs nothing as testing scaffolding.
        ActionCost(0.0)
    }
}

#[derive(Debug)]
pub(crate) struct ParticipateMultiPartyPayjoinOutcome {
    /// Time left on the payment obligation
    time_left: i32,
    /// Base cost: fee_paid + amount handled. In sats
    base_cost: f64,
}

impl ParticipateMultiPartyPayjoinOutcome {
    fn cost(&self) -> ActionCost {
        // TODO: model the participation utility as a linear function of the progression of the session
        // For now this costs nothing as testing scaffolding.
        ActionCost(0.0)
    }
}

#[derive(Debug)]
pub(crate) struct ConsolidationOutcome {
    /// Base cost: fee_paid in sats.
    base_cost: f64,
}

impl ConsolidationOutcome {
    fn cost(&self, _consolidation_weight: f64) -> ActionCost {
        debug!("ConsolidationEvent cost: 0");
        ActionCost(0.0)
    }
}

/// State of the wallet that can be used to potential enumerate actions
#[derive(Debug)]
pub(crate) struct WalletView {
    payment_obligations: Vec<PaymentObligationData>,
    payjoin_proposals: Vec<(MessageId, BulletinBoardId, PayjoinProposal)>,
    active_multi_party_payjoins: Vec<BulletinBoardId>,
    new_multi_party_payjoins: Vec<(BulletinBoardId, MessageId)>,
    current_timestep: TimeStep,
    wallet_id: WalletId,
    spendable_utxos_count: usize,
    // TODO: utxos, feerate, cospend oppurtunities, etc.
}

impl WalletView {
    pub(crate) fn new(
        payment_obligations: Vec<PaymentObligationData>,
        payjoin_proposals: Vec<(MessageId, BulletinBoardId, PayjoinProposal)>,
        new_multi_party_payjoins: Vec<(BulletinBoardId, MessageId)>,
        active_multi_party_payjoins: Vec<BulletinBoardId>,
        current_timestep: TimeStep,
        wallet_id: WalletId,
        spendable_utxos_count: usize,
    ) -> Self {
        Self {
            payment_obligations,
            payjoin_proposals,
            active_multi_party_payjoins,
            new_multi_party_payjoins,
            current_timestep,
            wallet_id,
            spendable_utxos_count,
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

    // Check if the wallet initiated a payjoin
    let old_initiated_payjoins = old_info.initiated_payjoins;
    let new_initiated_payjoins = new_info.initiated_payjoins.clone();
    if let Some((payment_obligation_id, _)) = new_initiated_payjoins
        .difference(old_initiated_payjoins)
        .into_iter()
        .next()
    {
        let po = payment_obligation_id.with(&sim).data();
        let amount_handled = po.amount.to_float_in(bitcoin::Denomination::Satoshi);
        events.push(PredictedOutcome::InitiatePayjoin(InitiatePayjoinOutcome {
            time_left: po.deadline.0 as i32 - wallet_view.current_timestep.0 as i32,
            base_cost: fee_paid_total + amount_handled,
        }));
    }

    let old_received_payjoins = old_info.received_payjoins;
    let new_received_payjoins = new_info.received_payjoins.clone();
    if let Some((payment_obligation_id, _)) = new_received_payjoins
        .difference(old_received_payjoins)
        .into_iter()
        .next()
    {
        let po = payment_obligation_id.with(&sim).data();
        let amount_handled = po.amount.to_float_in(bitcoin::Denomination::Satoshi);
        events.push(PredictedOutcome::RespondToPayjoin(
            RespondToPayjoinOutcome {
                base_cost: fee_paid_total + amount_handled,
            },
        ));
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
    /// The decision space of the unilateral spender is the set of all payment obligations and payjoin proposals
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
pub(crate) struct PayjoinStrategy;

impl Strategy for PayjoinStrategy {
    fn enumerate_candidate_actions(&self, state: &WalletView) -> Vec<Action> {
        if state.payment_obligations.is_empty() {
            return vec![Action::Wait];
        }
        let mut actions = vec![];
        for po in state.payment_obligations.iter() {
            // TODO: some payment obligations may not be suitable for payjoin. i.e if the receiver opts out
            actions.push(Action::InitiatePayjoin(po.id));
        }

        // Check for messages from other wallets
        for (message_id, bulletin_board_id, payjoin_proposal) in state.payjoin_proposals.iter() {
            // We should evaluate responding using all payment obligations that have not been handled
            for po in state.payment_obligations.iter() {
                actions.push(Action::RespondToPayjoin(
                    payjoin_proposal.clone(),
                    po.id,
                    *bulletin_board_id,
                    *message_id,
                ));
            }
        }

        actions
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
        if !state.active_multi_party_payjoins.is_empty() {
            // If we have an active session we should actively participate in it
            debug_assert!(state.active_multi_party_payjoins.len() <= 1);
            for bulletin_board_id in state.active_multi_party_payjoins.iter() {
                actions.push(Action::ContinueParticipateMultiPartyPayjoin(
                    *bulletin_board_id,
                ));
            }
            return actions;
        }

        actions.push(Action::InitiateMultiPartyPayjoin(
            state.payment_obligations.iter().map(|po| po.id).collect(),
        ));

        actions
    }

    fn clone_box(&self) -> Box<dyn Strategy> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MultipartyPayjoinParticipantStrategy;

impl Strategy for MultipartyPayjoinParticipantStrategy {
    fn enumerate_candidate_actions(&self, state: &WalletView) -> Vec<Action> {
        if state.payment_obligations.is_empty() {
            return vec![Action::Wait];
        }

        let mut actions = vec![];
        //TODO: Only one multi-party payjoin session can be initiated at a time FOR NOW

        if let Some((bulletin_board_id, message_id)) = state.new_multi_party_payjoins.iter().next()
        {
            // TODO participate in one session at a time
            if state.active_multi_party_payjoins.is_empty() {
                for po in state.payment_obligations.iter() {
                    actions.push(Action::ParticipateMultiPartyPayjoin((
                        *message_id,
                        *bulletin_board_id,
                        po.id,
                    )));
                }
            }
        }

        // Or continue to participate in the existing session
        for bulletin_board_id in state.active_multi_party_payjoins.iter() {
            actions.push(Action::ContinueParticipateMultiPartyPayjoin(
                *bulletin_board_id,
            ));
        }
        actions
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
    /// Weight applied to fee savings in sats from payjoin transactions
    pub(crate) fee_savings_weight: f64,
    /// Weight applied to privacy utility from payjoin transactions
    pub(crate) privacy_weight: f64,
    /// Weight applied to deadline urgency for payment obligations
    pub(crate) payment_obligation_weight: f64,
    /// Weight applied to multi-party coordination value
    pub(crate) coordination_weight: f64,
    /// Weight applied to UTXO consolidation utility
    pub(crate) consolidation_weight: f64,
}

impl CompositeScorer {
    pub(crate) fn action_cost(
        &self,
        action: &Action,
        wallet_handle: &WalletHandleMut,
    ) -> ActionCost {
        let events = simulate_one_action(wallet_handle, action);
        // For now each action should only result in one event or none if we are waiting
        // TODO: wallets should evaluate waiting and reduce its cost if they are expecting payments from payjoin compatible wallets
        debug_assert!(events.len() <= 1);
        let mut cost = ActionCost(INHERENT_ACTION_COST);
        for event in events {
            match event {
                PredictedOutcome::PaymentObligationsHandled(outcomes) => {
                    for outcome in outcomes.iter() {
                        cost = cost + outcome.cost(self.payment_obligation_weight);
                    }
                }
                PredictedOutcome::InitiatePayjoin(event) => {
                    cost = cost + event.cost(self.privacy_weight);
                }
                PredictedOutcome::RespondToPayjoin(event) => {
                    cost = cost + event.cost();
                }
                PredictedOutcome::InitiateMultiPartyPayjoin(event) => {
                    cost = cost + event.cost();
                }
                PredictedOutcome::ParticipateMultiPartyPayjoin(event) => {
                    cost = cost + event.cost();
                }
                PredictedOutcome::Consolidation(event) => {
                    cost = cost + event.cost(self.consolidation_weight);
                }
            }
        }
        if matches!(action, Action::Wait) {
            let view = wallet_handle.wallet_view();
            let points = [(0.0, 2.0), (2.0, 1.0), (5.0, 0.0)];
            let mut penalty = 0.0;
            for po in view.payment_obligations.iter() {
                let time_left = po.deadline.0 as i32 - view.current_timestep.0 as i32;
                let urgency = piecewise_linear(time_left as f64, &points);
                penalty += po.amount.to_sat() as f64 * urgency * self.payment_obligation_weight;
            }
            cost = cost + ActionCost(penalty);
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
        "PayjoinStrategy" => Ok(Box::new(PayjoinStrategy)),
        "MultipartyPayjoinInitiatorStrategy" => Ok(Box::new(MultipartyPayjoinInitiatorStrategy)),
        "MultipartyPayjoinParticipantStrategy" => {
            Ok(Box::new(MultipartyPayjoinParticipantStrategy))
        }
        _ => Err(format!("Unknown strategy: {}", name)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{message::PayjoinProposal, wallet::PaymentObligationData, TimeStep};
    use bitcoin::Amount;

    // Helper to create a minimal WalletView for testing
    fn create_test_wallet_view(
        payment_obligations: Vec<PaymentObligationData>,
        payjoin_proposals: Vec<(MessageId, BulletinBoardId, PayjoinProposal)>,
    ) -> WalletView {
        WalletView::new(
            payment_obligations,
            payjoin_proposals,
            vec![],
            vec![],
            TimeStep(0),
            WalletId(0),
            0,
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
        let view = create_test_wallet_view(vec![po], vec![]);

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
            vec![],
            TimeStep(0),
            WalletId(0),
            3,
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
        let view = create_test_wallet_view(vec![po1, po2], vec![]);

        let actions = strategy.enumerate_candidate_actions(&view);

        // BatchSpender creates a single batch with all obligations
        assert_eq!(actions.len(), 1);
        assert!(actions
            .iter()
            .any(|a| matches!(a, Action::BatchSpend(ids) if ids.len() == 2)));
    }

    #[test]
    fn test_payjoin_strategy() {
        let strategy = PayjoinStrategy;
        let po = PaymentObligationData {
            id: PaymentObligationId(0),
            deadline: TimeStep(100),
            reveal_time: TimeStep(0),
            amount: Amount::from_sat(1000),
            from: WalletId(0),
            to: WalletId(1),
        };

        // Test without proposals should only return InitiatePayjoin
        let view = create_test_wallet_view(vec![po.clone()], vec![]);
        let actions = strategy.enumerate_candidate_actions(&view);

        assert_eq!(actions.len(), 1);
        assert!(actions
            .iter()
            .any(|a| matches!(a, Action::InitiatePayjoin(_))));

        // Test with proposals should return both InitiatePayjoin and RespondToPayjoin
        let proposal = PayjoinProposal {
            tx: crate::transaction::TxData {
                inputs: vec![],
                outputs: vec![],
                wallet_acks: vec![],
            },
            valid_till: TimeStep(200),
        };

        let view =
            create_test_wallet_view(vec![po], vec![(MessageId(0), BulletinBoardId(0), proposal)]);

        let actions = strategy.enumerate_candidate_actions(&view);

        assert_eq!(actions.len(), 2);
        assert!(actions
            .iter()
            .any(|a| matches!(a, Action::InitiatePayjoin(_))));
        assert!(actions
            .iter()
            .any(|a| matches!(a, Action::RespondToPayjoin(_, _, _, _))));
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
        let view = create_test_wallet_view(vec![po1, po2], vec![]);

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
        let strategy = MultipartyPayjoinInitiatorStrategy;

        // Only 1 unique receiver - not enough for multi-party
        let po1 = PaymentObligationData {
            id: PaymentObligationId(0),
            deadline: TimeStep(100),
            reveal_time: TimeStep(0),
            amount: Amount::from_sat(1000),
            from: WalletId(0),
            to: WalletId(1), // Same receiver
        };
        let po2 = PaymentObligationData {
            id: PaymentObligationId(1),
            deadline: TimeStep(100),
            reveal_time: TimeStep(0),
            amount: Amount::from_sat(2000),
            from: WalletId(0),
            to: WalletId(1), // Same receiver
        };
        let view = create_test_wallet_view(vec![po1, po2], vec![]);

        let actions = strategy.enumerate_candidate_actions(&view);

        // Should return Wait because we need at least 2 different receivers
        assert_eq!(actions.len(), 1);
        assert!(matches!(actions[0], Action::Wait));
    }

    #[test]
    fn test_multiparty_initiator_with_multiple_receivers() {
        let strategy = MultipartyPayjoinInitiatorStrategy;

        // 2 different receivers - enough for multi-party
        let po1 = PaymentObligationData {
            id: PaymentObligationId(0),
            deadline: TimeStep(100),
            reveal_time: TimeStep(0),
            amount: Amount::from_sat(1000),
            from: WalletId(0),
            to: WalletId(1), // Receiver 1
        };
        let po2 = PaymentObligationData {
            id: PaymentObligationId(1),
            deadline: TimeStep(100),
            reveal_time: TimeStep(0),
            amount: Amount::from_sat(2000),
            from: WalletId(0),
            to: WalletId(2), // Receiver 2 (different)
        };
        let view = create_test_wallet_view(vec![po1, po2], vec![]);

        let actions = strategy.enumerate_candidate_actions(&view);

        assert_eq!(actions.len(), 1);
        assert!(actions
            .iter()
            .any(|a| matches!(a, Action::InitiateMultiPartyPayjoin(ids) if ids.len() == 2)));
    }

    #[test]
    fn test_multiparty_initiator_only_wallet_0() {
        let strategy = MultipartyPayjoinInitiatorStrategy;

        let po = PaymentObligationData {
            id: PaymentObligationId(0),
            deadline: TimeStep(100),
            reveal_time: TimeStep(0),
            amount: Amount::from_sat(1000),
            from: WalletId(1), // Not wallet 0
            to: WalletId(2),
        };

        // Create view with wallet_id = 1 (not 0)
        let view = WalletView::new(
            vec![po],
            vec![],
            vec![],
            vec![],
            TimeStep(0),
            WalletId(1), // Wallet 1, not 0
            0,
        );

        let actions = strategy.enumerate_candidate_actions(&view);

        // Should return Wait because only Wallet 0 can initiate
        assert_eq!(actions.len(), 1);
        assert!(matches!(actions[0], Action::Wait));
    }

    #[test]
    fn test_multiparty_participant_with_new_invitation() {
        let strategy = MultipartyPayjoinParticipantStrategy;

        let po = PaymentObligationData {
            id: PaymentObligationId(0),
            deadline: TimeStep(100),
            reveal_time: TimeStep(0),
            amount: Amount::from_sat(1000),
            from: WalletId(0),
            to: WalletId(1),
        };

        // Create view with a new multi-party payjoin invitation
        let view = WalletView::new(
            vec![po],
            vec![],
            vec![(BulletinBoardId(0), MessageId(0))], // New invitation
            vec![],                                   // No active sessions yet
            TimeStep(0),
            WalletId(1),
            0,
        );

        let actions = strategy.enumerate_candidate_actions(&view);

        assert_eq!(actions.len(), 1);
        assert!(actions
            .iter()
            .any(|a| matches!(a, Action::ParticipateMultiPartyPayjoin(_))));
    }

    #[test]
    fn test_multiparty_participant_with_active_session() {
        let strategy = MultipartyPayjoinParticipantStrategy;

        let po = PaymentObligationData {
            id: PaymentObligationId(0),
            deadline: TimeStep(100),
            reveal_time: TimeStep(0),
            amount: Amount::from_sat(1000),
            from: WalletId(0),
            to: WalletId(1),
        };

        // Create view with an active session
        let view = WalletView::new(
            vec![po],
            vec![],
            vec![],
            vec![BulletinBoardId(0)], // Active session
            TimeStep(0),
            WalletId(1),
            0,
        );

        let actions = strategy.enumerate_candidate_actions(&view);

        assert_eq!(actions.len(), 1);
        assert!(actions
            .iter()
            .any(|a| matches!(a, Action::ContinueParticipateMultiPartyPayjoin(_))));
    }

    #[test]
    fn test_multiparty_participant_prefers_continue_when_invite_and_active() {
        let strategy = MultipartyPayjoinParticipantStrategy;

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
            vec![],
            vec![(BulletinBoardId(0), MessageId(0))],
            vec![BulletinBoardId(1)],
            TimeStep(0),
            WalletId(1),
            0,
        );

        let actions = strategy.enumerate_candidate_actions(&view);

        assert_eq!(actions.len(), 1);
        assert!(matches!(
            actions[0],
            Action::ContinueParticipateMultiPartyPayjoin(BulletinBoardId(1))
        ));
    }
}
