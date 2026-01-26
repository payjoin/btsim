use std::{collections::HashSet, iter::Sum, ops::Add};

use bitcoin::Amount;
use log::debug;

use crate::{
    bulletin_board::BulletinBoardId,
    message::{MessageId, PayjoinProposal},
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
}

#[derive(Debug)]
pub(crate) struct PaymentObligationHandledOutcome {
    /// Payment obligation amount
    amount_handled: f64,
    /// Balance difference after the action
    balance_difference: f64,
    /// Time left on the payment obligation
    time_left: i32,
}

impl PaymentObligationHandledOutcome {
    fn score(&self, payment_obligation_utility_factor: f64) -> ActionScore {
        let points = [
            (0.0, 2.0 * payment_obligation_utility_factor),
            (2.0, payment_obligation_utility_factor),
            (5.0, 0.0),
        ];
        let utility = piecewise_linear(self.time_left as f64, &points);
        let score = self.balance_difference + (self.amount_handled * utility);
        debug!("PaymentObligationHandledEvent score: {:?}", score);
        ActionScore(score)
    }
}

#[derive(Debug)]
pub(crate) struct InitiatePayjoinOutcome {
    /// Time left on the payment obligation
    time_left: i32,
    /// Amount of the payment obligation
    amount_handled: f64,
    /// Balance difference after the action
    balance_difference: f64,
    /// Fee savings from the payjoin
    fee_savings: Amount,
    /// Privacy score based on input/output mixing and timing analysis resistance
    privacy_score: f64,
}

impl InitiatePayjoinOutcome {
    /// Batching anxiety should increase and payjoin utility should decrease the closer the deadline is.
    /// This can be modeled as a inverse cubic function of the time left.
    /// Fee savings are modeled linearly based on the additional input/output structure of payjoins.
    fn score(&self, payjoin_utility_factor: f64) -> ActionScore {
        let points = [
            (0.0, 0.0),
            (2.0, payjoin_utility_factor),
            (5.0, 5.0 * payjoin_utility_factor),
        ];
        let utility = piecewise_linear(self.time_left as f64, &points);

        // Base utility score
        let base_score = self.balance_difference + (self.amount_handled * utility);

        // Add fee savings benefit (convert to float for calculation)
        let fee_benefit = self.fee_savings.to_float_in(bitcoin::Denomination::Satoshi);

        // Add privacy benefit (weighted by utility factor)
        let privacy_benefit = self.privacy_score * payjoin_utility_factor;

        let score = base_score + fee_benefit + privacy_benefit;
        debug!("InitiatePayjoinEvent score: {:?} (base: {:?}, fee: {:?}, privacy: {:?})",
               score, base_score, fee_benefit, privacy_benefit);
        ActionScore(score)
    }
}

#[derive(Debug)]
pub(crate) struct RespondToPayjoinOutcome {
    /// Amount of the payment obligation
    amount_handled: f64,
    /// Balance difference after the action
    balance_difference: f64,
    /// Fee savings from the payjoin
    fee_savings: Amount,
}

impl RespondToPayjoinOutcome {
    fn score(&self, payjoin_utility_factor: f64) -> ActionScore {
        // Responding to a payjoin should always be better than unilaterally spending at this point
        // As there is no interaction cost. TODO in the future we will want to model the cost of doing the last round of interaction with the counterparty

        // Since there is no final interaction cost, we can just score the balance difference and the amount handled
        // However the utility should be higher for fee saving an a privacy preservation.
        // TODO These last two are not factored in yet.
        let score = self.balance_difference + (payjoin_utility_factor * self.amount_handled);
        debug!("RespondToPayjoinEvent score: {:?}", score);

        ActionScore(score)
    }
}

#[derive(Debug)]
pub(crate) struct InitiateMultiPartyPayjoinOutcome {
    /// Time left on the payment obligation
    time_left: i32,
    /// Amount of the payment obligation
    amount_handled: f64,
    /// Balance difference after the action
    balance_difference: f64,
    /// Upper bound on the number of participants in the multi-party payjoin
    max_participants: u32,
}

impl InitiateMultiPartyPayjoinOutcome {
    fn score(&self, multi_party_payjoin_utility_factor: f64) -> ActionScore {
        // For now the score for initiating a multi-party payjoin is really high so it always happens no matter what
        let score = self.amount_handled * 100.0;
        debug!("InitiateMultiPartyPayjoinEvent score: {:?}", score);
        ActionScore(score)
    }
}

#[derive(Debug)]
pub(crate) struct ParticipateMultiPartyPayjoinOutcome {
    /// Time left on the payment obligation
    time_left: i32,
    /// Amount of the payment obligation
    amount_handled: f64,
}

impl ParticipateMultiPartyPayjoinOutcome {
    fn score(&self, multi_party_payjoin_utility_factor: f64) -> ActionScore {
        // TODO: score the participation as a linear function of the progression of the session
        let score = self.amount_handled * 100.0;
        debug!("ParticipateMultiPartyPayjoinEvent score: {:?}", score);
        ActionScore(score)
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
    ) -> Self {
        Self {
            payment_obligations,
            payjoin_proposals,
            active_multi_party_payjoins,
            new_multi_party_payjoins,
            current_timestep,
            wallet_id,
        }
    }
}
/// Calculate fee savings for a payjoin based on the typical structure:
/// - Payjoin adds one input and one output compared to separate transactions
/// - Fee savings = (2 separate txs) - (1 combined tx)
fn calculate_payjoin_fee_savings(amount: f64) -> Amount {
    // Rough estimate: payjoins typically save ~100-200 sats in fees
    // This is a simplified model - in reality it depends on:
    // - Current fee rate
    // - Input/output sizes
    // - Whether batching would have occurred anyway
    let base_savings_sats = 150.0;

    // Larger amounts might justify slightly higher fee savings due to more inputs
    // Cap at 2x for very large amounts
    let amount_factor = (amount / 100000.0).min(2.0);
    let total_savings = (base_savings_sats * (1.0 + amount_factor * 0.2)) as u64;

    Amount::from_sat(total_savings)
}

/// Calculate privacy score for a payjoin
/// Higher scores indicate better privacy benefits
fn calculate_payjoin_privacy_score(amount: f64) -> f64 {
    // Base privacy benefit from transaction structure obfuscation
    let base_privacy = 10.0;

    // Larger amounts get slightly higher privacy scores as they're more valuable to hide
    // Log scaling, capped at 1.0
    let amount_factor = (amount / 100000.0).ln_1p().min(1.0);

    // Random timing component (simplified - in reality depends on network timing)
    let timing_privacy = 2.0;

    base_privacy + (amount_factor * 5.0) + timing_privacy
}

fn get_payment_obligation_handled_outcome(
    payment_obligation_id: &PaymentObligationId,
    sim: &Simulation,
    current_timestep: TimeStep,
) -> PaymentObligationHandledOutcome {
    let payment_obligation = payment_obligation_id.with(&sim).data();
    let deadline = payment_obligation.deadline;
    let balance_difference = payment_obligation
        .amount
        .to_float_in(bitcoin::Denomination::Satoshi)
        * -1.0;
    PaymentObligationHandledOutcome {
        amount_handled: payment_obligation
            .amount
            .to_float_in(bitcoin::Denomination::Satoshi),
        balance_difference,
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

    if let Action::UnilateralSpend(payment_obligation_id) = action {
        let payment_obligation = payment_obligation_id.with(&sim).data();
        let deadline = payment_obligation.deadline;
        let balance_difference = payment_obligation
            .amount
            .to_float_in(bitcoin::Denomination::Satoshi)
            * -1.0;
        events.push(PredictedOutcome::PaymentObligationsHandled(vec![
            PaymentObligationHandledOutcome {
                amount_handled: payment_obligation
                    .amount
                    .to_float_in(bitcoin::Denomination::Satoshi),
                balance_difference,
                time_left: deadline.0 as i32 - wallet_view.current_timestep.0 as i32,
            },
        ]));
    }

    if let Action::BatchSpend(payment_obligation_ids) = action {
        let mut outcomes = vec![];
        for payment_obligation_id in payment_obligation_ids.iter() {
            outcomes.push(get_payment_obligation_handled_outcome(
                payment_obligation_id,
                &sim,
                wallet_view.current_timestep,
            ));
        }
        events.push(PredictedOutcome::PaymentObligationsHandled(outcomes));
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
        let balance_difference = amount_handled * -1.0; // TODO: fee's are not factored in yet
        events.push(PredictedOutcome::InitiatePayjoin(InitiatePayjoinOutcome {
            time_left: po.deadline.0 as i32 - wallet_view.current_timestep.0 as i32,
            amount_handled,
            balance_difference,
            fee_savings: calculate_payjoin_fee_savings(amount_handled),
            privacy_score: calculate_payjoin_privacy_score(amount_handled),
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
        let balance_difference = amount_handled * -1.0; // TODO: fee's are not factored in yet
        events.push(PredictedOutcome::RespondToPayjoin(
            RespondToPayjoinOutcome {
                amount_handled,
                balance_difference,
                fee_savings: Amount::ZERO, // TODO: implement this
            },
        ));
    }

    events
}

/// Strategies will pick one action to minimize their cost
/// TODO: Strategies should be composible. They should enform the action decision space scoring and doing actions should be handling by something else that has composed multiple strategies.
pub(crate) trait Strategy: std::fmt::Debug {
    fn enumerate_candidate_actions(&self, state: &WalletView) -> Vec<Action>;
    fn clone_box(&self) -> Box<dyn Strategy>;
}

#[derive(Debug, PartialEq, PartialOrd)]
pub(crate) struct ActionScore(f64);

impl Sum for ActionScore {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        Self(iter.map(|s| s.0).sum())
    }
}

impl Eq for ActionScore {}

impl Ord for ActionScore {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        assert!(!self.0.is_nan() && !other.0.is_nan());
        self.0.partial_cmp(&other.0).expect("Checked for NaNs")
    }
}

impl Add for ActionScore {
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
    pub(crate) initiate_payjoin_utility_factor: f64,
    pub(crate) respond_to_payjoin_utility_factor: f64,
    pub(crate) payment_obligation_utility_factor: f64,
    pub(crate) multi_party_payjoin_utility_factor: f64,
}

impl CompositeScorer {
    pub(crate) fn score_action(
        &self,
        action: &Action,
        wallet_handle: &WalletHandleMut,
    ) -> ActionScore {
        let events = simulate_one_action(wallet_handle, action);
        // For now each action should only result in one event or none if we are waiting
        // TODO: wallets should evaluate waiting and score it high if they are expecting payments from payjoin compatible wallets
        debug_assert!(events.len() <= 1);
        let mut score = ActionScore(0.0);
        for event in events {
            match event {
                PredictedOutcome::PaymentObligationsHandled(outcomes) => {
                    for outcome in outcomes.iter() {
                        score = score + outcome.score(self.payment_obligation_utility_factor);
                    }
                }
                PredictedOutcome::InitiatePayjoin(event) => {
                    score = score + event.score(self.initiate_payjoin_utility_factor);
                }
                PredictedOutcome::RespondToPayjoin(event) => {
                    score = score + event.score(self.respond_to_payjoin_utility_factor);
                }
                PredictedOutcome::InitiateMultiPartyPayjoin(event) => {
                    score = score + event.score(self.multi_party_payjoin_utility_factor);
                }
                PredictedOutcome::ParticipateMultiPartyPayjoin(event) => {
                    score = score + event.score(self.multi_party_payjoin_utility_factor);
                }
            }
        }
        score
    }
}

/// Creates a strategy instance from its name string
pub(crate) fn create_strategy(name: &str) -> Result<Box<dyn Strategy>, String> {
    match name {
        "UnilateralSpender" => Ok(Box::new(UnilateralSpender)),
        "BatchSpender" => Ok(Box::new(BatchSpender)),
        "PayjoinStrategy" => Ok(Box::new(PayjoinStrategy)),
        "MultipartyPayjoinInitiatorStrategy" => Ok(Box::new(MultipartyPayjoinInitiatorStrategy)),
        "MultipartyPayjoinParticipantStrategy" => {
            Ok(Box::new(MultipartyPayjoinParticipantStrategy))
        }
        _ => Err(format!("Unknown strategy: {}", name)),
    }
}
