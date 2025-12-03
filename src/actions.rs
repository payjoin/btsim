use std::{iter::Sum, ops::Add};

use bitcoin::Amount;
use log::debug;

use crate::{
    message::{MessageData, MessageId, MessageType, PayjoinProposal},
    wallet::{PaymentObligationData, PaymentObligationId, WalletHandleMut},
    TimeStep,
};

/// An Action a wallet can perform
#[derive(Debug)]
pub(crate) enum Action {
    /// Spend a payment obligation unilaterally
    UnilateralSpend(PaymentObligationId),
    /// Initiate a payjoin with a counterparty
    InitiatePayjoin(PaymentObligationId),
    /// respond to a payjoin proposal
    RespondToPayjoin(PayjoinProposal, PaymentObligationId, MessageId),
    /// Do nothing. There may be better oppurtunities to spend a payment obligation or participate in a payjoin.
    Wait,
}

/// Hypothetical outcomes of an action
#[derive(Debug)]
pub(crate) enum PredictedOutcome {
    PaymentObligationHandled(PaymentObligationHandledOutcome),
    InitiatePayjoin(InitiatePayjoinOutcome),
    RespondToPayjoin(RespondToPayjoinOutcome),
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
        let utility = {
            if self.time_left > 5 {
                0.0
            } else if self.time_left <= 5 && self.time_left > 2 {
                payment_obligation_utility_factor
            } else if self.time_left <= 2 && self.time_left > 0 {
                payment_obligation_utility_factor * 2.0
            } else {
                // Overdue
                payment_obligation_utility_factor * 10.0
            }
        };
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
    // TODO: somekind of privacy gained metric?
}

impl InitiatePayjoinOutcome {
    /// Batching anxiety should increase and payjoin utility should decrease the closer the deadline is.
    /// This can be modeled as a inverse cubic function of the time left.
    /// TODO: how do we model potential fee savings? Understanding that at most there will be one input and one output added could lead to a simple linear model.
    fn score(&self, payjoin_utility_factor: f64) -> ActionScore {
        let utility = {
            if self.time_left > 5 {
                payjoin_utility_factor * 5.0
            } else if self.time_left <= 5 && self.time_left >= 2 {
                // Riskier to initiate a payjoin the closer the deadline is
                payjoin_utility_factor
            } else {
                // Overdue, should not prefer to initiate a payjoin
                0.0
            }
        };
        let score = self.balance_difference + (self.amount_handled * utility);
        debug!("InitiatePayjoinEvent score: {:?}", score);
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

/// State of the wallet that can be used to potential enumerate actions
#[derive(Debug, Default)]
pub(crate) struct WalletView {
    payment_obligations: Vec<PaymentObligationData>,
    messages: Vec<MessageData>,
    current_timestep: TimeStep,
    // TODO: utxos, feerate, cospend oppurtunities, etc.
}

impl WalletView {
    pub(crate) fn new(
        payment_obligations: Vec<PaymentObligationData>,
        messages: Vec<MessageData>,
        current_timestep: TimeStep,
    ) -> Self {
        Self {
            payment_obligations,
            messages,
            current_timestep,
        }
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
        events.push(PredictedOutcome::PaymentObligationHandled(
            PaymentObligationHandledOutcome {
                amount_handled: payment_obligation
                    .amount
                    .to_float_in(bitcoin::Denomination::Satoshi),
                balance_difference,
                time_left: deadline.0 as i32 - wallet_view.current_timestep.0 as i32,
            },
        ));
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
            fee_savings: Amount::ZERO, // TODO: implement this
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
pub(crate) trait Strategy {
    fn enumerate_candidate_actions(&self, state: &WalletView) -> Vec<Action>;
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
}

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
        for message in state.messages.iter() {
            match &message.message {
                MessageType::InitiatePayjoin(payjoin_proposal) => {
                    // We should evaluate responding using all payment obligations that have not been handled
                    for po in state.payment_obligations.iter() {
                        actions.push(Action::RespondToPayjoin(
                            payjoin_proposal.clone(),
                            po.id,
                            message.id,
                        ));
                    }
                }
            }
        }

        actions
    }
}

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
}
// TODO: this should be a trait once we have different scoring strategies
pub(crate) struct CompositeScorer {
    pub(crate) payjoin_utility_factor: f64,
    pub(crate) payment_obligation_utility_factor: f64,
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
                PredictedOutcome::PaymentObligationHandled(event) => {
                    score = score + event.score(self.payment_obligation_utility_factor);
                }
                PredictedOutcome::InitiatePayjoin(event) => {
                    score = score + event.score(self.payjoin_utility_factor);
                }
                PredictedOutcome::RespondToPayjoin(event) => {
                    score = score + event.score(self.payjoin_utility_factor);
                }
            }
        }
        score
    }
}
