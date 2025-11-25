use std::{iter::Sum, ops::Add};

use bitcoin::Amount;

use crate::{
    message::{MessageData, MessageType, PayjoinProposal},
    wallet::{PaymentObligationData, PaymentObligationId, WalletHandleMut},
    TimeStep,
};

/// An Action a wallet can perform
pub(crate) enum Action {
    UnilateralSpend(PaymentObligationId),
    InitiatePayjoin(PaymentObligationId),
    ParticipateInPayjoin(PayjoinProposal),
    Wait,
}

/// Hypothetical outcomes of an action
pub(crate) enum Event {
    PaymentObligationHandled(PaymentObligationHandledEvent),
    InitiatePayjoin(InitiatePayjoinEvent),
}

pub(crate) struct PaymentObligationHandledEvent {
    /// Payment obligation amount
    amount_handled: Amount,
    /// Balance difference after the action
    balance_difference: Amount,
    /// Time left on the payment obligation
    time_left: i32,
}

impl PaymentObligationHandledEvent {
    fn score(&self, payment_obligation_utility_factor: f64) -> ActionScore {
        let deadline_anxiety = self.time_left.pow(3) as f64 / 50.0;
        ActionScore(
            self.balance_difference
                .to_float_in(bitcoin::Denomination::Satoshi)
                - (payment_obligation_utility_factor
                    * deadline_anxiety
                    * self
                        .amount_handled
                        .to_float_in(bitcoin::Denomination::Satoshi)),
        )
    }
}

pub(crate) struct InitiatePayjoinEvent {
    /// Time left on the payment obligation
    time_left: i32,
    /// Amount of the payment obligation
    amount_handled: Amount,
    /// Balance difference after the action
    balance_difference: Amount,
    /// Fee savings from the payjoin
    fee_savings: Amount,
    // TODO: somekind of privacy gained metric?
}

impl InitiatePayjoinEvent {
    /// Batching anxiety should increase the closer the deadline is.
    /// This can be modeled as a inverse cubic function of the time left.
    /// TODO: how do we model potential fee savings? Understanding that at most there will be one input and one output added could lead to a simple linear model.
    fn score(&self, batching_anxiety_factor: f64) -> ActionScore {
        let anxiety = batching_anxiety_factor * (self.time_left.pow(3) as f64 / 50.0) * -1.0;
        let score = self
            .balance_difference
            .to_float_in(bitcoin::Denomination::Satoshi)
            - (self
                .amount_handled
                .to_float_in(bitcoin::Denomination::Satoshi)
                * anxiety);

        ActionScore(score)
    }
}

// TODO: implement EventCost for each event, each trait impl should define its own lambda weights

// Each strategy will prioritize some specific actions over other to minimize its wallet cost function
// E.g the unilateral spender associates high cost with batched transaction perhaps bc they dont like interactivity and don't care much for privacy
// They want to ensure they never miss a deadline. In that case the weights behind their deadline misses are high and batched payments will be low. i.e high payment anxiety
// TODO: should strategies do more than one thing per timestep?

// Cost function should evalute over unhandled payment obligations and payjoin / cospend oppurtunities. i.e Given all the payment obligations

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

fn simulate_one_action(wallet_handle: &WalletHandleMut, action: &Action) -> Vec<Event> {
    let wallet_view = wallet_handle.wallet_view();
    let mut events = vec![];
    let old_info = wallet_handle.info().clone();
    let old_balance = wallet_handle.handle().effective_balance();

    // Deep clone the simulation
    let mut sim = wallet_handle.sim.clone();
    let mut predicated_wallet_handle = wallet_handle.data().id.with_mut(&mut sim);
    predicated_wallet_handle.do_action(action);
    let new_info = wallet_handle.info().clone();
    let new_balance = wallet_handle.handle().effective_balance();

    // Check for handled payment obligations -- we only handle one payment obligatoin per action. This may change in the future.
    // We may also want to evaluate bundles of actions.
    let handled_payment_obligations_diff = old_info
        .handled_payment_obligations
        .difference(new_info.handled_payment_obligations.clone())
        .into_iter()
        .next();
    if let Some(payment_obligation) = handled_payment_obligations_diff {
        let payment_obligation = payment_obligation.with(&sim).data();
        let deadline = payment_obligation.deadline;

        events.push(Event::PaymentObligationHandled(
            PaymentObligationHandledEvent {
                amount_handled: payment_obligation.amount,
                balance_difference: old_balance - new_balance,
                time_left: deadline.0 as i32 - wallet_view.current_timestep.0 as i32,
            },
        ));
    }

    // Check if the wallet initiated a payjoin
    let old_payment_obligation_to_payjoin = old_info.payment_obligation_to_payjoin;
    let new_payment_obligation_to_payjoin = new_info.payment_obligation_to_payjoin;
    if let Some((payment_obligation_id, _)) = old_payment_obligation_to_payjoin
        .iter()
        .filter(|(payment_obligation_id, _)| {
            !new_payment_obligation_to_payjoin.contains_key(payment_obligation_id)
        })
        .into_iter()
        .next()
    {
        let po = payment_obligation_id.with(&sim).data();
        events.push(Event::InitiatePayjoin(InitiatePayjoinEvent {
            time_left: po.deadline.0 as i32 - wallet_view.current_timestep.0 as i32,
            amount_handled: po.amount,
            balance_difference: old_balance - new_balance,
            fee_savings: Amount::ZERO, // TODO: implement this
        }));
    }

    // TODO: check if we processed any messages and create events for payjoins that were participated in

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
                    actions.push(Action::ParticipateInPayjoin(payjoin_proposal.clone()));
                }
                _ => (),
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
        let mut score = ActionScore(0.0);
        for event in events {
            match event {
                Event::PaymentObligationHandled(event) => {
                    score = score + event.score(self.payment_obligation_utility_factor);
                }
                Event::InitiatePayjoin(event) => {
                    score = score + event.score(self.payjoin_utility_factor);
                }
            }
        }
        score
    }
}
