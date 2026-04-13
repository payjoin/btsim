use bdk_coin_select::{
    metrics::LowestFee, Candidate, ChangePolicy, CoinSelector, DrainWeights, Target,
    TR_DUST_RELAY_MIN_VALUE,
};
use log::warn;

use crate::transaction::Outpoint;

pub(crate) struct CoinCandidate {
    pub(crate) outpoint: Outpoint,
    pub(crate) amount_sats: u64,
    pub(crate) weight_wu: u32,
    pub(crate) is_segwit: bool,
}

/// Long-term feerate for coin selection (10 sat/vb = 2.5 sat/wu).
pub(crate) fn long_term_feerate() -> bdk_coin_select::FeeRate {
    bdk_coin_select::FeeRate::from_sat_per_wu(2.5)
}

/// Run BNB coin selection over candidates for the given target.
/// Falls back to greedy selection if BNB finds no solution.
/// Returns None if no selection can meet the target.
pub(crate) fn select_bnb(candidates: &[CoinCandidate], target: Target) -> Option<Vec<Outpoint>> {
    let bdk_candidates: Vec<Candidate> = candidates
        .iter()
        .map(|c| Candidate {
            value: c.amount_sats,
            weight: c.weight_wu,
            input_count: 1,
            is_segwit: c.is_segwit,
        })
        .collect();

    let mut coin_selector = CoinSelector::new(&bdk_candidates);

    let drain_weights = DrainWeights::default();
    let dust_limit = TR_DUST_RELAY_MIN_VALUE;
    let ltfr = long_term_feerate();
    let change_policy =
        ChangePolicy::min_value_and_waste(drain_weights, dust_limit, target.fee.rate, ltfr);
    let metric = LowestFee {
        target,
        long_term_feerate: ltfr,
        change_policy,
    };

    if let Err(err) = coin_selector.run_bnb(metric, 100_000) {
        warn!("BNB failed to find a solution: {}", err);
        if coin_selector.select_until_target_met(target).is_err() {
            return None;
        }
    }

    Some(
        coin_selector
            .apply_selection(candidates)
            .map(|c| c.outpoint)
            .collect(),
    )
}

/// Return all candidate outpoints (consolidation / spend-all strategy).
pub(crate) fn select_all(candidates: &[CoinCandidate]) -> Vec<Outpoint> {
    candidates.iter().map(|c| c.outpoint).collect()
}
