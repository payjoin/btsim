use crate::transaction::Outpoint;
use crate::wallet::WalletId;
use bitcoin::Amount;

#[derive(Debug, Clone)]
pub(crate) struct UtxoWithAmount {
    pub(crate) outpoint: Outpoint,
    pub(crate) amount: Amount,
}

/// A UTXO from the order book
#[derive(Debug, Clone)]
pub(crate) struct OrderBookEntry {
    pub(crate) utxo: UtxoWithAmount,
    pub(crate) owner: WalletId,
}

fn amount_distance(a: Amount, b: Amount) -> u64 {
    a.to_sat().abs_diff(b.to_sat())
}

/// Returns order book entries sorted by value asymmetry relative to the taker's UTXOs.
/// Each entry is scored by the minimum amount distance to any taker UTXO,
/// so the best-matched makers appear first.
pub(crate) fn generate_candidates(
    order_book: &[OrderBookEntry],
    taker_utxos: &[UtxoWithAmount],
) -> Vec<OrderBookEntry> {
    let mut scored: Vec<(u64, &OrderBookEntry)> = order_book
        .iter()
        .map(|entry| {
            let min_dist = taker_utxos
                .iter()
                .map(|t| amount_distance(entry.utxo.amount, t.amount))
                .min()
                .unwrap_or(u64::MAX);
            (min_dist, entry)
        })
        .collect();

    scored.sort_unstable_by(|(dist_a, a), (dist_b, b)| {
        dist_a
            .cmp(dist_b)
            .then_with(|| a.utxo.outpoint.txid.0.cmp(&b.utxo.outpoint.txid.0))
            .then_with(|| a.utxo.outpoint.index.cmp(&b.utxo.outpoint.index))
    });

    scored.into_iter().map(|(_, entry)| entry.clone()).collect()
}
