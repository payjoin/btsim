use crate::{
    blocks::{BroadcastSetHandleMut, BroadcastSetId},
    Epoch, Simulation,
};
use bdk_coin_select::{
    metrics::LowestFee, Candidate, ChangePolicy, CoinSelector, Drain, DrainWeights, Target,
    TR_DUST_RELAY_MIN_VALUE, TR_KEYSPEND_TXIN_WEIGHT,
};
use bitcoin::{transaction::InputWeightPrediction, Amount};
use im::{OrdSet, Vector};

use crate::transaction::*;

define_entity_mut_updatable!(
    Wallet,
    {
        pub(crate) id: WalletId,
        pub(crate) addresses: Vec<AddressId>,         // TODO split into internal/external?
        pub(crate) own_transactions: Vec<TxId>,       // transactions originating from this wallet
        pub(crate) last_wallet_info_id: WalletInfoId, // Monotone
    },
    {
        pub(crate) broadcast_set_id: BroadcastSetId,
        pub(crate) payment_obligations: OrdSet<PaymentObligationId>,
        pub(crate) expected_payments: OrdSet<PaymentObligationId>,
        pub(crate) broadcast_transactions: Vector<TxId>,
        pub(crate) received_transactions: Vector<TxId>,
        pub(crate) confirmed_utxos: OrdSet<Outpoint>,    // TODO locktimes
        pub(crate) unconfirmed_transactions: OrdSet<TxId>,
        pub(crate) unconfirmed_txos: OrdSet<Outpoint>,  // compute CPFP cost
        pub(crate) unconfirmed_spends: OrdSet<Outpoint>, // RBFable
    }
);

// TODO WalletHandle
impl<'a> WalletHandle<'a> {
    pub(crate) fn data(&self) -> &'a WalletData {
        &self.sim.wallet_data[self.id.0]
    }

    pub(crate) fn info(&self) -> &'a WalletInfo {
        &self.sim.wallet_info[self.data().last_wallet_info_id.0]
    }

    // TODO give utxo list as argument so that different variants can be used
    // TODO return change information
    pub(crate) fn select_coins(
        &self,
        target: Target,
        long_term_feerate: bitcoin::FeeRate,
    ) -> (impl Iterator<Item = OutputHandle<'a>>, Drain) {
        // TODO change
        // TODO group by address
        let utxos: Vec<OutputHandle<'a>> = self.unspent_coins().collect();

        let candidates: Vec<Candidate> = utxos
            .iter()
            .enumerate()
            .map(|(i, o)| Candidate {
                value: o.data().amount.to_sat(),
                weight: TR_KEYSPEND_TXIN_WEIGHT,
                input_count: 1,
                is_segwit: true,
            })
            .collect();

        let mut coin_selector = CoinSelector::new(&candidates);
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

        match coin_selector.run_bnb(metric, 100_000) {
            Err(err) => {
                println!("BNB failed to find a solution: {}", err);

                coin_selector
                    .select_until_target_met(target)
                    .expect("coin selection should always succeed since payments consider budger lower bound");
            }
            Ok(score) => {
                println!("BNB found a solution with score {}", score);
            }
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
    pub(crate) fn data_mut<'b>(&'b mut self) -> &'b mut WalletData {
        &mut self.sim.wallet_data[self.id.0]
    }

    pub(crate) fn new_address(&mut self) -> AddressId {
        let id = self.sim.new_address(self.id);
        self.sim.wallet_data[self.id.0].addresses.push(id);
        id
    }

    pub(crate) fn new_tx<F>(&mut self, build: F) -> TxHandle
    where
        F: FnOnce(&mut TxData, &Simulation),
    {
        let id = self.sim.new_tx(build);
        self.data_mut().own_transactions.push(id);
        TxHandle::new(self.sim, id)
    }

    pub(crate) fn broadcast(
        &'a mut self,
        txs: impl IntoIterator<Item = TxId>,
    ) -> BroadcastSetHandleMut<'a> {
        let mut wallet_info = self.info().clone();

        let txs = Vector::from_iter(txs);

        wallet_info.broadcast_transactions.append(txs.clone());

        // TODO refactor boilerplate for updating wallet ID
        let id = WalletInfoId(self.sim.wallet_info.len());
        self.sim.wallet_info.push(wallet_info);
        let data = self.data_mut();
        data.last_wallet_info_id = id;

        self.sim.broadcast(txs)
    }
}
define_entity!(
    PaymentObligation,
    {
        pub(crate) deadline: Epoch, // block height? time step?
        pub(crate) amount: Amount,
        pub(crate) from: WalletId,
        pub(crate) to: WalletId,
    },
    {
        // TODO coin selection strategy agnostic (pessimal?) spendable balance lower
        // bound
    }
);

define_entity!(Address, {
    pub(crate) wallet_id: WalletId,
    // TODO script_type
    // TODO internal
    // TODO silent payments
}, {});

impl From<AddressData> for InputWeightPrediction {
    fn from(_: AddressData) -> Self {
        // TODO match on script_type
        bitcoin::transaction::InputWeightPrediction::P2TR_KEY_DEFAULT_SIGHASH
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
