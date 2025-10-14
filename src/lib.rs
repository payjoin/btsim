use bdk_coin_select::{Target, TargetFee, TargetOutputs};
use bitcoin::{Amount, Weight};
use im::{OrdMap, OrdSet, Vector};
use rand::SeedableRng;
use rand_chacha::ChaChaRng;

use crate::{
    blocks::{
        BlockData, BlockHandle, BlockId, BlockInfo, BroadcastSetData, BroadcastSetHandleMut,
        BroadcastSetId, BroadcastSetInfo,
    },
    transaction::{Input, Outpoint, Output, TxData, TxHandle, TxId, TxInfo},
    wallet::{
        AddressData, AddressId, PaymentObligationData, PaymentObligationId, WalletData, WalletId,
        WalletInfo, WalletInfoId,
    },
};

#[macro_use]
mod macros;
mod blocks;
mod transaction;
mod wallet;

// TODO use bitcoin::transaction::predict_weight(inputs: IntoIter<InputWeightPrediction>, output_lengths: IntoIter<u64>)

// all have RBF and non-RBF variants?
enum CoinSelectionStrategy {
    FIFO,
    SpendAll,
    BNB,
    // TODO brute force pre-computed for cost function
}

// total fee budget
//   - cap average over entire history, to work within estimated budget overall
//     - this is a soft fail, resulting in missed payments
//     - failure mode is broadcasting highest possible feerate at deadline, miss by time it takes to confirm

// cost function evaluates:
// - do nothings vs. unilateral build txn vs. build multiparty txn
// - if payment nearing deadline, sign tx discharging it
//   - immediately broadcast min relay fee txn based on deadline anxiety
//     - ... if it were not for privacy loss terms:
//       - desire not to unilaterally link inputs
//       - desire to minimize RBF / double spending / failed coinjoin sessions
//       - agent time preference and fallback strategy dictate balance between these
//   - batching strategy: powerset over payment obligations, loss diminishes with set size, and avoid evaluating below threshold
//   - link aversive strategy:
//     - never in unilateral txs
//     - simulate independent clients in multiparty txs
//
// experience measured loss as well based on how much the deadline was missed by? or just measure error objectively?

// Clock iterations:
// - new block
// - coinjoin opportunity alternating with new block. all agents available at all times.
// - listening for coinjoins
// - agents sleeping, simulated wallclock time
//   - parameterize with zef results?

// Lookup tables, generate in O(1) space without repetition as joins (datalog in rust thing?)
// - denomination combinations by size, ordered set of tiny vectors indicating denomination combinations
// - coin selection table, combinations of coins (how to limit? random ordering?)
// - other wallets' input combinations (up to size 2-3), cardinality estimation by counting quotient filter?

// Decomposition
// - given predefined outputs (payment targets)
//   - later, account for address reuse likelyhood in failed txn context?
//   - what about breakdown to unilateral spend on timeout?
// - take power set of candidate output set
// - sum density over a window, until some saturation limit
// - loss is (limit - value)/limit ?

// Peer coin selection
// - randomized score based on chain tip, xor metric over pairs
// - hypotehtical input combinations -> hypothetical decomposition evaluation
//
//

// https://ishaana.com/blog/wallet_fingerprinting/fingerprints_final.png

// petgraph tx graph
// rustworkx economic graph
// cost function for payments
// timestep abstraction, track txn broadcast
// mining, mempool
// vector clock of (top level) entity IDs? with nested spans?

// agent has state, re-evaluates at every timestep
// enumerate payment obligations
// check fees
// calculate cost for meeting obligation
// RBF or create txs estimated to meet deadline
// coin selection:
// - min size
// - core
// - privacy

// TODO AddressType extend with sizes
//
// OrdMap, OrdSet -> HashMap HashSet - where?
// just accept randomization and test that simulation is replicable even with

// TODO data() and info() fetchers from handle, deref into touple?
// TODO break down into define_id, define_handle, define_handle_mut, define_data, define_info, define_info_id
// TODO define_sequenced_entity (broadcast set, monad-ish) vs. define_mut_entity (wallet, append only updates)
// TODO handle enum for broadcastset data?
//

// TODO: unsued do we need this?
// #[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
// struct TxByFeerate(FeeRate, TxId);

/// Wrapper type for timestep index
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub(crate) struct Epoch(usize);

#[derive(Debug)]
struct SimulationBuilder {
    prng: ChaChaRng,
    /// Number of wallets/agents in the simulation
    num_wallets: usize,
    /// Total number of epochs for the simulation
    max_epochs: usize,
    /// How many blocks are mined between epochs
    block_interval: usize,
}

impl SimulationBuilder {
    fn new_random(num_wallets: usize, max_epochs: usize, block_interval: usize) -> Self {
        debug_assert!(num_wallets >= 2);
        let chacha = ChaChaRng::from_rng(rand::thread_rng()).unwrap();
        Self {
            prng: chacha,
            num_wallets,
            max_epochs,
            block_interval,
        }
    }

    fn new(seed: u64, num_wallets: usize, max_epochs: usize, block_interval: usize) -> Self {
        debug_assert!(num_wallets >= 2);
        let chacha = ChaChaRng::seed_from_u64(seed);
        Self {
            prng: chacha,
            num_wallets,
            max_epochs,
            block_interval,
        }
    }

    fn build(self) -> Simulation {
        let mut sim = Simulation::new();

        for _ in 0..self.num_wallets {
            sim.new_wallet();
        }

        sim
    }
}

/// all entities are numbered sequentially
#[derive(Debug, PartialEq, Eq, Default)] // TODO remove Default
struct Simulation {
    // primary information
    wallet_data: Vec<WalletData>,
    payment_data: Vec<PaymentObligationData>,
    address_data: Vec<AddressData>,
    tx_data: Vec<TxData>, // all are implicitly broadcast for now
    broadcast_set_data: Vec<BroadcastSetData>,
    // TODO mempools, = orderings / replacements of broadcast_sets
    block_data: Vec<BlockData>,

    // secondary information (indexes)
    spends: OrdMap<Outpoint, OrdSet<TxId>>,
    wallet_info: Vec<WalletInfo>,
    block_info: Vec<BlockInfo>,
    tx_info: Vec<TxInfo>,
    broadcast_set_info: Vec<BroadcastSetInfo>,
}

impl<'a> Simulation {
    fn new() -> Self {
        let mut sim = Simulation::default();

        // genesis block has empty coinbase
        sim.tx_data.push(TxData::default());
        sim.tx_info.push(TxInfo {
            fee: Amount::from_sat(0),
            weight: Weight::from_wu(0),
        });
        sim.block_data.push(BlockData {
            parent: None,
            coinbase_tx: TxId(0),
            confirmed_txs: vec![],
        });
        sim.block_info.push(BlockInfo {
            height: 0,
            spent: OrdSet::default(),
            created: OrdSet::default(),
            utxos: OrdSet::default(),
            all_confirmed_txs: OrdSet::default(),
            confirmed_txs: OrdSet::default(),
        });

        // empty initial broadcast set
        sim.broadcast_set_data
            .push(BroadcastSetData::Block(sim.genesis_block()));
        sim.broadcast_set_info.push(BroadcastSetInfo {
            parent_id: None,
            chain_tip_id: sim.genesis_block(),
            unconfirmed_txs: OrdSet::default(),
            invalidated_txs: OrdSet::default(),
        });

        sim
    }

    fn build_universe(&mut self) {
        let mut wallets = self.wallet_data.clone();
        let addresses = wallets
            .iter()
            .map(|w| self.new_address(w.id))
            .collect::<Vec<_>>();

        // For now we just mine a coinbase transaction for each wallet
        let mut i = 0;
        for (wallet, address) in wallets.iter_mut().zip(addresses.iter()) {
            let broadcast_set = BroadcastSetHandleMut {
                id: BroadcastSetId(i),
                sim: self,
            };
            let coinbase_tx = broadcast_set
                .construct_block_template(Weight::MAX_BLOCK)
                .mine(*address, self);

            wallet.own_transactions.push(coinbase_tx.coinbase_tx().id);

            self.assert_invariants();
            i += 1;
        }

        // In this universe we only have payment intent between the first (alice) and second (bob) wallet
        // TODO: in the future payment obligations will be randomized between the wallet and their amounts, as well as deadlines
        let payment = PaymentObligationData {
            amount: Amount::from_int_btc(20),
            from: wallets[0].id,
            to: wallets[1].id,
            deadline: Epoch(2), // TODO 102
                                // TODO coinbase maturity
        };
        self.payment_data.push(payment);
        self.assert_invariants();
    }

    fn tick(&mut self) {
        let payment_obligation = self.payment_data.pop().unwrap();
        let to_wallet_id = payment_obligation.to.with(self).clone().id;
        let from_wallet_id = payment_obligation.from.with_mut(self).clone().id;

        let change_addr = self.new_address(from_wallet_id);
        let to_addr = self.new_address(to_wallet_id);

        let target = Target {
            fee: TargetFee {
                rate: bdk_coin_select::FeeRate::from_sat_per_vb(1.0),
                replace: None,
            },
            outputs: TargetOutputs {
                value_sum: payment_obligation.amount.to_sat(),
                weight_sum: 34, // TODO use payment.to to derive an address, payment.into() ?
                n_outputs: 1,
            },
        };

        let long_term_feerate = bitcoin::FeeRate::from_sat_per_vb(10).unwrap();
        let mut from_wallet = from_wallet_id.with_mut(self);
        let (selected_coins, drain) = from_wallet.select_coins(target, long_term_feerate);

        let spend = from_wallet
            .new_tx(|tx, _| {
                tx.inputs = selected_coins
                    .map(|o| Input {
                        outpoint: o.outpoint,
                    })
                    .collect();

                tx.outputs = vec![
                    Output {
                        amount: payment_obligation.amount,
                        address_id: to_addr,
                    },
                    Output {
                        amount: Amount::from_sat(drain.value),
                        address_id: change_addr,
                    },
                ];
            })
            .id;

        from_wallet.broadcast(std::iter::once(spend));
    }

    fn genesis_block(&self) -> BlockId {
        BlockId(0)
    }

    // TODO remove
    fn get_block(&'a self, id: BlockId) -> BlockHandle<'a> {
        id.with(&self)
    }

    // TODO remove
    fn get_tx(&'a self, id: TxId) -> TxHandle<'a> {
        id.with(&self)
    }

    fn new_wallet(&mut self) -> WalletId {
        // TODO wallet_handle?
        let last_wallet_info_id = WalletInfoId(self.wallet_info.len());
        self.wallet_info.push(WalletInfo {
            broadcast_set_id: BroadcastSetId(self.broadcast_set_data.len() - 1), // FIXME refactor
            payment_obligations: OrdSet::<PaymentObligationId>::default(),
            expected_payments: OrdSet::<PaymentObligationId>::default(),
            broadcast_transactions: Vector::<TxId>::default(),
            received_transactions: Vector::<TxId>::default(),
            unconfirmed_transactions: OrdSet::<TxId>::default(),
            unconfirmed_txos: OrdSet::<Outpoint>::default(),
            confirmed_utxos: OrdSet::<Outpoint>::default(),
            unconfirmed_spends: OrdSet::<Outpoint>::default(),
        });

        let id = WalletId(self.wallet_data.len());
        self.wallet_data.push(WalletData {
            id,
            last_wallet_info_id,
            addresses: Vec::default(),
            own_transactions: Vec::default(),
        });
        id
    }

    // TODO move to wallet impl?
    fn new_address(&mut self, owner: WalletId) -> AddressId {
        let id = AddressId(self.address_data.len());
        self.address_data.push(AddressData { wallet_id: owner });
        id
    }

    fn new_tx<F>(&mut self, build: F) -> TxId
    where
        F: FnOnce(&mut TxData, &Simulation),
    {
        let txid = TxId(self.tx_data.len());
        let mut tx = TxData::default();

        build(&mut tx, &self);

        let tx_info = TxInfo::new(&tx, self);

        // TODO check all inputs unspent

        // TODO check transaction validity, calculate input values, feerate, weight

        for i in &tx.inputs {
            if !self.spends.contains_key(&i.outpoint) {
                self.spends.insert(i.outpoint, OrdSet::<TxId>::default());
            }
            self.spends[&i.outpoint].insert(txid);
        }
        self.tx_data.push(tx);
        self.tx_info.push(tx_info);

        txid
    }

    fn new_block(&'a mut self, data: BlockData, info: BlockInfo) -> BlockHandle<'a> {
        let id = BlockId(self.block_data.len());

        self.block_data.push(data);
        self.block_info.push(info); // TODO compute this here by accepting BlockTemplate and coinbase_tx?

        // TODO refactor, return mut handle from process_block, clean up IDs
        BroadcastSetId(self.broadcast_set_data.len() - 1)
            .with_mut(self)
            .process_block(id);

        id.with(self)
    }

    fn broadcast(&'a mut self, txs: impl IntoIterator<Item = TxId>) -> BroadcastSetHandleMut<'a> {
        // TODO BroadcastSetHandle
        let bx_id = BroadcastSetId(self.broadcast_set_data.len() - 1);
        bx_id.with_mut(self).broadcast(txs)
    }

    // FIXME debug only code?
    fn assert_invariants(&self) {
        assert!(self
            .broadcast_set_info
            .last()
            .unwrap()
            .unconfirmed_txs
            .clone()
            .intersection(self.block_info.last().unwrap().all_confirmed_txs.clone())
            .is_empty());

        self.wallet_info.iter().for_each(|w| {
            assert!(w
                .confirmed_utxos
                .clone()
                .intersection(w.unconfirmed_txos.clone())
                .is_empty());
        });

        self.wallet_info.iter().for_each(|w| {
            assert!(
                OrdSet::<TxId>::from_iter(w.broadcast_transactions.clone().into_iter())
                    .intersection(OrdSet::from_iter(
                        w.received_transactions.clone().into_iter()
                    ))
                    .is_empty()
            );
        });

        // TODO for all wallets, ensure their confirmed and unconfirmed utxos form a partition (their intersections are empty and their union is describes the corresponding block info and broadcast state)
        // take union and compare size to sum of sizes, and check equality with global structures
    }
}

#[cfg(test)]
mod tests {
    use bdk_coin_select::{Target, TargetFee, TargetOutputs};
    use im::{ordset, vector};

    use crate::transaction::{Input, Output};

    use super::*;

    #[test]
    fn it_works() {
        let mut sim = SimulationBuilder::new(42, 2, 10, 1).build();
        sim.assert_invariants();

        let alice = WalletId(0);
        let bob = WalletId(1);

        sim.build_universe();

        let coinbase_tx = alice.with(&sim).data().own_transactions[0].with(&sim).id;

        sim.tick();
        sim.assert_invariants();

        // TODO: tick creates the unconfirmed tx and then spends, these asserts have been commented out bc we cannot check the status of unconfirmed txs
        // Until we have a mining entity
        // assert_eq!(spend, TxId(3));

        // assert_eq!(spend.with(&sim).info().weight, Weight::from_wu(688));

        // assert_eq!(
        //     alice.with(&sim).data().own_transactions,
        //     vec![coinbase_tx, spend]
        // );

        // assert_eq!(
        //     alice.with(&sim).info().broadcast_transactions,
        //     Vector::default()
        // );

        // // these fields are not updated until broadcast
        // assert_eq!(
        //     alice.with(&sim).info().confirmed_utxos,
        //     OrdSet::from_iter(coinbase_tx.with(&sim).outpoints())
        // );
        // assert_eq!(alice.with(&sim).info().unconfirmed_spends, ordset![]);

        // alice.with_mut(&mut sim).broadcast(std::iter::once(spend));

        let spend = alice.with_mut(&mut sim).data().own_transactions[1];

        assert_eq!(
            alice.with(&sim).info().unconfirmed_spends,
            OrdSet::from_iter(coinbase_tx.with(&sim).outpoints())
        );

        assert_eq!(
            alice.with(&sim).info().unconfirmed_txos,
            OrdSet::from_iter(spend.with(&sim).outpoints().skip(1))
        );

        assert_eq!(
            bob.with(&sim).info().unconfirmed_txos,
            OrdSet::from_iter(spend.with(&sim).outpoints().take(1))
        );

        assert_eq!(
            alice.with(&sim).info().broadcast_transactions,
            vector![spend]
        );

        assert!(bob.with(&sim).info().received_transactions.contains(&spend));

        // TODO mine another block, check wallet utxos, et

        println!("{:?}", sim);
    }
}
