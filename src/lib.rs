use bitcoin::{Amount, Weight};
use im::{OrdMap, OrdSet, Vector};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaChaRng;

use crate::{
    blocks::{
        BlockData, BlockHandle, BlockId, BlockInfo, BroadcastSetData, BroadcastSetHandleMut,
        BroadcastSetId, BroadcastSetInfo,
    },
    transaction::{Outpoint, TxData, TxHandle, TxId, TxInfo},
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
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash, Default)]
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
        let mut sim = Simulation {
            wallet_data: Vec::new(),
            payment_data: Vec::new(),
            address_data: Vec::new(),
            tx_data: Vec::new(),
            broadcast_set_data: Vec::new(),
            block_data: Vec::new(),
            current_epoch: Epoch(0),
            max_epochs: Epoch(0),
            prng: ChaChaRng::from_rng(rand::thread_rng()).unwrap(),
            spends: OrdMap::new(),
            wallet_info: Vec::new(),
            block_info: Vec::new(),
            tx_info: Vec::new(),
            broadcast_set_info: Vec::new(),
        };
        sim.max_epochs = Epoch(self.max_epochs);
        sim.prng = self.prng;

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

        for _ in 0..self.num_wallets {
            sim.new_wallet();
        }

        // Create one more wallet to simulate the "miners"
        sim.new_wallet();

        sim
    }
}

/// all entities are numbered sequentially
#[derive(Debug, PartialEq, Eq)]
struct Simulation {
    // primary information
    wallet_data: Vec<WalletData>,
    payment_data: Vec<PaymentObligationData>,
    address_data: Vec<AddressData>,
    tx_data: Vec<TxData>, // all are implicitly broadcast for now
    broadcast_set_data: Vec<BroadcastSetData>,
    // TODO mempools, = orderings / replacements of broadcast_sets
    block_data: Vec<BlockData>,
    current_epoch: Epoch,
    max_epochs: Epoch,
    prng: ChaChaRng,

    // secondary information (indexes)
    spends: OrdMap<Outpoint, OrdSet<TxId>>,
    wallet_info: Vec<WalletInfo>,
    block_info: Vec<BlockInfo>,
    tx_info: Vec<TxInfo>,
    broadcast_set_info: Vec<BroadcastSetInfo>,
}

impl<'a> Simulation {
    fn build_universe(&mut self) {
        let wallets = self.wallet_data.clone();
        let addresses = wallets
            .iter()
            .map(|w| w.id.with_mut(self).new_address())
            .collect::<Vec<_>>();

        // For now we just mine a coinbase transaction for each wallet
        let mut i = 0;
        for address in addresses.iter() {
            let coinbases_to_receive = self.prng.gen_range(1..10);
            for _ in 0..coinbases_to_receive {
                let _ = BroadcastSetHandleMut {
                    id: BroadcastSetId(i),
                    sim: self,
                }
                .construct_block_template(Weight::MAX_BLOCK)
                .mine(*address, self);

                // Do we need to track this here? or will it be tracked in the broadcast set update_wallets?
                // wallet.own_transactions.push(coinbase_tx.coinbase_tx().id);

                self.assert_invariants();
                i += 1;
            }
        }

        self.new_payment_obligation();
        self.assert_invariants();
    }

    fn tick(&mut self) {
        if self.payment_data.is_empty() {
            println!("No payment obligations, skipping epoch");
            self.current_epoch = Epoch(self.current_epoch.0 + 1);
            // Lets create a random payment obligation
            self.new_payment_obligation();
            self.assert_invariants();
            return;
        }

        let payment_obligation = self.payment_data.first().expect("checked above").clone();
        let mut from_wallet = payment_obligation.from.with_mut(self);
        let spend = from_wallet
            .fufill_payment_obligation()
            .expect("Payment obligation should have atleast one value");

        let bset = from_wallet.broadcast(std::iter::once(spend));
        bset.construct_block_template(Weight::MAX_BLOCK)
            .mine(self.miner_address(), self);
        self.current_epoch = Epoch(self.current_epoch.0 + 1);

        self.assert_invariants();
    }

    fn run(&mut self) {
        let max_epochs = self.max_epochs;
        while self.current_epoch < max_epochs {
            println!("Epoch {}", self.current_epoch.0);
            self.tick();
            // TODO: call this only in debug / testmode?
            self.assert_invariants();
        }
    }

    fn genesis_block(&self) -> BlockId {
        BlockId(0)
    }

    fn miner_address(&self) -> AddressId {
        // The "miners" wallet is the last wallet in the simulation
        AddressId(self.wallet_data.len() - 1)
    }

    // TODO remove
    fn get_tx(&'a self, id: TxId) -> TxHandle<'a> {
        id.with(&self)
    }

    /// Creates a random payment obligation between two wallets.
    fn new_payment_obligation(&mut self) {
        if self.max_epochs.0 - self.current_epoch.0 < 2 {
            // Not enough epochs left to create a payment obligation
            return;
        }
        // The last wallet is the "miner"
        let max_wallets = self.wallet_data.len() - 1;
        let from = self.prng.gen_range(0..max_wallets);
        let mut to = self.prng.gen_range(0..max_wallets);
        if to == from {
            to = (to + 1) % max_wallets;
        }
        let amount = self.prng.gen_range(1..5);
        let deadline = self.prng.gen_range(self.current_epoch.0..self.max_epochs.0);
        let to_addr = WalletId(to).with_mut(self).new_address();
        // First insert payment obligation into simulation
        self.payment_data.push(PaymentObligationData {
            amount: Amount::from_int_btc(amount),
            from: WalletId(from),
            to: to_addr,
            deadline: Epoch(deadline),
        });
        let payment_obligation_id = PaymentObligationId(self.payment_data.len() - 1);

        // Then insert into to_wallet's expected payments
        let last_wallet_info_id = self.wallet_data[to].last_wallet_info_id;
        self.wallet_info[last_wallet_info_id.0]
            .expected_payments
            .insert(payment_obligation_id);

        // Then insert into from_wallet's payment obligations
        let last_wallet_info_id = self.wallet_data[from].last_wallet_info_id;
        self.wallet_info[last_wallet_info_id.0]
            .payment_obligations
            .insert(payment_obligation_id);
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

        // TODO: assert that expected payments and payment obligations met
        // TODO: assert that for each payment obligation, the from wallet has the expected payment

        // TODO for all wallets, ensure their confirmed and unconfirmed utxos form a partition (their intersections are empty and their union is describes the corresponding block info and broadcast state)
        // take union and compare size to sum of sizes, and check equality with global structures
    }
}

impl std::fmt::Display for Simulation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Simulation State ===")?;
        writeln!(f, "Current Epoch: {}", self.current_epoch.0)?;
        writeln!(f, "Max Epochs: {}", self.max_epochs.0)?;
        writeln!(f, "\nWallets: {}", self.wallet_data.len())?;

        for (i, wallet) in self.wallet_data.iter().enumerate() {
            writeln!(f, "\nWallet {}:", i)?;
            writeln!(f, "  Own Transactions: {:?}", wallet.own_transactions)?;
            writeln!(f, "  Addresses: {:?}", wallet.addresses)?;
            writeln!(
                f,
                "  Broadcast Transactions: {:?}",
                self.wallet_info[wallet.last_wallet_info_id.0].broadcast_transactions
            )?;
            writeln!(
                f,
                "  Received Transactions: {:?}",
                self.wallet_info[wallet.last_wallet_info_id.0].received_transactions
            )?;
            writeln!(
                f,
                "  Unconfirmed Transactions: {:?}",
                self.wallet_info[wallet.last_wallet_info_id.0].unconfirmed_transactions
            )?;
            writeln!(
                f,
                "  Confirmed UTXOs: {:?}",
                self.wallet_info[wallet.last_wallet_info_id.0].confirmed_utxos
            )?;
            writeln!(
                f,
                "  Unconfirmed UTXOs: {:?}",
                self.wallet_info[wallet.last_wallet_info_id.0].unconfirmed_txos
            )?;
            writeln!(
                f,
                "  Unconfirmed Spends: {:?}",
                self.wallet_info[wallet.last_wallet_info_id.0].unconfirmed_spends
            )?;
        }

        writeln!(
            f,
            "\nUnfulfilled Payment Obligations: {}",
            self.payment_data.len()
        )?;
        for (i, payment) in self.payment_data.iter().enumerate() {
            writeln!(f, "\nPayment {}:", i)?;
            writeln!(f, "  Amount: {}", payment.amount)?;
            writeln!(f, "  From: Wallet {}", payment.from.0)?;
            writeln!(f, "  To: Wallet {}", payment.to.0)?;
            writeln!(f, "  Deadline: Epoch {}", payment.deadline.0)?;
        }

        writeln!(f, "\nBlocks: {}", self.block_data.len())?;
        writeln!(f, "Broadcast Sets: {}", self.broadcast_set_data.len())?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use bdk_coin_select::{Target, TargetFee, TargetOutputs};
    use im::{ordset, vector};

    use crate::transaction::{Input, Output};

    use super::*;

    #[test]
    fn test_universe() {
        let mut sim = SimulationBuilder::new(42, 2, 10, 1).build();
        sim.assert_invariants();
        sim.build_universe();
        sim.run();
        sim.assert_invariants();

        println!("{}", sim);

        // TODO: re-enable these assertions
        // let spend = alice.with_mut(&mut sim).data().own_transactions[1];

        // assert_eq!(
        //     alice.with(&sim).info().unconfirmed_spends,
        //     OrdSet::from_iter(coinbase_tx.with(&sim).outpoints())
        // );

        // assert_eq!(
        //     alice.with(&sim).info().unconfirmed_txos,
        //     OrdSet::from_iter(spend.with(&sim).outpoints().skip(1))
        // );

        // assert_eq!(
        //     bob.with(&sim).info().unconfirmed_txos,
        //     OrdSet::from_iter(spend.with(&sim).outpoints().take(1))
        // );

        // assert_eq!(
        //     alice.with(&sim).info().broadcast_transactions,
        //     vector![spend]
        // );

        // assert!(bob.with(&sim).info().received_transactions.contains(&spend));
    }

    #[test]
    fn it_works() {
        let mut sim = SimulationBuilder::new(42, 2, 10, 1).build();

        sim.assert_invariants();

        let alice = sim.new_wallet();
        sim.assert_invariants();
        let bob = sim.new_wallet();
        sim.assert_invariants();

        let alice_coinbase_addr = alice.with_mut(&mut sim).new_address();
        sim.assert_invariants();

        // TODO sim.current_broadcast_set()
        let initial_bx = BroadcastSetHandleMut {
            id: BroadcastSetId(0),
            sim: &mut sim,
        };

        let coinbase_tx = initial_bx
            .construct_block_template(Weight::MAX_BLOCK)
            .mine(alice_coinbase_addr, &mut sim)
            .coinbase_tx()
            .id;

        sim.assert_invariants();

        assert_eq!(alice.with(&sim).data().own_transactions, vec![coinbase_tx]);
        assert_eq!(
            alice.with(&sim).info().confirmed_utxos,
            OrdSet::from_iter(coinbase_tx.with(&sim).outpoints())
        );

        // TODO coinbase maturity

        let payment = PaymentObligationData {
            amount: Amount::from_int_btc(20),
            from: WalletId(0),
            to: bob.with_mut(&mut sim).new_address(),
            deadline: Epoch(2), // TODO 102
        };
        sim.assert_invariants();

        let bob_payment_addr = bob.with_mut(&mut sim).new_address();
        sim.assert_invariants();
        let alice_change_addr = alice.with_mut(&mut sim).new_address();
        sim.assert_invariants();

        let target = Target {
            fee: TargetFee {
                rate: bdk_coin_select::FeeRate::from_sat_per_vb(1.0),
                replace: None,
            },
            outputs: TargetOutputs {
                value_sum: payment.amount.to_sat(),
                weight_sum: 34, // TODO use payment.to to derive an address, payment.into() ?
                n_outputs: 1,
            },
        };

        let long_term_feerate = bitcoin::FeeRate::from_sat_per_vb(10).unwrap();

        let spend = alice
            .with_mut(&mut sim)
            .new_tx(|tx, sim| {
                // TODO use select_coins
                let (inputs, drain) = alice.with(&sim).select_coins(target, long_term_feerate);

                tx.inputs = inputs
                    .map(|o| Input {
                        outpoint: o.outpoint,
                    })
                    .collect();

                tx.outputs = vec![
                    Output {
                        amount: payment.amount,
                        address_id: bob_payment_addr,
                    },
                    Output {
                        amount: Amount::from_sat(drain.value),
                        address_id: alice_change_addr,
                    },
                ];
            })
            .id;
        sim.assert_invariants();

        assert_eq!(spend, TxId(2));

        assert_eq!(spend.with(&sim).info().weight, Weight::from_wu(688));

        assert_eq!(
            alice.with(&sim).data().own_transactions,
            vec![coinbase_tx, spend]
        );

        assert_eq!(
            alice.with(&sim).info().broadcast_transactions,
            Vector::default()
        );

        // these fields are not updated until broadcast
        assert_eq!(
            alice.with(&sim).info().confirmed_utxos,
            OrdSet::from_iter(coinbase_tx.with(&sim).outpoints())
        );
        assert_eq!(alice.with(&sim).info().unconfirmed_spends, ordset![]);

        alice.with_mut(&mut sim).broadcast(std::iter::once(spend));

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
