use bdk_coin_select::{
    metrics::LowestFee, Candidate, ChangePolicy, CoinSelector, Drain, DrainWeights, Target,
    TR_DUST_RELAY_MIN_VALUE, TR_KEYSPEND_TXIN_WEIGHT,
};
use bitcoin::{
    transaction::{predict_weight, InputWeightPrediction},
    Amount, FeeRate, ScriptBuf, Weight, WitnessProgram,
};
use im::{OrdMap, OrdSet, Vector};
use paste::paste;
use rand::SeedableRng;
use rand_chacha::ChaChaRng;

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
macro_rules! define_entity_id_and_handle {
    ( $base:ident ) => {
        paste! {
            /// Type safe, copyable references into the simulation structure,
            /// used for internal references and as non-borrow pointer
            /// analogues.
            #[derive(Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Clone, Copy)] // TODO Remove Ord?
            struct [<$base Id>](usize); // TODO Outpoint is OutputId, InputId is also a tuple

            /// Ephemeral view into primary and second information of $base.
            #[derive(Debug, PartialEq, Eq, Clone, Copy)]
            struct [<$base Handle>]<'a> {
                sim: &'a Simulation,
                id: [<$base Id>],
            }

            impl<'a> [<$base Id>] {
                /// Borrow the simulation, reifying the id to a [<$base Handle>]
                /// for read access to entity.
                fn with(&self, sim: &'a Simulation) -> [<$base Handle>]<'a>
                {
                    [<$base Handle>]::new(sim, *self)
                }
            }

            impl<'a> [<$base Handle>]<'a> {
                fn new(sim: &'a Simulation, id: [<$base Id>]) -> Self {
                    Self { sim, id }
                }
            }

            impl<'a> From<[<$base Handle>]<'a>> for [<$base Id>] {
                fn from(handle: [<$base Handle>]) -> [<$base Id>] {
                    handle.id
                }
            }
        }
    };
}

macro_rules! define_entity_data {
    (
        $base:ident,
        $data_fields:tt
    ) => {
        paste! {
            /// Primary information associated with $base.
            #[derive(Debug, PartialEq, Eq, Clone)]
            struct [<$base Data>] $data_fields
        }
    };
}

macro_rules! define_entity_info {
    (
        $base:ident,
        $info_fields:tt
    ) => {
        paste! {
            /// Secondary (derived) information associated with $base.
            #[derive(Debug, PartialEq, Eq, Clone)]
            struct [<$base Info>] $info_fields
        }
    };
}

macro_rules! define_entity_info_id {
    (
        $base:ident
    ) => {
        paste! {
            #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
            struct [<$base InfoId>](usize);
        }
    };
}

macro_rules! define_entity_handle_mut {
    (
        $base:ident
    ) => {
        paste! {
            #[derive(Debug)]
            struct [<$base HandleMut>]<'a> {
                sim: &'a mut Simulation,
                id: [<$base Id>],
            }

            impl<'a> [<$base Id>] {
                fn with_mut(&self, sim: &'a mut Simulation) -> [<$base HandleMut>]<'a> {
                    [<$base HandleMut>]::new(sim, *self)
                }
            }

            impl<'a> [<$base HandleMut>]<'a> {
                fn new(sim: &'a mut Simulation, id: [<$base Id>]) -> Self {
                    Self { sim, id }
                }
            }
                    // Implement Deref to allow [<$base HandleMut>] to be used as [<$base Handle>]
            impl<'a> std::ops::Deref for [<$base HandleMut>]<'a> {
                type Target = [<$base Handle>]<'a>;

                fn deref(&self) -> &Self::Target {
                    // safety: [<$base Handle>] does not allow mutating sim
                    unsafe {
                        &*(self as *const [<$base HandleMut>]<'a> as *const [<$base Handle>]<'a>)
                    }
                }
            }

            impl<'a> From<[<$base HandleMut>]<'a>> for [<$base Id>] {
                fn from(handle: [<$base HandleMut>]) -> [<$base Id>] {
                    handle.id
                }
            }
        }
    };
}

macro_rules! define_entity {
    (
        $base:ident,
        $data_fields:tt,
        $info_fields:tt
    ) => {
        define_entity_id_and_handle!($base);
        define_entity_data!($base, $data_fields);
        define_entity_info!($base, $info_fields);
    };
}

// only wallet.. eliminate?
macro_rules! define_entity_mut_updatable {
    (
        $base:ident,
        $data_fields:tt,
        $info_fields:tt
    ) => {
        paste! {
            define_entity_id_and_handle!($base);
            define_entity_handle_mut!($base);
            define_entity_info_id!($base);
            define_entity_data!($base, $data_fields);
            define_entity_info!($base, $info_fields);
        }
    };
}

define_entity!(
    PaymentObligation,
    {
        deadline: usize, // block height? time step?
        amount: Amount,
        from: WalletId,
        to: WalletId,
    },
    {
        // TODO coin selection strategy agnostic (pessimal?) spendable balance lower
        // bound
    }
);

define_entity!(
    Block,
    {
        // block data
        parent: Option<BlockId>,
        coinbase_tx: TxId,
        confirmed_txs: Vec<TxId>,
    },
    {
        // TODO total size
        // TODO total fees
        height: usize,
        spent: OrdSet<Outpoint>,
        created: OrdSet<Outpoint>,
        utxos: OrdSet<Outpoint>,
        confirmed_txs: OrdSet<TxId>,
        all_confirmed_txs: OrdSet<TxId>,
    }
);

define_entity!(Address, {
    wallet_id: WalletId,
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
    fn data(&'a self) -> &'a AddressData {
        &self.sim.address_data[self.id.0]
    }

    fn wallet(&self) -> WalletHandle<'a> {
        self.data().wallet_id.with(self.sim)
    }
}

define_entity!(
    Tx,
    {
        // version, locktime, witness flag
        inputs: Vec<Input>,
        outputs: Vec<Output>,
    },
    {
        fee: Amount,
        weight: Weight,
    }
);

// TODO rename to OutputId?
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
struct Outpoint {
    txid: TxId,
    index: usize,
}

impl<'a> Outpoint {
    fn with(&self, sim: &'a Simulation) -> OutputHandle<'a> {
        OutputHandle {
            sim,
            outpoint: *self,
        }
    }
}

// TODO rename to InputData?
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
struct Input {
    outpoint: Outpoint, // sequence,
                        // witness?
}

struct InputId {
    txid: TxId,
    index: usize,
}

struct InputHandle<'a> {
    sim: &'a Simulation,
    id: InputId,
}

impl<'a> InputHandle<'a> {
    fn data(&self) -> &'a Input {
        &self.sim.get_tx(self.id.txid).data().inputs[self.id.index]
    }

    fn prevout(&self) -> OutputHandle<'a> {
        self.data().outpoint.with(self.sim)
    }
}

impl<'a> From<InputHandle<'a>> for Output {
    fn from(handle: InputHandle<'a>) -> Output {
        *handle.prevout().data()
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
struct Output {
    amount: Amount,
    address_id: AddressId,
}

impl From<Output> for bitcoin::transaction::TxOut {
    fn from(o: Output) -> Self {
        // FIXME refactor into fn encode_as_txo(enum { AddressId, Index, Outpoint  })
        // TODO handle multiple address types
        let mut program = [0u8; 32];
        // TODO tag, segregate from txos encoding indexes?
        program[0] = o
            .address_id
            .0
            .try_into()
            .expect("TODO support more than 256 addresses");

        let witness_program =
            WitnessProgram::new(bitcoin::WitnessVersion::V1, &program[..]).unwrap();
        let script_pubkey = ScriptBuf::new_witness_program(&witness_program);

        bitcoin::transaction::TxOut {
            value: o.amount,
            script_pubkey,
        }
    }
}

impl Output {
    fn size(&self) -> usize {
        bitcoin::transaction::TxOut::from(*self).size()
    }

    fn address<'a>(&self, sim: &'a Simulation) -> AddressHandle<'a> {
        self.address_id.with(sim)
    }

    fn wallet<'a>(&self, sim: &'a Simulation) -> WalletHandle<'a> {
        self.address(sim).wallet()
    }

    fn wallet_mut<'a>(&self, sim: &'a mut Simulation) -> WalletHandleMut<'a> {
        let owner_id = self.address(sim).data().wallet_id;
        owner_id.with_mut(sim)
    }
}

#[derive(Clone, Copy)]
struct OutputHandle<'a> {
    sim: &'a Simulation,
    outpoint: Outpoint,
}

impl From<OutputHandle<'_>> for InputWeightPrediction {
    fn from(output: OutputHandle<'_>) -> Self {
        Self::from(output.address())
    }
}

impl<'a> OutputHandle<'a> {
    fn data(&self) -> &'a Output {
        &self.sim.get_tx(self.outpoint.txid).data().outputs[self.outpoint.index]
    }

    fn address(&'a self) -> AddressHandle<'a> {
        self.data().address_id.with(self.sim)
    }

    fn wallet(&'a self) -> WalletHandle<'a> {
        self.data().wallet(self.sim)
    }

    fn wallet_mut<'b>(&self, sim: &'b mut Simulation) -> WalletHandleMut<'b> {
        self.data().wallet_mut(sim)
    }
}

impl<'a> From<OutputHandle<'a>> for Output {
    fn from(handle: OutputHandle<'a>) -> Output {
        *handle.data()
    }
}

impl<'a> From<OutputHandle<'a>> for Outpoint {
    fn from(handle: OutputHandle<'a>) -> Outpoint {
        handle.outpoint
    }
}

impl<'a> TxHandle<'a> {
    fn data(&self) -> &'a TxData {
        &self.sim.tx_data[self.id.0]
    }

    fn info(&self) -> &'a TxInfo {
        &self.sim.tx_info[self.id.0]
    }

    fn is_coinbase(&self) -> bool {
        self.data().inputs.is_empty()
    }

    fn outpoints(&self) -> impl Iterator<Item = Outpoint> {
        let txid = self.id;
        (0..self.data().outputs.len()).map(move |index| Outpoint { txid, index })
    }
    fn outputs(&'a self) -> impl Iterator<Item = OutputHandle<'a>> {
        self.outpoints().map(|outpoint| OutputHandle {
            sim: self.sim,
            outpoint,
        })
    }

    fn inputs(&'a self) -> impl Iterator<Item = InputHandle<'a>> {
        let txid = self.id;
        let sim = self.sim;
        (0..self.data().inputs.len()).map(move |index| InputHandle {
            sim,
            id: InputId { txid, index },
        })
    }

    // TODO fn prevouts(self) -> impl IntoIterator??
    // TODO confirmed
    // TODO previous txs
}

impl Default for TxData {
    fn default() -> Self {
        Self {
            inputs: Vec::default(),
            outputs: Vec::default(),
        }
    }
}

impl TxInfo {
    fn new(tx: &TxData, sim: &Simulation) -> Self {
        // TODO Result with invalid txn error?
        // TODO refactor into a method.. on Simulation? on tx accepting simulation?
        let prevouts = tx.inputs.iter().map(|i| i.outpoint.with(&sim));

        let weight = predict_weight(
            prevouts.clone().map(|i| InputWeightPrediction::from(i)),
            tx.outputs.iter().map(|o| o.size()),
        );

        // TODO separate to a different index struct
        let total_input_amount: Amount = prevouts.map(|o| o.data().amount).sum();
        let total_output_amount = tx.outputs.iter().map(|o| o.amount).sum();

        // TODO Result
        assert!(tx.inputs.is_empty() || total_output_amount <= total_input_amount);

        let fees = if tx.inputs.is_empty() {
            Amount::default()
        } else {
            total_input_amount - total_output_amount // TODO
        };

        TxInfo { fee: fees, weight }
    }

    fn feerate(self) -> FeeRate {
        self.fee / self.weight
    }
}

#[derive(Debug)]
struct ChainParams {
    initial_subsidy: Amount,
    halving_interval: usize,
    max_block_weight: Weight,
}

impl ChainParams {
    fn subsidy(self, height: usize) -> Amount {
        Amount::from_sat(self.initial_subsidy.to_sat() >> (height / self.halving_interval))
        // TODO BIP 42
    }
}

impl Default for ChainParams {
    fn default() -> Self {
        ChainParams {
            initial_subsidy: Amount::from_int_btc(50),
            halving_interval: 210_000,
            max_block_weight: Weight::from_wu(400000),
        }
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct TxByFeerate(FeeRate, TxId);

// TODO refactor?
define_entity_id_and_handle!(BroadcastSet);
define_entity_handle_mut!(BroadcastSet);
define_entity_info!(BroadcastSet, {
    parent_id: Option<BroadcastSetId>,
    chain_tip_id: BlockId,
    // new_txs: Vec<TxId>,
    unconfirmed_txs: OrdSet<TxId>,
    invalidated_txs: OrdSet<TxId>,
});

#[derive(Debug, PartialEq, Eq, Clone)]
enum BroadcastSetData {
    Block(BlockId),
    Transactions(Vec<TxId>),
}

impl<'a> BroadcastSetHandle<'a> {
    fn data(&'a self) -> &'a BroadcastSetData {
        &self.sim.broadcast_set_data[self.id.0]
    }

    fn info(&'a self) -> &'a BroadcastSetInfo {
        &self.sim.broadcast_set_info[self.id.0]
    }
}

impl<'a> BroadcastSetHandleMut<'a> {
    fn process_block(self, block_id: BlockId) -> Self {
        let block = block_id.with(self.sim);

        let mut unconfirmed_txs = self.info().unconfirmed_txs.clone();
        let mut invalidated_txs = self.info().invalidated_txs.clone();

        for tx in block.data().txs() {
            unconfirmed_txs.remove(&tx);

            // also remove conflicting transactions
            // TODO refactor tx.with(self.sim).spent_coins() impl Iterator Outppoint?
            for input in &tx.with(self.sim).data().inputs {
                for conflicting_tx in self.sim.spends[&input.outpoint]
                    .without(tx)
                    .intersection(unconfirmed_txs.clone())
                // FIXME pr to im, unnecessary clone
                {
                    unconfirmed_txs.remove(&conflicting_tx);
                    invalidated_txs.insert(conflicting_tx);
                }
            }
        }

        self.sim
            .broadcast_set_data
            .push(BroadcastSetData::Block(block_id));

        self.sim.broadcast_set_info.push(BroadcastSetInfo {
            parent_id: Some(self.id),
            chain_tip_id: block_id,
            unconfirmed_txs,
            invalidated_txs,
        });

        let id = BroadcastSetId(self.sim.broadcast_set_data.len() - 1);

        Self { sim: self.sim, id }.update_wallets()
    }

    // TODO non Mut BroadcastSetHandle fn parent(&'a self) -> Self

    fn chain_tip(&'a self) -> BlockHandle<'a> {
        self.info().chain_tip_id.with(self.sim)
    }

    fn update_wallets(self) -> Self {
        let mut wallet_infos = std::collections::HashMap::<WalletId, WalletInfo>::default();

        let id = self.id;

        let mut update_wallet_info = |wallet: &WalletHandle, update: &dyn Fn(&mut WalletInfo)| {
            // impl not allowed in closure but rustc suggests adding it?
            if !wallet_infos.contains_key(&wallet.id) {
                let mut new_info = wallet.info().clone();
                new_info.broadcast_set_id = id;
                wallet_infos.insert(wallet.id, new_info);
            }

            update(wallet_infos.get_mut(&wallet.id).unwrap())
        };

        match self.data() {
            BroadcastSetData::Transactions(new_txs) => {
                for tx in new_txs {
                    for input in tx.with(self.sim).inputs() {
                        let prevout = input.prevout();
                        let wallet = prevout.wallet();

                        update_wallet_info(&wallet, &|i: &mut WalletInfo| {
                            i.unconfirmed_spends.insert(input.data().outpoint);
                        });
                    }

                    for output in tx.with(self.sim).outputs() {
                        let wallet = output.wallet();

                        update_wallet_info(&wallet, &|info: &mut WalletInfo| {
                            // don't treat wallet generated transactions as received transactions
                            // FIXME O(n) contains()
                            if !wallet.data().own_transactions.contains(tx) {
                                info.received_transactions.push_back(*tx);
                            }

                            info.unconfirmed_transactions.insert(*tx);
                            info.unconfirmed_txos.insert(output.outpoint);
                        });
                    }
                }
            }
            BroadcastSetData::Block(_) => {
                for tx in &self.chain_tip().info().confirmed_txs {
                    for input in tx.with(self.sim).inputs() {
                        let prevout = input.prevout();
                        let wallet = prevout.wallet();

                        update_wallet_info(&wallet, &|info: &mut WalletInfo| {
                            info.broadcast_set_id = self.id;
                            info.confirmed_utxos.remove(&input.data().outpoint);
                            info.unconfirmed_txos.remove(&input.data().outpoint);
                            info.unconfirmed_spends.remove(&input.data().outpoint);
                            info.unconfirmed_transactions.remove(tx);
                        })
                    }

                    for output in tx.with(self.sim).outputs() {
                        let wallet = output.wallet();

                        update_wallet_info(&wallet, &|info: &mut WalletInfo| {
                            // TODO no .contains() check needed if checking self.all_txs?
                            // FIXME O(n) + O(n) contains()
                            if !wallet.data().own_transactions.contains(tx) {
                                if !info.received_transactions.contains(tx) {
                                    info.received_transactions.push_back(*tx);
                                }
                            }
                            info.confirmed_utxos.insert(output.outpoint);
                            info.unconfirmed_txos.remove(&output.outpoint);
                            info.unconfirmed_transactions.remove(tx);
                        })
                    }
                }
            }
        };

        for (wallet_id, info) in wallet_infos {
            // TODO assert consistency of wallet info with global index? confirmed & unconfirmed intersections are empty

            let id = WalletInfoId(self.sim.wallet_info.len());
            self.sim.wallet_info.push(info); // TODO append?
            wallet_id.with_mut(self.sim).data_mut().last_wallet_info_id = id;
        }

        self
    }

    // TODO Deref, move to non mut
    fn unconfirmed_txs(&'a self) -> impl IntoIterator<Item = TxHandle<'a>> {
        self.info()
            .unconfirmed_txs
            .iter()
            .map(|tx| self.sim.get_tx(*tx))
    }

    fn broadcast(self, txs: impl IntoIterator<Item = TxId>) -> Self {
        let previously_unconfirmed_txs = &self.info().unconfirmed_txs;

        let all_confirmed_txs = &self.chain_tip().info().all_confirmed_txs;

        let new_txs: Vec<TxId> = txs
            .into_iter()
            .filter(|tx| !previously_unconfirmed_txs.contains(tx))
            .filter(|tx| !all_confirmed_txs.contains(tx))
            .collect();

        let unconfirmed_txs = previously_unconfirmed_txs
            .clone()
            .union(OrdSet::from(&new_txs));

        let data = BroadcastSetData::Transactions(new_txs);

        let bxset = BroadcastSetInfo {
            parent_id: Some(self.id),
            chain_tip_id: self.chain_tip().id,
            unconfirmed_txs,
            invalidated_txs: self.info().invalidated_txs.clone(),
        };

        let id = BroadcastSetId(self.sim.broadcast_set_data.len());

        self.sim.broadcast_set_data.push(data);
        self.sim.broadcast_set_info.push(bxset);

        Self { sim: self.sim, id }.update_wallets()
    }

    // TODO move to its own MempoolState objects that implement tx ordering policy
    // for each tx, build unconfirmed transitive closure of parents
    // order these by feerate
    // prune double counted parents
    // solve for satisfiability WRT double spending
    // knaptime optzn
    fn construct_block_template(&self, max_weight: Weight) -> BlockTemplate {
        let last_block = self.chain_tip();

        let mut utxos = last_block.info().utxos.clone();
        let mut spent = OrdSet::<Outpoint>::default();
        let mut created = OrdSet::<Outpoint>::default();
        let mut confirmed_txs = Vec::<TxId>::default();

        let mut remaining_weight = max_weight;

        'tx: for tx in self.unconfirmed_txs() {
            // skip if too large
            if tx.info().weight >= remaining_weight {
                continue 'tx;
            }

            let mut tx_utxos = utxos.clone();
            let mut tx_spent = spent.clone();
            let mut tx_created = created.clone();

            for input in &tx.data().inputs {
                if tx_utxos.remove(&input.outpoint).is_none() {
                    // skip if spending a spent txo
                    continue 'tx;
                }

                tx_spent.insert(input.outpoint);
            }

            for outpoint in tx.outpoints() {
                tx_utxos.insert(outpoint);
                tx_created.insert(outpoint);
            }

            // Transaction is valid,
            confirmed_txs.push(tx.id);
            utxos = tx_utxos;
            spent = tx_spent;
            created = tx_created;
            remaining_weight -= tx.info().weight;
        }

        BlockTemplate {
            parent: last_block.id,
            txs: confirmed_txs,
            utxos,
            spent,
            created,
        }
    }
}

// Ephemeral data type, no entity ID and not retained just a helper for constructing blocks
struct BlockTemplate {
    parent: BlockId,
    txs: Vec<TxId>,
    spent: OrdSet<Outpoint>,
    created: OrdSet<Outpoint>,
    utxos: OrdSet<Outpoint>,
}

impl<'a> BlockTemplate {
    fn mine(self, rewards_to: AddressId, sim: &'a mut Simulation) -> BlockHandle<'a> {
        let parent_block = self.parent.with(sim);

        let height = 1 + parent_block.info().height;
        let subsidy = ChainParams::default().subsidy(height); // TODO make parameter of simulation

        let fees = self.txs.iter().map(|tx| sim.get_tx(*tx).info().fee).sum();

        let block_rewards = subsidy + fees;

        let mut confirmed_txs = OrdSet::from(&self.txs);

        let coinbase_tx = sim.new_tx(|tx, _| {
            tx.outputs.push(Output {
                address_id: rewards_to,
                amount: block_rewards,
            });
        });

        confirmed_txs.insert(coinbase_tx);

        let parent_block = self.parent.with(sim); // recreate since new_tx needs a mut borrow of sim
        let all_confirmed_txs = parent_block
            .info()
            .all_confirmed_txs
            .clone()
            .union(confirmed_txs.clone());

        // TODO refactor _mut()?
        let rewards_wallet = rewards_to.with(sim).wallet().id;
        sim.wallet_data[rewards_wallet.0]
            .own_transactions
            .push(coinbase_tx);

        let mut utxos = self.utxos;
        let mut created = self.created;

        let outpoint = Outpoint {
            txid: coinbase_tx,
            index: 0,
        };
        utxos.insert(outpoint);
        created.insert(outpoint);

        // TODO refactor, blockinfo shouldn't be created here
        sim.new_block(
            BlockData {
                parent: Some(self.parent),
                coinbase_tx,
                confirmed_txs: self.txs,
            },
            BlockInfo {
                height,
                utxos,
                created,
                spent: self.spent,
                confirmed_txs,
                all_confirmed_txs,
            },
        )
    }
}

impl BlockData {
    fn txs<'a>(&'a self) -> impl Iterator<Item = &TxId> {
        // TODO why is confirmed_txs by ref?
        std::iter::once(&self.coinbase_tx).chain(self.confirmed_txs.iter())
    }
}

impl<'a> BlockHandle<'a> {
    fn data(&self) -> &'a BlockData {
        &self.sim.block_data[self.id.0]
    }

    // TODO TxHandle
    fn txs(&'a self) -> impl Iterator<Item = &'a TxId> {
        self.data().txs()
    }

    fn info(&self) -> &'a BlockInfo {
        &self.sim.block_info[self.id.0]
    }

    fn parent(&self) -> Option<Self> {
        self.data().parent.map(|id| Self { sim: self.sim, id })
    }

    fn coinbase_tx(&self) -> TxHandle {
        self.sim.get_tx(self.data().coinbase_tx)
    }
}

define_entity_mut_updatable!(
    Wallet,
    {
        id: WalletId,
        addresses: Vec<AddressId>,         // TODO split into internal/external?
        own_transactions: Vec<TxId>,       // transactions originating from this wallet
        last_wallet_info_id: WalletInfoId, // Monotone
    },
    {
        broadcast_set_id: BroadcastSetId,
        payment_obligations: OrdSet<PaymentObligationId>,
        expected_payments: OrdSet<PaymentObligationId>,
        broadcast_transactions: Vector<TxId>,
        received_transactions: Vector<TxId>,
        confirmed_utxos: OrdSet<Outpoint>,    // TODO locktimes
        unconfirmed_transactions: OrdSet<TxId>,
        unconfirmed_txos: OrdSet<Outpoint>,  // compute CPFP cost
        unconfirmed_spends: OrdSet<Outpoint>, // RBFable
    }
);

// TODO WalletHandle
impl<'a> WalletHandle<'a> {
    fn data(&self) -> &'a WalletData {
        &self.sim.wallet_data[self.id.0]
    }

    fn info(&self) -> &'a WalletInfo {
        &self.sim.wallet_info[self.data().last_wallet_info_id.0]
    }

    // TODO give utxo list as argument so that different variants can be used
    // TODO return change information
    fn select_coins(
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
            .map(|outpoint| OutputHandle {
                outpoint: *outpoint,
                sim: self.sim,
            })
    }

    fn unspent_coins(&self) -> impl Iterator<Item = OutputHandle<'a>> + '_ {
        self.potentially_spendable_txos()
            .filter(|o| !self.info().unconfirmed_spends.contains(&o.outpoint))
    }

    fn double_spendable_coins(&self) -> impl Iterator<Item = OutputHandle<'a>> + '_ {
        self.potentially_spendable_txos()
            .filter(|o| self.info().unconfirmed_spends.contains(&o.outpoint))
    }
}

impl<'a> WalletHandleMut<'a> {
    fn data_mut<'b>(&'b mut self) -> &'b mut WalletData {
        &mut self.sim.wallet_data[self.id.0]
    }

    fn new_address(&mut self) -> AddressId {
        let id = self.sim.new_address(self.id);
        self.sim.wallet_data[self.id.0].addresses.push(id);
        id
    }

    fn new_tx<F>(&mut self, build: F) -> TxHandle
    where
        F: FnOnce(&mut TxData, &Simulation),
    {
        let id = self.sim.new_tx(build);
        self.data_mut().own_transactions.push(id);
        TxHandle::new(self.sim, id)
    }

    fn broadcast(&'a mut self, txs: impl IntoIterator<Item = TxId>) -> BroadcastSetHandleMut<'a> {
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
    use bdk_coin_select::{TargetFee, TargetOutputs};
    use im::{ordset, vector};

    use super::*;

    #[test]
    fn it_works() {
        let mut sim = SimulationBuilder::new_random(2, 10, 1).build();

        sim.assert_invariants();

        let alice = WalletId(0);
        let bob = WalletId(1);

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
            to: WalletId(1),
            deadline: 2, // TODO 102
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
