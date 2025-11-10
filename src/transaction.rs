use crate::wallet::{AddressHandle, AddressId, WalletHandle, WalletHandleMut};
use crate::Simulation;
use bitcoin::consensus::Decodable;
use bitcoin::transaction::{predict_weight, InputWeightPrediction};
use bitcoin::{Amount, Weight};
use bitcoin::{FeeRate, ScriptBuf, WitnessProgram};

define_entity!(
    Tx,
    {
        // version, locktime, witness flag
        pub(crate) inputs: Vec<Input>,
        pub(crate) outputs: Vec<Output>,
    },
    {
        pub(crate) fee: Amount,
        pub(crate) weight: Weight,
    }
);

impl From<TxData> for lattice_psbt::UnOrderedTransaction {
    fn from(tx: TxData) -> Self {
        let mut psbt = lattice_psbt::UnOrderedTransaction::default();
        for input in tx.inputs.iter() {
            psbt.add_input(lattice_psbt::Vin::from(*input));
        }
        for output in tx.outputs.iter() {
            psbt.add_output(lattice_psbt::Vout::from(*output));
        }
        psbt
    }
}

impl From<TxId> for bitcoin::Txid {
    fn from(txid: TxId) -> Self {
        let mut buf = [0u8; 32];
        let txid_bytes = txid.0.to_le_bytes();
        buf[..txid_bytes.len()].copy_from_slice(&txid_bytes);
        bitcoin::Txid::consensus_decode(&mut &buf[..]).expect("32 bytes should never fail")
    }
}

// TODO rename to OutputId?
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub(crate) struct Outpoint {
    pub(crate) txid: TxId,
    pub(crate) index: usize,
}

impl<'a> Outpoint {
    fn with(&self, sim: &'a Simulation) -> OutputHandle<'a> {
        OutputHandle {
            sim,
            outpoint: *self,
        }
    }
}

impl From<Outpoint> for bitcoin::OutPoint {
    fn from(outpoint: Outpoint) -> Self {
        bitcoin::OutPoint::new(outpoint.txid.into(), outpoint.index as u32)
    }
}

// TODO rename to InputData?
#[derive(Debug, PartialEq, Eq, Clone, Copy, PartialOrd, Ord)]
pub(crate) struct Input {
    pub(crate) outpoint: Outpoint, // sequence,
                                   // witness?
}

impl From<Input> for lattice_psbt::Vin {
    fn from(input: Input) -> Self {
        lattice_psbt::Vin::default().with_outpoint(input.outpoint.into())
    }
}

#[derive(Debug, PartialEq, Clone, Copy, Eq, PartialOrd, Ord)]
pub(crate) struct InputId {
    pub(crate) txid: TxId,
    pub(crate) index: usize,
}

pub(crate) struct InputHandle<'a> {
    sim: &'a Simulation,
    pub(crate) id: InputId,
}

impl<'a> InputHandle<'a> {
    pub(crate) fn data(&self) -> &'a Input {
        &self.sim.get_tx(self.id.txid).data().inputs[self.id.index]
    }

    pub(crate) fn prevout(&self) -> OutputHandle<'a> {
        self.data().outpoint.with(self.sim)
    }
}

impl<'a> From<InputHandle<'a>> for Output {
    fn from(handle: InputHandle<'a>) -> Output {
        *handle.prevout().data()
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(crate) struct Output {
    pub(crate) amount: Amount,
    pub(crate) address_id: AddressId,
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

impl From<Output> for lattice_psbt::Vout {
    fn from(output: Output) -> Self {
        lattice_psbt::Vout::from_output(&bitcoin::transaction::TxOut::from(output))
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
pub(crate) struct OutputHandle<'a> {
    sim: &'a Simulation,
    pub(crate) outpoint: Outpoint,
}

impl From<OutputHandle<'_>> for InputWeightPrediction {
    fn from(output: OutputHandle<'_>) -> Self {
        Self::from(output.address())
    }
}

impl<'a> OutputHandle<'a> {
    pub(crate) fn new(sim: &'a Simulation, outpoint: Outpoint) -> Self {
        Self { sim, outpoint }
    }

    pub(crate) fn outpoint(&self) -> Outpoint {
        self.outpoint
    }

    pub(crate) fn data(&self) -> &'a Output {
        &self.sim.get_tx(self.outpoint.txid).data().outputs[self.outpoint.index]
    }

    pub(crate) fn address(&'a self) -> AddressHandle<'a> {
        self.data().address_id.with(self.sim)
    }

    pub(crate) fn wallet(&'a self) -> WalletHandle<'a> {
        self.data().wallet(self.sim)
    }

    pub(crate) fn wallet_mut<'b>(&self, sim: &'b mut Simulation) -> WalletHandleMut<'b> {
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
    pub(crate) fn data(&self) -> &'a TxData {
        &self.sim.tx_data[self.id.0]
    }

    pub(crate) fn info(&self) -> &'a TxInfo {
        &self.sim.tx_info[self.id.0]
    }

    pub(crate) fn is_coinbase(&self) -> bool {
        self.data().inputs.is_empty()
    }

    pub(crate) fn outpoints(&self) -> impl Iterator<Item = Outpoint> {
        let txid = self.id;
        (0..self.data().outputs.len()).map(move |index| Outpoint { txid, index })
    }
    pub(crate) fn outputs(&'a self) -> impl Iterator<Item = OutputHandle<'a>> {
        self.outpoints().map(|outpoint| OutputHandle {
            sim: self.sim,
            outpoint,
        })
    }

    pub(crate) fn inputs(&'a self) -> impl Iterator<Item = InputHandle<'a>> {
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
    pub(crate) fn new(tx: &TxData, sim: &Simulation) -> Self {
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

    pub(crate) fn feerate(self) -> FeeRate {
        self.fee / self.weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_txid_encoding() {
        let txid = TxId(1);
        let txid_from_bytes = bitcoin::Txid::from(txid);
    }
}
