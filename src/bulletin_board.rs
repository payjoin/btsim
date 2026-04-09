use crate::{
    cospend::UtxoWithMetadata,
    transaction::{Outpoint, Output},
};

define_entity!(
    BulletinBoard,
    {
        pub(crate) id: BulletinBoardId,
        pub(crate) messages: Vec<BroadcastMessageType>,
    },
    {
    }
);

impl<'a> BulletinBoardHandle<'a> {
    pub(crate) fn data(&self) -> &'a BulletinBoardData {
        &self.sim.bulletin_boards[self.id.0]
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub(crate) enum BroadcastMessageType {
    AcceptCoSpend(Vec<UtxoWithMetadata>),
    ContributeInputs(Outpoint),
    ContributeOutputs(Output),
    ReadyToSign(),
}
