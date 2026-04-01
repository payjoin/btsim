use crate::{bulletin_board::BulletinBoardId, wallet::WalletId};

#[derive(Debug, PartialEq, Eq, Clone)]
pub(crate) enum MessageType {
    /// Initiate a multi-party payjoin
    InitiateMultiPartyPayjoin(BulletinBoardId),
}

define_entity!(
    Message,
    {
        pub(crate) id: MessageId,
        pub(crate) message: MessageType,
        pub(crate) from: WalletId,
        // None if meant as a broadcast message
        pub(crate) to: WalletId,
    },
    {
    }
);
define_entity_handle_mut!(Message);

impl<'a> MessageHandle<'a> {
    pub(crate) fn data(&self) -> &'a MessageData {
        &self.sim.messages[self.id.0]
    }
}

impl<'a> MessageHandleMut<'a> {
    pub(crate) fn post(&mut self, message: MessageData) {
        self.sim.messages.push(message);
    }
}
