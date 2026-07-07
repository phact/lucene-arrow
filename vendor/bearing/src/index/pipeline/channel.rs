// SPDX-License-Identifier: Apache-2.0

use std::io;

use crate::document::Document;

/// Sending half of a bounded channel.
#[derive(Clone)]
pub struct Sender {
    inner: crossbeam_channel::Sender<Document>,
}

/// Receiving half of a bounded channel.
#[derive(Clone)]
pub struct Receiver {
    inner: crossbeam_channel::Receiver<Document>,
}

/// Creates a bounded channel with the given capacity.
///
/// `send()` blocks when the channel is full.
/// Returns the sending and receiving halves.
pub fn bounded(capacity: usize) -> (Sender, Receiver) {
    let (tx, rx) = crossbeam_channel::bounded(capacity);
    (Sender { inner: tx }, Receiver { inner: rx })
}

impl Sender {
    /// Sends a document into the channel. Blocks if the channel is full.
    pub fn send(&self, doc: Document) -> io::Result<()> {
        self.inner
            .send(doc)
            .map_err(|_| io::Error::other("channel closed"))
    }
}

impl Receiver {
    /// Receives a document from the channel. Blocks if empty.
    /// Returns `None` when the channel is closed and drained.
    pub fn recv(&self) -> Option<Document> {
        self.inner.recv().ok()
    }
}
