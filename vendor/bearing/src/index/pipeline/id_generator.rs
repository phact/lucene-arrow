// SPDX-License-Identifier: Apache-2.0

//! Segment identity generation.

use crate::util::string_helper;

/// Generates unique 16-byte identifiers for segments.
///
/// Production implementations use OS randomness. Test implementations
/// return deterministic values for reproducible output.
pub trait IdGenerator: Send {
    /// Returns the next 16-byte identifier.
    fn next_id(&mut self) -> [u8; 16];
}

/// Generates random 16-byte segment identifiers.
///
/// Wraps the shared `string_helper::random_id()` utility.
#[derive(Debug)]
pub struct RandomIdGenerator;

impl IdGenerator for RandomIdGenerator {
    fn next_id(&mut self) -> [u8; 16] {
        string_helper::random_id()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_16_bytes() {
        let mut id_gen = RandomIdGenerator;
        let id = id_gen.next_id();
        assert_eq!(id.len(), 16);
    }

    #[test]
    fn successive_ids_differ() {
        let mut id_gen = RandomIdGenerator;
        let a = id_gen.next_id();
        let b = id_gen.next_id();
        assert_ne!(a, b);
    }
}
