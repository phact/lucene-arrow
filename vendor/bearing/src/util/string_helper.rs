// SPDX-License-Identifier: Apache-2.0
//! String and ID utilities: random ID generation and byte sorting helpers.

use std::sync::Mutex;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

/// Length of segment IDs in bytes.
pub const ID_LENGTH: usize = 16;

/// Generates a 16-byte random ID.
/// Uses a simple xorshift128-based PRNG seeded from system time and thread ID.
/// In Java, this uses SecureRandom; here we use a simpler approach since
/// the IDs only need to be unique, not cryptographically secure.
pub fn random_id() -> [u8; ID_LENGTH] {
    static STATE: Mutex<Option<u128>> = Mutex::new(None);

    let mut state = STATE.lock().unwrap();
    if state.is_none() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        let seed_hi = now.as_nanos() as u64;
        let seed_lo = {
            let tid = format!("{:?}", thread::current().id());
            tid.bytes()
                .fold(0u64, |h, b| h.wrapping_mul(31).wrapping_add(b as u64))
                | 1
        };

        // Mix with xorshift
        let mut x0 = seed_hi;
        let mut x1 = seed_lo;
        for _ in 0..10 {
            let s0 = x1;
            let s1 = x0 ^ (x0 << 23);
            x0 = s0;
            x1 = s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26);
        }

        *state = Some(((x0 as u128) << 64) | (x1 as u128));
    }

    let val = state.unwrap();
    let next = val.wrapping_add(1);
    *state = Some(next);

    val.to_be_bytes()
}

/// Returns the length of the shared prefix between two byte slices.
pub fn bytes_difference(a: &[u8], b: &[u8]) -> usize {
    let min_len = a.len().min(b.len());
    for i in 0..min_len {
        if a[i] != b[i] {
            return i;
        }
    }
    min_len
}

/// Returns the sort key length: the first differing byte position + 1.
pub fn sort_key_length(prior_term: &[u8], current_term: &[u8]) -> usize {
    bytes_difference(prior_term, current_term) + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_id_length() {
        let id = random_id();
        assert_len_eq_x!(&id, ID_LENGTH);
    }

    #[test]
    fn test_random_id_unique() {
        let id1 = random_id();
        let id2 = random_id();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_bytes_difference() {
        assert_eq!(bytes_difference(b"abc", b"abd"), 2);
        assert_eq!(bytes_difference(b"abc", b"abc"), 3);
        assert_eq!(bytes_difference(b"abc", b"xyz"), 0);
        assert_eq!(bytes_difference(b"", b"abc"), 0);
    }

    #[test]
    fn test_sort_key_length() {
        assert_eq!(sort_key_length(b"abc", b"abd"), 3);
        assert_eq!(sort_key_length(b"abc", b"abc"), 4);
        assert_eq!(sort_key_length(b"abc", b"xyz"), 1);
    }
}
