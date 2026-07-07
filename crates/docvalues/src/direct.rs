// SPDX-License-Identifier: Apache-2.0

//! Lucene `DirectWriter`/`DirectReader` packing (bulk form).
//!
//! Layout (all little-endian; see `reference/formats/lucene90-formats.md`):
//! byte-aligned widths write `bpv/8` LE bytes per value; widths 1/2/4 pack
//! LSB-first into LE longs; widths 12/20/28 pack value pairs
//! (`v1 | v2 << bpv`) into 3/5/7 LE bytes. `finish` appends padding so
//! readers can over-read safely.
//!
//! The pack side mirrors Bearing's `DirectWriter` byte-for-byte; the unpack
//! side is the bulk (whole-block) inverse — random access is a non-goal,
//! executors decode blocks whole (SPEC §11.3).

use lucene_arrow_core::{Error, Result};

pub const SUPPORTED_BITS_PER_VALUE: [u8; 14] = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64];

/// Bits needed for `max_value` (unsigned), rounded up to a supported width.
/// Matches Java `DirectWriter.unsignedBitsRequired` (0 → 1 bit).
pub fn unsigned_bits_required(max_value: i64) -> u8 {
    let raw = if max_value == 0 { 1 } else { 64 - (max_value as u64).leading_zeros() as u8 };
    *SUPPORTED_BITS_PER_VALUE
        .iter()
        .find(|&&s| s >= raw)
        .expect("raw bits cannot exceed 64")
}

/// Pack `values` at `bits_per_value` and append to `out`, including the
/// trailing padding bytes Java's `DirectWriter.finish` writes.
pub fn pack(values: &[i64], bits_per_value: u8, out: &mut Vec<u8>) {
    let bpv = bits_per_value as u32;
    if bpv == 0 {
        return;
    }
    let up_to = values.len();

    if bpv & 7 == 0 {
        let bytes_per_value = (bpv / 8) as usize;
        for &v in values {
            out.extend_from_slice(&(v as u64).to_le_bytes()[..bytes_per_value]);
        }
    } else if bpv < 8 {
        let values_per_long = (64 / bpv) as usize;
        let mut i = 0;
        while i < up_to {
            let mut packed: u64 = 0;
            for j in 0..values_per_long {
                if i + j < up_to {
                    packed |= (values[i + j] as u64) << (bpv * j as u32);
                }
            }
            let remaining = (up_to - i).min(values_per_long);
            let bytes_needed = (remaining * bpv as usize).div_ceil(8);
            out.extend_from_slice(&packed.to_le_bytes()[..bytes_needed]);
            i += values_per_long;
        }
    } else {
        // bpv 12, 20, 28: pairs
        let num_bytes_for_2 = (bpv * 2 / 8) as usize;
        let mut i = 0;
        while i < up_to {
            let l1 = values[i] as u64;
            let l2 = if i + 1 < up_to { values[i + 1] as u64 } else { 0 };
            let merged = l1 | (l2 << bpv);
            if bpv <= 16 {
                out.extend_from_slice(&(merged as u32).to_le_bytes()[..num_bytes_for_2]);
            } else {
                out.extend_from_slice(&merged.to_le_bytes()[..num_bytes_for_2]);
            }
            i += 2;
        }
    }

    // Padding for fast reads (Java DirectWriter.finish).
    let padding_bits = if bpv > 32 {
        64 - bpv
    } else if bpv > 16 {
        32 - bpv
    } else if bpv > 8 {
        16 - bpv
    } else {
        0
    };
    out.extend(std::iter::repeat_n(0u8, padding_bits.div_ceil(8) as usize));
}

/// Exact packed byte length `pack` produces for `count` values (payload +
/// padding). Useful for plan validation.
pub fn packed_len(count: usize, bits_per_value: u8) -> usize {
    let bpv = bits_per_value as usize;
    if bpv == 0 {
        return 0;
    }
    let payload = if bpv.is_multiple_of(8) {
        count * bpv / 8
    } else if bpv < 8 {
        let vpl = 64 / bpv;
        let full = count / vpl * 8;
        let rem = count % vpl;
        full + (rem * bpv).div_ceil(8)
    } else {
        count.div_ceil(2) * (bpv * 2 / 8)
    };
    let padding_bits =
        if bpv > 32 { 64 - bpv } else if bpv > 16 { 32 - bpv } else if bpv > 8 { 16 - bpv } else { 0 };
    payload + padding_bits.div_ceil(8)
}

/// Fused single-pass unpack: calls `emit` once per value, in order, with
/// no intermediate buffer (SPEC §11.3 "one read of packed bytes, one write
/// of Arrow lanes" — the epilogue lives in the closure and inlines).
///
/// Same bitstream fact the GPU kernel exploits: value `i` occupies bits
/// `[i·bpv, (i+1)·bpv)` little-endian, for every supported width.
pub fn for_each_unpacked(
    bytes: &[u8],
    bits_per_value: u8,
    count: usize,
    mut emit: impl FnMut(u64),
) -> Result<()> {
    let bpv = bits_per_value as u32;
    if !SUPPORTED_BITS_PER_VALUE.contains(&bits_per_value) {
        return Err(Error::corrupt(format!("unsupported bits per value: {bits_per_value}")));
    }
    // Total payload bits; padding guarantees at least ceil(count*bpv/8)
    // bytes exist, verified once up front so the hot loop is check-free.
    let need = (count as u64 * bpv as u64).div_ceil(8) as usize;
    if bytes.len() < need {
        return Err(Error::corrupt(format!(
            "packed data truncated: need {need} bytes, have {}",
            bytes.len()
        )));
    }

    let mask = if bpv == 64 { u64::MAX } else { (1u64 << bpv) - 1 };
    // Fast path while a full unaligned 8-byte read is in bounds; scalar
    // byte-assembly tail for the last few values.
    let mut i = 0usize;
    if bytes.len() >= 8 {
        let last_full = bytes.len() - 8;
        while i < count {
            let bit = i as u64 * bpv as u64;
            let byte = (bit >> 3) as usize;
            if byte > last_full {
                break;
            }
            // Safety: byte + 8 <= bytes.len() by the check above.
            let word = u64::from_le_bytes(
                unsafe { bytes.get_unchecked(byte..byte + 8) }.try_into().expect("8 bytes"),
            );
            let shift = (bit & 7) as u32;
            if bpv + shift <= 64 {
                emit((word >> shift) & mask);
                i += 1;
            } else {
                // Only possible for bpv == 64 with shift == 0 handled above;
                // widths >56 are byte-aligned so shift == 0 always.
                unreachable!("supported widths never straddle 9 bytes");
            }
        }
    }
    while i < count {
        let bit = i as u64 * bpv as u64;
        let byte = (bit >> 3) as usize;
        let avail = bytes.len() - byte;
        let mut buf = [0u8; 8];
        buf[..avail.min(8)].copy_from_slice(&bytes[byte..byte + avail.min(8)]);
        let word = u64::from_le_bytes(buf);
        let shift = (bit & 7) as u32;
        emit((word >> shift) & mask);
        i += 1;
    }
    Ok(())
}

/// Unpack `count` unsigned values of `bits_per_value` from `bytes`.
///
/// `bytes` may be longer than needed (padding, coalesced extents); only the
/// packed payload is read. This is the CPU reference for the GPU
/// funnel-shift kernels (SPEC §11.3): same input, bit-identical output.
pub fn unpack(bytes: &[u8], bits_per_value: u8, count: usize) -> Result<Vec<u64>> {
    let bpv = bits_per_value as u32;
    if !SUPPORTED_BITS_PER_VALUE.contains(&bits_per_value) {
        return Err(Error::corrupt(format!("unsupported bits per value: {bits_per_value}")));
    }
    let mut out = Vec::with_capacity(count);

    if bpv & 7 == 0 {
        let bytes_per_value = (bpv / 8) as usize;
        let need = count * bytes_per_value;
        if bytes.len() < need {
            return Err(Error::corrupt(format!("packed data truncated: need {need}, have {}", bytes.len())));
        }
        for chunk in bytes[..need].chunks_exact(bytes_per_value) {
            let mut b = [0u8; 8];
            b[..bytes_per_value].copy_from_slice(chunk);
            out.push(u64::from_le_bytes(b));
        }
    } else if bpv < 8 {
        let values_per_long = (64 / bpv) as usize;
        let mask = (1u64 << bpv) - 1;
        let mut i = 0;
        let mut off = 0;
        while i < count {
            let remaining = (count - i).min(values_per_long);
            let bytes_needed = (remaining * bpv as usize).div_ceil(8);
            if bytes.len() < off + bytes_needed {
                return Err(Error::corrupt("packed data truncated (sub-byte group)"));
            }
            let mut b = [0u8; 8];
            b[..bytes_needed].copy_from_slice(&bytes[off..off + bytes_needed]);
            let packed = u64::from_le_bytes(b);
            for j in 0..remaining {
                out.push((packed >> (bpv * j as u32)) & mask);
            }
            off += bytes_needed;
            i += values_per_long;
        }
    } else {
        // bpv 12, 20, 28: pairs in 3/5/7 bytes
        let num_bytes_for_2 = (bpv * 2 / 8) as usize;
        let mask = (1u64 << bpv) - 1;
        let mut i = 0;
        let mut off = 0;
        while i < count {
            if bytes.len() < off + num_bytes_for_2 {
                return Err(Error::corrupt("packed data truncated (pair group)"));
            }
            let mut b = [0u8; 8];
            b[..num_bytes_for_2].copy_from_slice(&bytes[off..off + num_bytes_for_2]);
            let merged = u64::from_le_bytes(b);
            out.push(merged & mask);
            if i + 1 < count {
                out.push((merged >> bpv) & mask);
            }
            off += num_bytes_for_2;
            i += 2;
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_required_rounds_to_supported() {
        assert_eq!(unsigned_bits_required(0), 1);
        assert_eq!(unsigned_bits_required(1), 1);
        assert_eq!(unsigned_bits_required(2), 2);
        assert_eq!(unsigned_bits_required(255), 8);
        assert_eq!(unsigned_bits_required(256), 12);
        assert_eq!(unsigned_bits_required(1 << 20), 24);
        assert_eq!(unsigned_bits_required(-1), 64); // as unsigned: 2^64-1
    }

    #[test]
    fn pack_unpack_round_trips_every_width() {
        for &bpv in &SUPPORTED_BITS_PER_VALUE {
            for count in [1usize, 2, 3, 5, 16, 17, 64, 129] {
                let mask = if bpv == 64 { u64::MAX } else { (1u64 << bpv) - 1 };
                let values: Vec<i64> = (0..count)
                    .map(|i| ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) & mask) as i64)
                    .collect();
                let mut packed = Vec::new();
                pack(&values, bpv, &mut packed);
                assert_eq!(packed.len(), packed_len(count, bpv), "len bpv={bpv} count={count}");
                let unpacked = unpack(&packed, bpv, count).unwrap();
                let expected: Vec<u64> = values.iter().map(|&v| v as u64).collect();
                assert_eq!(unpacked, expected, "bpv={bpv} count={count}");
            }
        }
    }

    #[test]
    fn unpack_rejects_truncated_input() {
        let mut packed = Vec::new();
        pack(&[1, 2, 3, 4], 12, &mut packed);
        assert!(unpack(&packed[..packed.len() - 3], 12, 4).is_err());
    }
}
