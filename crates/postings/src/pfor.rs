// SPDX-License-Identifier: Apache-2.0

//! ForDeltaUtil block decode (128 doc deltas → absolute doc ids).
//!
//! Ported from Bearing `encoding/pfor.rs` (itself a port of Lucene's
//! `ForDeltaUtil`), with one behavioral fix: the collapsed-lane prefix
//! sums use **wrapping** arithmetic. Java relies on silent int overflow
//! there — per-lane sums never carry (bpv ≤ 3 ⇒ lane sums ≤ 224 < 256),
//! but the packed 32-bit sum routinely exceeds `i32::MAX`, which panics
//! in Rust debug builds (found decoding real Java segments; Bearing's
//! `for_delta_decode` has the panic).

pub const BLOCK_SIZE: usize = 128;

fn mask8(b: u32) -> i32 {
    (0x0101_0101u32.wrapping_mul((1u32 << b) - 1)) as i32
}
fn mask16(b: u32) -> i32 {
    (0x0001_0001u32.wrapping_mul((1u32 << b) - 1)) as i32
}
fn mask32(b: u32) -> i32 {
    ((1u64 << b) - 1) as i32
}

fn mask_for(primitive: u32, b: u32) -> i32 {
    match primitive {
        8 => mask8(b),
        16 => mask16(b),
        _ => mask32(b),
    }
}

fn expand8(ints: &mut [i32; BLOCK_SIZE]) {
    for i in (0..32).rev() {
        let l = ints[i];
        ints[i] = (l >> 24) & 0xFF;
        ints[32 + i] = (l >> 16) & 0xFF;
        ints[64 + i] = (l >> 8) & 0xFF;
        ints[96 + i] = l & 0xFF;
    }
}

fn expand16(ints: &mut [i32; BLOCK_SIZE]) {
    for i in (0..64).rev() {
        let l = ints[i];
        ints[i] = (l >> 16) & 0xFFFF;
        ints[64 + i] = l & 0xFFFF;
    }
}

fn wrapping_prefix_sum(arr: &mut [i32], base: i32) {
    let mut sum = base;
    for val in arr.iter_mut() {
        sum = sum.wrapping_add(*val);
        *val = sum;
    }
}

/// Interleaved FOR bit-unpack (Lucene ForUtil layout).
fn decode_ints(ints: &mut [i32; BLOCK_SIZE], bpv: u32, primitive_size: u32) {
    let num_ints_per_shift = (bpv * 4) as usize;
    let num_collapsed = (BLOCK_SIZE as u32 * primitive_size / 32) as usize;
    let mask =
        if bpv == primitive_size { -1i32 } else { mask_for(primitive_size, bpv) };

    let mut tmp = [0i32; BLOCK_SIZE];
    tmp[..num_ints_per_shift].copy_from_slice(&ints[..num_ints_per_shift]);

    let mut idx = 0usize;
    let mut shift = (primitive_size - bpv) as i32;
    while shift >= 0 {
        for &packed in &tmp[..num_ints_per_shift] {
            ints[idx] = (packed >> shift) & mask;
            idx += 1;
        }
        shift -= bpv as i32;
    }

    let remaining_bits_per_int = (shift + bpv as i32) as u32;
    if remaining_bits_per_int > 0 && idx < num_collapsed {
        let mask_full = mask_for(primitive_size, remaining_bits_per_int);
        let mut tmp_idx = 0usize;
        let mut remaining_bits = remaining_bits_per_int;
        while idx < num_collapsed {
            let mut b = bpv as i32 - remaining_bits as i32;
            let mut l = (tmp[tmp_idx] & mask_for(primitive_size, remaining_bits)) << b;
            tmp_idx += 1;
            while b >= remaining_bits_per_int as i32 {
                b -= remaining_bits_per_int as i32;
                l |= (tmp[tmp_idx] & mask_full) << b;
                tmp_idx += 1;
            }
            if b > 0 {
                l |= (tmp[tmp_idx] >> (remaining_bits_per_int as i32 - b))
                    & mask_for(primitive_size, b as u32);
                remaining_bits = remaining_bits_per_int - b as u32;
            } else {
                remaining_bits = remaining_bits_per_int;
            }
            ints[idx] = l;
            idx += 1;
        }
    }
}

/// Decode one ForDelta block: `bpv`-packed deltas at `bytes`, prefix-summed
/// from `base`. Returns `(docs, bytes_consumed)`.
pub fn for_delta_decode(bpv: u32, bytes: &[u8], base: i32) -> Option<([i32; BLOCK_SIZE], usize)> {
    let num_ints_per_shift = (bpv * 4) as usize;
    let needed = num_ints_per_shift * 4;
    if bytes.len() < needed || bpv == 0 || bpv > 32 {
        return None;
    }
    let mut ints = [0i32; BLOCK_SIZE];
    for (i, slot) in ints[..num_ints_per_shift].iter_mut().enumerate() {
        let off = i * 4;
        *slot = i32::from_le_bytes(bytes[off..off + 4].try_into().ok()?);
    }

    let primitive_size: u32 = if bpv <= 3 {
        8
    } else if bpv <= 10 {
        16
    } else {
        32
    };
    decode_ints(&mut ints, bpv, primitive_size);

    if bpv <= 3 {
        // prefixSum8: lane-wise sums in the collapsed form (wrapping —
        // lanes cannot carry at bpv ≤ 3), then expand and add offsets.
        wrapping_prefix_sum(&mut ints[..32], 0);
        expand8(&mut ints);
        let l0 = base;
        let l1 = l0 + ints[31];
        let l2 = l1 + ints[63];
        let l3 = l2 + ints[95];
        for i in 0..32 {
            ints[i] += l0;
            ints[32 + i] += l1;
            ints[64 + i] += l2;
            ints[96 + i] += l3;
        }
    } else if bpv <= 10 {
        wrapping_prefix_sum(&mut ints[..64], 0);
        expand16(&mut ints);
        let l0 = base;
        let l1 = base + ints[63];
        for i in 0..64 {
            ints[i] += l0;
            ints[64 + i] += l1;
        }
    } else {
        wrapping_prefix_sum(&mut ints[..BLOCK_SIZE], base);
    }
    Some((ints, needed))
}
