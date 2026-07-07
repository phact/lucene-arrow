// SPDX-License-Identifier: Apache-2.0
//! Frame-of-reference (FOR) and patched FOR (PFOR) integer encoding utilities.

use std::io;
use std::io::BufRead;
use std::io::Cursor;
use std::io::Write;

use super::packed::bits_required;
use super::varint;

pub const BLOCK_SIZE: usize = 128;

fn write_le_int(w: &mut dyn Write, val: i32) -> io::Result<()> {
    w.write_all(&val.to_le_bytes())
}

// --- ForUtil ---
// Frame-of-Reference bit packing for 128 integers.

// MASKS arrays for remainder bit handling during encode.
const MASKS8: [i32; 8] = build_masks8();
const MASKS16: [i32; 16] = build_masks16();
const MASKS32: [i32; 32] = build_masks32();

const fn expand_mask8(mask8: i32) -> i32 {
    let m16 = mask8 | (mask8 << 8);
    m16 | (m16 << 16)
}

const fn expand_mask16(mask16: i32) -> i32 {
    mask16 | (mask16 << 16)
}

const fn build_masks8() -> [i32; 8] {
    let mut masks = [0i32; 8];
    let mut i = 0;
    while i < 8 {
        masks[i] = expand_mask8((1 << i) - 1);
        i += 1;
    }
    masks
}

const fn build_masks16() -> [i32; 16] {
    let mut masks = [0i32; 16];
    let mut i = 0;
    while i < 16 {
        masks[i] = expand_mask16((1 << i) - 1);
        i += 1;
    }
    masks
}

const fn build_masks32() -> [i32; 32] {
    let mut masks = [0i32; 32];
    let mut i = 0;
    while i < 31 {
        masks[i] = (1i32 << i) - 1;
        i += 1;
    }
    // i=31: (1<<31) overflows i32, use wrapping: 0x80000000 - 1 = 0x7FFFFFFF = -1 as i32
    masks[31] = -1;
    masks
}

/// Collapse 128 ints by interleaving 4 groups of 32, packing 4 values per int.
fn collapse8(ints: &mut [i32; BLOCK_SIZE]) {
    for i in 0..32 {
        ints[i] = (ints[i] << 24) | (ints[32 + i] << 16) | (ints[64 + i] << 8) | ints[96 + i];
    }
}

/// Collapse 128 ints by interleaving 2 groups of 64, packing 2 values per int.
fn collapse16(ints: &mut [i32; BLOCK_SIZE]) {
    for i in 0..64 {
        ints[i] = (ints[i] << 16) | ints[64 + i];
    }
}

/// Expand 32 packed ints into 128 values by extracting 4 bytes per int.
/// Reverse of [`collapse8`].
fn expand8(ints: &mut [i32; BLOCK_SIZE]) {
    for i in (0..32).rev() {
        let l = ints[i];
        ints[i] = (l >> 24) & 0xFF;
        ints[32 + i] = (l >> 16) & 0xFF;
        ints[64 + i] = (l >> 8) & 0xFF;
        ints[96 + i] = l & 0xFF;
    }
}

/// Expand 64 packed ints into 128 values by extracting 2 halfwords per int.
/// Reverse of [`collapse16`].
fn expand16(ints: &mut [i32; BLOCK_SIZE]) {
    for i in (0..64).rev() {
        let l = ints[i];
        ints[i] = (l >> 16) & 0xFFFF;
        ints[64 + i] = l & 0xFFFF;
    }
}

/// Encode 128 longs with the given bits per value using FOR bit packing.
/// Uses ForUtil thresholds: bpv<=8 → collapse8, <=16 → collapse16, else → no collapse.
pub fn encode(longs: &[i64; BLOCK_SIZE], bpv: u32, out: &mut dyn Write) -> io::Result<()> {
    if bpv == 0 {
        return Ok(());
    }
    let mut ints = [0i32; BLOCK_SIZE];
    for i in 0..BLOCK_SIZE {
        ints[i] = longs[i] as i32;
    }
    let primitive_size = if bpv <= 8 {
        collapse8(&mut ints);
        8
    } else if bpv <= 16 {
        collapse16(&mut ints);
        16
    } else {
        32
    };
    encode_ints(&ints, bpv, primitive_size, out)
}

/// Core encode: 3-phase bit packing into tmp, then write as 32-bit ints.
fn encode_ints(
    ints: &[i32; BLOCK_SIZE],
    bpv: u32,
    primitive_size: u32,
    out: &mut dyn Write,
) -> io::Result<()> {
    let num_ints = (BLOCK_SIZE as u32) * primitive_size / 32;
    let num_ints_per_shift = bpv * 4;
    let mut tmp = [0i32; BLOCK_SIZE];

    // Phase 1: shift-and-OR groups of collapsed ints into tmp
    let mut idx: usize = 0;
    let mut shift = (primitive_size - bpv) as i32;
    for t in &mut tmp[..num_ints_per_shift as usize] {
        *t = ints[idx] << shift;
        idx += 1;
    }
    shift -= bpv as i32;
    while shift >= 0 {
        for t in &mut tmp[..num_ints_per_shift as usize] {
            *t |= ints[idx] << shift;
            idx += 1;
        }
        shift -= bpv as i32;
    }

    // Phase 2: handle remainder bits for values spanning int boundaries
    let remaining_bits_per_int = (shift + bpv as i32) as u32;
    let mask_remaining = match primitive_size {
        8 => MASKS8[remaining_bits_per_int as usize],
        16 => MASKS16[remaining_bits_per_int as usize],
        _ => MASKS32[remaining_bits_per_int as usize],
    };

    let mut tmp_idx: usize = 0;
    let mut remaining_bits_per_value = bpv;
    while idx < num_ints as usize {
        if remaining_bits_per_value >= remaining_bits_per_int {
            remaining_bits_per_value -= remaining_bits_per_int;
            tmp[tmp_idx] |= (ints[idx] >> remaining_bits_per_value as i32) & mask_remaining;
            tmp_idx += 1;
            if remaining_bits_per_value == 0 {
                idx += 1;
                remaining_bits_per_value = bpv;
            }
        } else {
            let (mask1, mask2) = match primitive_size {
                8 => (
                    MASKS8[remaining_bits_per_value as usize],
                    MASKS8[(remaining_bits_per_int - remaining_bits_per_value) as usize],
                ),
                16 => (
                    MASKS16[remaining_bits_per_value as usize],
                    MASKS16[(remaining_bits_per_int - remaining_bits_per_value) as usize],
                ),
                _ => (
                    MASKS32[remaining_bits_per_value as usize],
                    MASKS32[(remaining_bits_per_int - remaining_bits_per_value) as usize],
                ),
            };
            tmp[tmp_idx] |=
                (ints[idx] & mask1) << (remaining_bits_per_int - remaining_bits_per_value) as i32;
            idx += 1;
            remaining_bits_per_value += bpv - remaining_bits_per_int;
            tmp[tmp_idx] |= (ints[idx] >> remaining_bits_per_value as i32) & mask2;
            tmp_idx += 1;
        }
    }

    // Phase 3: write output as 32-bit ints
    for &val in &tmp[..num_ints_per_shift as usize] {
        write_le_int(out, val)?;
    }

    Ok(())
}

/// Decode 128 FOR-encoded values with the given bits per value.
/// Reverse of [`encode`]: reads packed LE ints, unpacks bits, then expands.
pub fn decode(
    bpv: u32,
    cursor: &mut Cursor<&[u8]>,
    longs: &mut [i64; BLOCK_SIZE],
) -> io::Result<()> {
    if bpv == 0 {
        longs.fill(0);
        return Ok(());
    }

    let num_ints_per_shift = (bpv * 4) as usize;
    let bytes_needed = num_ints_per_shift * 4;
    let mut ints = [0i32; BLOCK_SIZE];

    {
        let buf = cursor.fill_buf()?;
        if buf.len() < bytes_needed {
            return Err(io::Error::from(io::ErrorKind::UnexpectedEof));
        }
        for (i, slot) in ints[..num_ints_per_shift].iter_mut().enumerate() {
            let off = i * 4;
            *slot = i32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
        }
    }
    cursor.consume(bytes_needed);

    let primitive_size = if bpv <= 8 {
        8
    } else if bpv <= 16 {
        16
    } else {
        32
    };

    // Decode: reverse the 3-phase bit packing
    decode_ints(&mut ints, bpv, primitive_size);

    // Expand collapsed values
    if bpv <= 8 {
        expand8(&mut ints);
    } else if bpv <= 16 {
        expand16(&mut ints);
    }

    for i in 0..BLOCK_SIZE {
        longs[i] = ints[i] as i64;
    }
    Ok(())
}

/// Core decode: extract bit-packed values from packed ints.
///
/// Reverses `encode_ints`: each packed int contains multiple values shifted and
/// OR'd together at `primitive_size` granularity. For primitive_size=8, the
/// expanded MASKS8 operate on 4 bytes in parallel (SIMD-like trick matching
/// Java's approach).
fn decode_ints(ints: &mut [i32; BLOCK_SIZE], bpv: u32, primitive_size: u32) {
    let num_ints_per_shift = (bpv * 4) as usize;
    let num_collapsed = (BLOCK_SIZE as u32 * primitive_size / 32) as usize;

    // When bpv == primitive_size, all bits are used — no masking needed
    let mask = if bpv == primitive_size {
        -1i32
    } else {
        match primitive_size {
            8 => MASKS8[bpv as usize],
            16 => MASKS16[bpv as usize],
            _ => MASKS32[bpv as usize],
        }
    };

    // Save packed data — extraction overwrites ints[]
    let mut tmp = [0i32; BLOCK_SIZE];
    tmp[..num_ints_per_shift].copy_from_slice(&ints[..num_ints_per_shift]);

    // Phase 1: extract shift-aligned groups
    // First group gets the topmost bpv bits within each primitive
    let mut idx = 0usize;
    let mut shift = (primitive_size - bpv) as i32;
    for &packed in &tmp[..num_ints_per_shift] {
        ints[idx] = (packed >> shift) & mask;
        idx += 1;
    }
    shift -= bpv as i32;
    while shift >= 0 {
        for &packed in &tmp[..num_ints_per_shift] {
            ints[idx] = (packed >> shift) & mask;
            idx += 1;
        }
        shift -= bpv as i32;
    }

    // Phase 2: extract remaining values from leftover bits in packed ints.
    // After phase 1, each packed int has `remaining_bits_per_int` unused lower bits.
    let remaining_bits_per_int = (shift + bpv as i32) as u32;
    if remaining_bits_per_int > 0 && idx < num_collapsed {
        let mask_full = match primitive_size {
            8 => MASKS8[remaining_bits_per_int as usize],
            16 => MASKS16[remaining_bits_per_int as usize],
            _ => MASKS32[remaining_bits_per_int as usize],
        };
        let mut tmp_idx = 0usize;
        let mut remaining_bits = remaining_bits_per_int;
        while idx < num_collapsed {
            let mut b = bpv as i32 - remaining_bits as i32;
            let rem_mask = match primitive_size {
                8 => MASKS8[remaining_bits as usize],
                16 => MASKS16[remaining_bits as usize],
                _ => MASKS32[remaining_bits as usize],
            };
            let mut l = (tmp[tmp_idx] & rem_mask) << b;
            tmp_idx += 1;
            while b >= remaining_bits_per_int as i32 {
                b -= remaining_bits_per_int as i32;
                l |= (tmp[tmp_idx] & mask_full) << b;
                tmp_idx += 1;
            }
            if b > 0 {
                let b_mask = match primitive_size {
                    8 => MASKS8[b as usize],
                    16 => MASKS16[b as usize],
                    _ => MASKS32[b as usize],
                };
                l |= (tmp[tmp_idx] >> (remaining_bits_per_int as i32 - b)) & b_mask;
                remaining_bits = remaining_bits_per_int - b as u32;
            } else {
                remaining_bits = remaining_bits_per_int;
            }
            ints[idx] = l;
            idx += 1;
        }
    }
}

// --- PForUtil ---
// Patched Frame-of-Reference with up to 7 exceptions.

pub const MAX_EXCEPTIONS: usize = 7;

/// Encode 128 values using PForUtil (patched FOR).
///
/// Uses a min-heap to find the top MAX_EXCEPTIONS+1 values. The heap minimum
/// determines the patched bpv. Exceptions (values exceeding the patched mask)
/// are stored as (index, highBits) byte pairs after the FOR-encoded base values.
pub fn pfor_encode(longs: &mut [i64; BLOCK_SIZE], out: &mut dyn Write) -> io::Result<()> {
    // Find the top MAX_EXCEPTIONS+1 values using a min-heap approach.
    // We maintain a sorted array of size MAX_EXCEPTIONS+1 with the minimum at [0].
    let mut top = [0u64; MAX_EXCEPTIONS + 1];
    let mut top_count = 0;

    // Seed with first MAX_EXCEPTIONS+1 values
    for &v in &longs[..=(MAX_EXCEPTIONS.min(BLOCK_SIZE - 1))] {
        top[top_count] = v as u64;
        top_count += 1;
    }
    top[..top_count].sort_unstable();
    let mut top_value = top[0]; // min of top values

    // Process remaining values
    for &v in &longs[(MAX_EXCEPTIONS + 1)..BLOCK_SIZE] {
        let v = v as u64;
        if v > top_value {
            top[0] = v;
            top[..top_count].sort_unstable();
            top_value = top[0];
        }
    }

    // Find max of all values
    let mut max: u64 = 0;
    for &t in &top[..top_count] {
        max = max.max(t);
    }

    let max_bits_required = bits_required(max);

    // Compute patched bits: reduce by at most 8 (patch stored as byte)
    let patched_bits_required = bits_required(top_value).max(max_bits_required.saturating_sub(8));

    // Count exceptions: heap entries (excluding root/min) that exceed the patched mask
    let max_unpatched_value = if patched_bits_required >= 64 {
        u64::MAX
    } else {
        (1u64 << patched_bits_required) - 1
    };
    let mut num_exceptions: u32 = 0;
    // In the heap, index 0 is the min (topValue). Count entries 1..top_count that are exceptions.
    for &t in &top[1..top_count] {
        if t > max_unpatched_value {
            num_exceptions += 1;
        }
    }

    // Collect exception pairs and mask values
    let mut exceptions = [(0u8, 0u8); MAX_EXCEPTIONS];
    let mut exception_count: usize = 0;
    if num_exceptions > 0 {
        for (i, v) in longs.iter_mut().enumerate() {
            let val = *v as u64;
            if val > max_unpatched_value {
                exceptions[exception_count] = (i as u8, (val >> patched_bits_required) as u8);
                *v = (val & max_unpatched_value) as i64;
                exception_count += 1;
            }
        }
        debug_assert_eq!(exception_count, num_exceptions as usize);
    }

    // Check all-equal (after masking) with maxBitsRequired <= 8 — matches Java's path
    let all_equal = longs.iter().all(|&v| v == longs[0]);
    if all_equal && max_bits_required <= 8 {
        // Shift exception high bits left by patchedBitsRequired (Java line 94)
        for exc in exceptions[..exception_count].iter_mut() {
            exc.1 = ((exc.1 as u64) << patched_bits_required) as u8;
        }
        let token = (num_exceptions as u8) << 5; // bpv = 0
        out.write_all(&[token])?;
        varint::write_vint(out, longs[0] as i32)?;
    } else {
        let token = ((num_exceptions as u8) << 5) | (patched_bits_required as u8);
        out.write_all(&[token])?;
        encode(longs, patched_bits_required, out)?;
    }

    // Write exception patches as (index, highBits) pairs
    for &(idx, high_bits) in &exceptions[..exception_count] {
        out.write_all(&[idx])?;
        out.write_all(&[high_bits])?;
    }

    Ok(())
}

/// Decode 128 PFOR-encoded values.
/// Reverse of [`pfor_encode`]: reads token byte, decodes base values, applies exception patches.
pub fn pfor_decode(cursor: &mut Cursor<&[u8]>, longs: &mut [i64; BLOCK_SIZE]) -> io::Result<()> {
    let token = {
        let buf = cursor.fill_buf()?;
        if buf.is_empty() {
            return Err(io::Error::from(io::ErrorKind::UnexpectedEof));
        }
        buf[0] as u32
    };
    cursor.consume(1);

    let bpv = token & 0x1F;
    if bpv == 0 {
        let value = varint::read_vint_cursor(cursor)? as i64;
        longs.fill(value);
    } else {
        decode(bpv, cursor, longs)?;
    }

    let num_exceptions = (token >> 5) as usize;
    let exception_bytes = num_exceptions * 2;
    {
        let buf = cursor.fill_buf()?;
        if buf.len() < exception_bytes {
            return Err(io::Error::from(io::ErrorKind::UnexpectedEof));
        }
        for i in 0..num_exceptions {
            let position = buf[i * 2] as usize;
            let patch = buf[i * 2 + 1] as i64;
            longs[position] |= patch << bpv;
        }
    }
    cursor.consume(exception_bytes);
    Ok(())
}

// --- ForDeltaUtil ---
// Delta encoding for doc IDs.

// These use different collapse thresholds from ForUtil.

/// Returns the bits required to encode the given raw delta values (OR-based).
/// Unlike `delta_bits_required` which takes cumulative values, this takes raw deltas directly.
pub fn for_delta_bits_required(deltas: &[i32; BLOCK_SIZE]) -> u32 {
    let mut or = 0i32;
    for &d in deltas.iter() {
        or |= d;
    }
    bits_required(or as u64)
}

/// Encode raw deltas using ForDelta thresholds:
/// bpv<=3→collapse8, <=10→collapse16, else→no collapse.
pub fn for_delta_encode(
    bpv: u32,
    deltas: &[i32; BLOCK_SIZE],
    out: &mut dyn Write,
) -> io::Result<()> {
    let mut ints = [0i32; BLOCK_SIZE];
    ints.copy_from_slice(deltas);

    let primitive_size = if bpv <= 3 {
        collapse8(&mut ints);
        8
    } else if bpv <= 10 {
        collapse16(&mut ints);
        16
    } else {
        32
    };
    encode_ints(&ints, bpv, primitive_size, out)
}

/// Decode 128 FOR-delta-encoded values with prefix-sum.
/// Reads a block of FOR-encoded deltas, then converts to absolute values via
/// running sum starting from `base` (last doc ID from the previous block).
pub fn for_delta_decode(
    bpv: u32,
    cursor: &mut Cursor<&[u8]>,
    base: i32,
    ints: &mut [i32; BLOCK_SIZE],
) -> io::Result<()> {
    if bpv == 0 {
        // All deltas are 0 — fill with base (all same value)
        ints.fill(base);
        return Ok(());
    }

    let num_ints_per_shift = (bpv * 4) as usize;
    let bytes_needed = num_ints_per_shift * 4;
    ints.fill(0);

    {
        let buf = cursor.fill_buf()?;
        if buf.len() < bytes_needed {
            return Err(io::Error::from(io::ErrorKind::UnexpectedEof));
        }
        for (i, slot) in ints[..num_ints_per_shift].iter_mut().enumerate() {
            let off = i * 4;
            *slot = i32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
        }
    }
    cursor.consume(bytes_needed);

    // ForDelta uses different collapse thresholds from ForUtil
    let primitive_size: u32 = if bpv <= 3 {
        8
    } else if bpv <= 10 {
        16
    } else {
        32
    };

    // Decode bit packing
    decode_ints(ints, bpv, primitive_size);

    // Prefix-sum with expand, matching Java's prefixSum8/16/32
    if bpv <= 3 {
        // prefixSum8: sum collapsed 32 ints (base=0), expand, then add per-group offsets
        prefix_sum(&mut ints[..32], 0);
        expand8(ints);
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
        // prefixSum16: sum collapsed 64 ints (base=0), expand, then add per-half offsets
        prefix_sum(&mut ints[..64], 0);
        expand16(ints);
        let l0 = base;
        let l1 = base + ints[63];
        for i in 0..64 {
            ints[i] += l0;
            ints[64 + i] += l1;
        }
    } else {
        // prefixSum32: simple running sum over all 128 values
        prefix_sum(&mut ints[..BLOCK_SIZE], base);
    }

    Ok(())
}

/// Running prefix sum: each element becomes the cumulative sum starting from `base`.
fn prefix_sum(arr: &mut [i32], base: i32) {
    let mut sum = base;
    for val in arr.iter_mut() {
        sum += *val;
        *val = sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Returns the number of bytes needed to encode 128 values with the given bits per value.
    fn num_bytes(bpv: u32) -> u32 {
        bpv << 4
    }

    /// Returns the bits required to encode the deltas of the given values.
    fn delta_bits_required(longs: &[i64; BLOCK_SIZE]) -> u32 {
        let mut max_delta: u64 = 0;
        let mut prev = 0i64;
        for &v in longs.iter() {
            let delta = v - prev;
            max_delta = max_delta.max(delta as u64);
            prev = v;
        }
        bits_required(max_delta)
    }

    /// Encode doc ID deltas with the given bits per value.
    fn encode_deltas(bpv: u32, longs: &[i64; BLOCK_SIZE], out: &mut dyn Write) -> io::Result<()> {
        let mut deltas = [0i64; BLOCK_SIZE];
        let mut prev = 0i64;
        for (i, &v) in longs.iter().enumerate() {
            deltas[i] = v - prev;
            prev = v;
        }
        encode(&deltas, bpv, out)
    }

    #[test]
    fn test_for_util_num_bytes() {
        assert_eq!(num_bytes(0), 0);
        assert_eq!(num_bytes(1), 16);
        assert_eq!(num_bytes(8), 128);
        assert_eq!(num_bytes(16), 256);
        assert_eq!(num_bytes(32), 512);
    }

    #[test]
    fn test_for_util_encode_all_zeros() {
        let mut out = Vec::new();
        let longs = [0i64; BLOCK_SIZE];
        encode(&longs, 0, &mut out).unwrap();
        assert_is_empty!(&out);
    }

    #[test]
    fn test_for_util_encode_1bit() {
        let mut out = Vec::new();
        let mut longs = [0i64; BLOCK_SIZE];
        for (i, long) in longs.iter_mut().enumerate() {
            *long = (i & 1) as i64;
        }
        encode(&longs, 1, &mut out).unwrap();
        assert_len_eq_x!(&&out, 16); // 1 bit * 128 / 8 = 16 bytes
    }

    #[test]
    fn test_for_util_encode_8bit() {
        let mut out = Vec::new();
        let mut longs = [0i64; BLOCK_SIZE];
        for (i, long) in longs.iter_mut().enumerate() {
            *long = i as i64;
        }
        encode(&longs, 8, &mut out).unwrap();
        assert_len_eq_x!(&&out, 128); // 8 bits * 128 / 8 = 128 bytes
    }

    #[test]
    fn test_pfor_all_equal() {
        let mut out = Vec::new();
        let mut longs = [42i64; BLOCK_SIZE];
        pfor_encode(&mut longs, &mut out).unwrap();
        let bytes = &out;
        // Token byte with bpv=0, then VLong(42)
        assert_eq!(bytes[0], 0); // token: numExceptions=0, bpv=0
        assert_eq!(bytes[1], 42); // VLong(42)
    }

    #[test]
    fn test_pfor_small_values() {
        let mut out = Vec::new();
        let mut longs = [0i64; BLOCK_SIZE];
        for (i, long) in longs.iter_mut().enumerate() {
            *long = (i % 4) as i64; // values 0-3, need 2 bits
        }
        pfor_encode(&mut longs, &mut out).unwrap();
        let bytes = &out;
        // Token byte: no exceptions, bpv=2
        let token = bytes[0];
        let num_exceptions = token >> 5;
        let bpv = token & 0x1F;
        assert_eq!(num_exceptions, 0);
        assert_eq!(bpv, 2);
        // Data: 2 bits * 128 / 8 = 32 bytes
        assert_eq!(bytes.len(), 1 + 32);
    }

    #[test]
    fn test_delta_bits_required() {
        let mut longs = [0i64; BLOCK_SIZE];
        // Sequential: deltas are all 1
        for (i, long) in longs.iter_mut().enumerate() {
            *long = i as i64;
        }
        assert_eq!(delta_bits_required(&longs), 1);

        // All same: deltas are all 0 (except first)
        let mut longs2 = [5i64; BLOCK_SIZE];
        longs2[0] = 5;
        // First delta is 5, rest are 0
        assert_eq!(delta_bits_required(&longs2), 3);
    }

    #[test]
    fn test_encode_deltas() {
        let mut out = Vec::new();
        let mut longs = [0i64; BLOCK_SIZE];
        for (i, long) in longs.iter_mut().enumerate() {
            *long = i as i64;
        }
        // Deltas are all 1 (except first which is 0), needs 1 bit
        encode_deltas(1, &longs, &mut out).unwrap();
        assert_len_eq_x!(&&out, 16); // 1 bit * 128 / 8 = 16 bytes
    }

    #[test]
    fn test_for_delta_bits_required() {
        // All deltas = 1 → OR = 1 → 1 bit
        let mut deltas = [1i32; BLOCK_SIZE];
        assert_eq!(for_delta_bits_required(&deltas), 1);

        // Deltas with max = 7 → OR includes 7 → 3 bits
        deltas[0] = 7;
        assert_eq!(for_delta_bits_required(&deltas), 3);

        // Deltas with max = 255 → 8 bits
        let mut deltas2 = [1i32; BLOCK_SIZE];
        deltas2[0] = 255;
        assert_eq!(for_delta_bits_required(&deltas2), 8);

        // Deltas with max = 1023 → 10 bits
        let mut deltas3 = [1i32; BLOCK_SIZE];
        deltas3[0] = 1023;
        assert_eq!(for_delta_bits_required(&deltas3), 10);
    }

    #[test]
    fn test_for_delta_encode_collapse8() {
        // bpv <= 3 uses collapse8 (primitive_size=8)
        let mut out = Vec::new();
        let mut deltas = [1i32; BLOCK_SIZE];
        deltas[0] = 3; // max delta = 3, bpv = 2
        let bpv = for_delta_bits_required(&deltas);
        assert_eq!(bpv, 2);
        for_delta_encode(bpv, &deltas, &mut out).unwrap();
        assert_eq!(out.len(), (bpv as usize) * 16); // bpv * 16 bytes
    }

    #[test]
    fn test_for_delta_encode_collapse16() {
        // bpv 4-10 uses collapse16 (primitive_size=16)
        let mut out = Vec::new();
        let mut deltas = [1i32; BLOCK_SIZE];
        deltas[0] = 512; // bits_required(512|1) = 10
        let bpv = for_delta_bits_required(&deltas);
        assert!((4..=10).contains(&bpv), "bpv={bpv} should use collapse16");
        for_delta_encode(bpv, &deltas, &mut out).unwrap();
        assert_eq!(out.len(), (bpv as usize) * 16);
    }

    #[test]
    fn test_for_delta_encode_collapse32() {
        // bpv > 10 uses collapse32 (primitive_size=32)
        let mut out = Vec::new();
        let mut deltas = [1i32; BLOCK_SIZE];
        deltas[0] = 2048; // bits_required(2048|1) = 12
        let bpv = for_delta_bits_required(&deltas);
        assert_gt!(bpv, 10);
        for_delta_encode(bpv, &deltas, &mut out).unwrap();
        assert_eq!(out.len(), (bpv as usize) * 16);
    }

    // --- Byte-exact cross-validation tests against Java Lucene 10.3.2 ForUtil ---
    // Expected bytes generated by ForUtilDump.java using reflection on the
    // package-private ForUtil/ForDeltaUtil/PForUtil classes.

    #[test]
    fn test_for_util_bpv1_bytes_match_java() {
        // Ported from org.apache.lucene.codecs.lucene103.TestForUtil
        let mut out = Vec::new();
        let mut longs = [0i64; BLOCK_SIZE];
        for (i, v) in longs.iter_mut().enumerate() {
            *v = (i & 1) as i64;
        }
        encode(&longs, 1, &mut out).unwrap();
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF,
            0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF,
        ];
        assert_eq!(&out, expected);
    }

    #[test]
    fn test_for_util_bpv3_bytes_match_java() {
        // Ported from org.apache.lucene.codecs.lucene103.TestForUtil
        let mut out = Vec::new();
        let mut longs = [0i64; BLOCK_SIZE];
        for (i, v) in longs.iter_mut().enumerate() {
            *v = (i % 8) as i64;
        }
        encode(&longs, 3, &mut out).unwrap();
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x10, 0x10, 0x10, 0x10, 0x34, 0x34, 0x34, 0x34,
            0x59, 0x59, 0x59, 0x59, 0x7D, 0x7D, 0x7D, 0x7D,
            0x80, 0x80, 0x80, 0x80, 0xA7, 0xA7, 0xA7, 0xA7,
            0xCA, 0xCA, 0xCA, 0xCA, 0xED, 0xED, 0xED, 0xED,
            0x11, 0x11, 0x11, 0x11, 0x37, 0x37, 0x37, 0x37,
            0x59, 0x59, 0x59, 0x59, 0x7F, 0x7F, 0x7F, 0x7F,
        ];
        assert_eq!(&out, expected);
    }

    #[test]
    fn test_for_util_bpv8_bytes_match_java() {
        // Ported from org.apache.lucene.codecs.lucene103.TestForUtil
        let mut out = Vec::new();
        let mut longs = [0i64; BLOCK_SIZE];
        for (i, v) in longs.iter_mut().enumerate() {
            *v = i as i64;
        }
        encode(&longs, 8, &mut out).unwrap();
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x60, 0x40, 0x20, 0x00, 0x61, 0x41, 0x21, 0x01,
            0x62, 0x42, 0x22, 0x02, 0x63, 0x43, 0x23, 0x03,
            0x64, 0x44, 0x24, 0x04, 0x65, 0x45, 0x25, 0x05,
            0x66, 0x46, 0x26, 0x06, 0x67, 0x47, 0x27, 0x07,
            0x68, 0x48, 0x28, 0x08, 0x69, 0x49, 0x29, 0x09,
            0x6A, 0x4A, 0x2A, 0x0A, 0x6B, 0x4B, 0x2B, 0x0B,
            0x6C, 0x4C, 0x2C, 0x0C, 0x6D, 0x4D, 0x2D, 0x0D,
            0x6E, 0x4E, 0x2E, 0x0E, 0x6F, 0x4F, 0x2F, 0x0F,
            0x70, 0x50, 0x30, 0x10, 0x71, 0x51, 0x31, 0x11,
            0x72, 0x52, 0x32, 0x12, 0x73, 0x53, 0x33, 0x13,
            0x74, 0x54, 0x34, 0x14, 0x75, 0x55, 0x35, 0x15,
            0x76, 0x56, 0x36, 0x16, 0x77, 0x57, 0x37, 0x17,
            0x78, 0x58, 0x38, 0x18, 0x79, 0x59, 0x39, 0x19,
            0x7A, 0x5A, 0x3A, 0x1A, 0x7B, 0x5B, 0x3B, 0x1B,
            0x7C, 0x5C, 0x3C, 0x1C, 0x7D, 0x5D, 0x3D, 0x1D,
            0x7E, 0x5E, 0x3E, 0x1E, 0x7F, 0x5F, 0x3F, 0x1F,
        ];
        assert_eq!(&out, expected);
    }

    #[test]
    fn test_for_util_bpv10_bytes_match_java() {
        // Ported from org.apache.lucene.codecs.lucene103.TestForUtil
        let mut out = Vec::new();
        let mut longs = [0i64; BLOCK_SIZE];
        for (i, v) in longs.iter_mut().enumerate() {
            *v = (i * 8) as i64;
        }
        encode(&longs, 10, &mut out).unwrap();
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x34, 0x80, 0x14, 0x00, 0x03, 0x82, 0x01, 0x02,
            0x12, 0x84, 0x12, 0x04, 0x0D, 0x86, 0x05, 0x06,
            0x10, 0x88, 0x10, 0x08, 0x35, 0x8A, 0x15, 0x0A,
            0x23, 0x8C, 0x21, 0x0C, 0x18, 0x8E, 0x18, 0x0E,
            0x0D, 0x90, 0x05, 0x10, 0x28, 0x92, 0x28, 0x12,
            0x37, 0x94, 0x17, 0x14, 0x03, 0x96, 0x01, 0x16,
            0x1E, 0x98, 0x1E, 0x18, 0x0E, 0x9A, 0x06, 0x1A,
            0x00, 0x9C, 0x00, 0x1C, 0x38, 0x9E, 0x18, 0x1E,
            0x23, 0xA0, 0x21, 0x20, 0x24, 0xA2, 0x24, 0x22,
            0x0E, 0xA4, 0x06, 0x24, 0x18, 0xA6, 0x18, 0x26,
            0x3A, 0xA8, 0x1A, 0x28, 0x03, 0xAA, 0x01, 0x2A,
            0x2A, 0xAC, 0x2A, 0x2C, 0x0E, 0xAE, 0x06, 0x2E,
            0x30, 0xB0, 0x30, 0x30, 0x3B, 0xB2, 0x1B, 0x32,
            0x23, 0xB4, 0x21, 0x34, 0x30, 0xB6, 0x30, 0x36,
            0x0F, 0xB8, 0x07, 0x38, 0x08, 0xBA, 0x08, 0x3A,
            0x3D, 0xBC, 0x1D, 0x3C, 0x03, 0xBE, 0x01, 0x3E,
            0x36, 0xC0, 0x36, 0x40, 0x0F, 0xC2, 0x07, 0x42,
            0x20, 0xC4, 0x20, 0x44, 0x3E, 0xC6, 0x1E, 0x46,
            0x23, 0xC8, 0x21, 0x48, 0x3C, 0xCA, 0x3C, 0x4A,
            0x0F, 0xCC, 0x07, 0x4C, 0x38, 0xCE, 0x38, 0x4E,
        ];
        assert_eq!(&out, expected);
    }

    #[test]
    fn test_for_util_bpv16_bytes_match_java() {
        // Ported from org.apache.lucene.codecs.lucene103.TestForUtil
        let mut out = Vec::new();
        let mut longs = [0i64; BLOCK_SIZE];
        for (i, v) in longs.iter_mut().enumerate() {
            *v = (i * 512) as i64;
        }
        encode(&longs, 16, &mut out).unwrap();
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x00, 0x80, 0x00, 0x00, 0x00, 0x82, 0x00, 0x02,
            0x00, 0x84, 0x00, 0x04, 0x00, 0x86, 0x00, 0x06,
            0x00, 0x88, 0x00, 0x08, 0x00, 0x8A, 0x00, 0x0A,
            0x00, 0x8C, 0x00, 0x0C, 0x00, 0x8E, 0x00, 0x0E,
            0x00, 0x90, 0x00, 0x10, 0x00, 0x92, 0x00, 0x12,
            0x00, 0x94, 0x00, 0x14, 0x00, 0x96, 0x00, 0x16,
            0x00, 0x98, 0x00, 0x18, 0x00, 0x9A, 0x00, 0x1A,
            0x00, 0x9C, 0x00, 0x1C, 0x00, 0x9E, 0x00, 0x1E,
            0x00, 0xA0, 0x00, 0x20, 0x00, 0xA2, 0x00, 0x22,
            0x00, 0xA4, 0x00, 0x24, 0x00, 0xA6, 0x00, 0x26,
            0x00, 0xA8, 0x00, 0x28, 0x00, 0xAA, 0x00, 0x2A,
            0x00, 0xAC, 0x00, 0x2C, 0x00, 0xAE, 0x00, 0x2E,
            0x00, 0xB0, 0x00, 0x30, 0x00, 0xB2, 0x00, 0x32,
            0x00, 0xB4, 0x00, 0x34, 0x00, 0xB6, 0x00, 0x36,
            0x00, 0xB8, 0x00, 0x38, 0x00, 0xBA, 0x00, 0x3A,
            0x00, 0xBC, 0x00, 0x3C, 0x00, 0xBE, 0x00, 0x3E,
            0x00, 0xC0, 0x00, 0x40, 0x00, 0xC2, 0x00, 0x42,
            0x00, 0xC4, 0x00, 0x44, 0x00, 0xC6, 0x00, 0x46,
            0x00, 0xC8, 0x00, 0x48, 0x00, 0xCA, 0x00, 0x4A,
            0x00, 0xCC, 0x00, 0x4C, 0x00, 0xCE, 0x00, 0x4E,
            0x00, 0xD0, 0x00, 0x50, 0x00, 0xD2, 0x00, 0x52,
            0x00, 0xD4, 0x00, 0x54, 0x00, 0xD6, 0x00, 0x56,
            0x00, 0xD8, 0x00, 0x58, 0x00, 0xDA, 0x00, 0x5A,
            0x00, 0xDC, 0x00, 0x5C, 0x00, 0xDE, 0x00, 0x5E,
            0x00, 0xE0, 0x00, 0x60, 0x00, 0xE2, 0x00, 0x62,
            0x00, 0xE4, 0x00, 0x64, 0x00, 0xE6, 0x00, 0x66,
            0x00, 0xE8, 0x00, 0x68, 0x00, 0xEA, 0x00, 0x6A,
            0x00, 0xEC, 0x00, 0x6C, 0x00, 0xEE, 0x00, 0x6E,
            0x00, 0xF0, 0x00, 0x70, 0x00, 0xF2, 0x00, 0x72,
            0x00, 0xF4, 0x00, 0x74, 0x00, 0xF6, 0x00, 0x76,
            0x00, 0xF8, 0x00, 0x78, 0x00, 0xFA, 0x00, 0x7A,
            0x00, 0xFC, 0x00, 0x7C, 0x00, 0xFE, 0x00, 0x7E,
        ];
        assert_eq!(&out, expected);
    }

    #[test]
    fn test_for_util_bpv20_bytes_match_java() {
        // Ported from org.apache.lucene.codecs.lucene103.TestForUtil
        let mut out = Vec::new();
        let mut longs = [0i64; BLOCK_SIZE];
        for (i, v) in longs.iter_mut().enumerate() {
            *v = (i * 8192) as i64;
        }
        encode(&longs, 20, &mut out).unwrap();
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x00, 0x0A, 0x00, 0x00, 0x0A, 0x00, 0x00, 0x02,
            0x00, 0x02, 0x00, 0x04, 0xA4, 0x00, 0x00, 0x06,
            0x00, 0x00, 0x00, 0x08, 0x60, 0x0A, 0x00, 0x0A,
            0x0A, 0x00, 0x00, 0x0C, 0x00, 0x08, 0x00, 0x0E,
            0xAA, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x12,
            0xC0, 0x0A, 0x00, 0x14, 0x0A, 0x00, 0x00, 0x16,
            0x00, 0x0E, 0x00, 0x18, 0xB0, 0x00, 0x00, 0x1A,
            0x00, 0x00, 0x00, 0x1C, 0x20, 0x0B, 0x00, 0x1E,
            0x0B, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00, 0x22,
            0xB6, 0x00, 0x00, 0x24, 0x00, 0x00, 0x00, 0x26,
            0x80, 0x0B, 0x00, 0x28, 0x0B, 0x00, 0x00, 0x2A,
            0x00, 0x0A, 0x00, 0x2C, 0xBC, 0x00, 0x00, 0x2E,
            0x00, 0x00, 0x00, 0x30, 0xE0, 0x0B, 0x00, 0x32,
            0x0C, 0x00, 0x00, 0x34, 0x00, 0x00, 0x00, 0x36,
            0xC2, 0x00, 0x00, 0x38, 0x00, 0x00, 0x00, 0x3A,
            0x40, 0x0C, 0x00, 0x3C, 0x0C, 0x00, 0x00, 0x3E,
            0x00, 0x06, 0x00, 0x40, 0xC8, 0x00, 0x00, 0x42,
            0x00, 0x00, 0x00, 0x44, 0xA0, 0x0C, 0x00, 0x46,
            0x0C, 0x00, 0x00, 0x48, 0x00, 0x0C, 0x00, 0x4A,
            0xCE, 0x00, 0x00, 0x4C, 0x00, 0x00, 0x00, 0x4E,
            0x00, 0x0D, 0x00, 0x50, 0x0D, 0x00, 0x00, 0x52,
            0x00, 0x02, 0x00, 0x54, 0xD4, 0x00, 0x00, 0x56,
            0x00, 0x00, 0x00, 0x58, 0x60, 0x0D, 0x00, 0x5A,
            0x0D, 0x00, 0x00, 0x5C, 0x00, 0x08, 0x00, 0x5E,
            0xDA, 0x00, 0x00, 0x60, 0x00, 0x00, 0x00, 0x62,
            0xC0, 0x0D, 0x00, 0x64, 0x0D, 0x00, 0x00, 0x66,
            0x00, 0x0E, 0x00, 0x68, 0xE0, 0x00, 0x00, 0x6A,
            0x00, 0x00, 0x00, 0x6C, 0x20, 0x0E, 0x00, 0x6E,
            0x0E, 0x00, 0x00, 0x70, 0x00, 0x04, 0x00, 0x72,
            0xE6, 0x00, 0x00, 0x74, 0x00, 0x00, 0x00, 0x76,
            0x80, 0x0E, 0x00, 0x78, 0x0E, 0x00, 0x00, 0x7A,
            0x00, 0x0A, 0x00, 0x7C, 0xEC, 0x00, 0x00, 0x7E,
            0x00, 0x00, 0x00, 0x80, 0xE0, 0x0E, 0x00, 0x82,
            0x0F, 0x00, 0x00, 0x84, 0x00, 0x00, 0x00, 0x86,
            0xF2, 0x00, 0x00, 0x88, 0x00, 0x00, 0x00, 0x8A,
            0x40, 0x0F, 0x00, 0x8C, 0x0F, 0x00, 0x00, 0x8E,
            0x00, 0x06, 0x00, 0x90, 0xF8, 0x00, 0x00, 0x92,
            0x00, 0x00, 0x00, 0x94, 0xA0, 0x0F, 0x00, 0x96,
            0x0F, 0x00, 0x00, 0x98, 0x00, 0x0C, 0x00, 0x9A,
            0xFE, 0x00, 0x00, 0x9C, 0x00, 0x00, 0x00, 0x9E,
        ];
        assert_eq!(&out, expected);
    }

    #[test]
    fn test_for_delta_bpv2_bytes_match_java() {
        // Ported from org.apache.lucene.codecs.lucene103.ForDeltaUtil
        let mut out = Vec::new();
        let mut deltas = [1i32; BLOCK_SIZE];
        deltas[0] = 3;
        for_delta_encode(2, &deltas, &mut out).unwrap();
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x55, 0x55, 0x55, 0xD5, 0x55, 0x55, 0x55, 0x55,
            0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55,
            0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55,
            0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55,
        ];
        assert_eq!(&out, expected);
    }

    #[test]
    fn test_for_delta_bpv5_bytes_match_java() {
        // Ported from org.apache.lucene.codecs.lucene103.ForDeltaUtil
        let mut out = Vec::new();
        let mut deltas = [1i32; BLOCK_SIZE];
        deltas[0] = 31;
        for_delta_encode(5, &deltas, &mut out).unwrap();
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x42, 0x08, 0x42, 0xF8, 0x42, 0x08, 0x42, 0x08,
            0x42, 0x08, 0x42, 0x08, 0x42, 0x08, 0x42, 0x08,
            0x43, 0x08, 0x43, 0x08, 0x42, 0x08, 0x42, 0x08,
            0x42, 0x08, 0x42, 0x08, 0x42, 0x08, 0x42, 0x08,
            0x42, 0x08, 0x42, 0x08, 0x43, 0x08, 0x43, 0x08,
            0x42, 0x08, 0x42, 0x08, 0x42, 0x08, 0x42, 0x08,
            0x42, 0x08, 0x42, 0x08, 0x42, 0x08, 0x42, 0x08,
            0x43, 0x08, 0x43, 0x08, 0x42, 0x08, 0x42, 0x08,
            0x42, 0x08, 0x42, 0x08, 0x42, 0x08, 0x42, 0x08,
            0x42, 0x08, 0x42, 0x08, 0x43, 0x08, 0x43, 0x08,
        ];
        assert_eq!(&out, expected);
    }

    #[test]
    fn test_pfor_exceptions_bytes_match_java() {
        // Ported from org.apache.lucene.codecs.lucene103.PForUtil
        let mut out = Vec::new();
        let mut longs = [0i64; BLOCK_SIZE];
        for (i, v) in longs.iter_mut().enumerate() {
            *v = (i % 4) as i64;
        }
        longs[10] = 500;
        longs[50] = 1000;
        longs[100] = 2000;
        pfor_encode(&mut longs, &mut out).unwrap();
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x63, 0x00, 0x00, 0x00, 0x00, 0x24, 0x24, 0x24,
            0x24, 0x49, 0x49, 0x49, 0x49, 0x6D, 0x6D, 0x6D,
            0x6D, 0x00, 0x00, 0x00, 0x00, 0x27, 0x27, 0x27,
            0x27, 0x48, 0x48, 0x40, 0x48, 0x6C, 0x6C, 0x6C,
            0x6C, 0x01, 0x01, 0x01, 0x01, 0x25, 0x25, 0x25,
            0x25, 0x48, 0x48, 0x48, 0x88, 0x6F, 0x6F, 0x6F,
            0x6F, 0x0A, 0x3E, 0x32, 0x7D, 0x64, 0xFA,
        ];
        assert_eq!(&out, expected);
    }

    // --- FOR decode round-trip tests ---

    #[test]
    fn test_for_decode_roundtrip_bpv1() {
        let mut longs = [0i64; BLOCK_SIZE];
        for (i, v) in longs.iter_mut().enumerate() {
            *v = (i & 1) as i64;
        }
        let mut out = Vec::new();
        encode(&longs, 1, &mut out).unwrap();

        let mut decoded = [0i64; BLOCK_SIZE];
        let mut input = Cursor::new(out.as_slice());
        decode(1, &mut input, &mut decoded).unwrap();
        assert_eq!(longs, decoded);
    }

    #[test]
    fn test_for_decode_roundtrip_bpv8() {
        let mut longs = [0i64; BLOCK_SIZE];
        for (i, v) in longs.iter_mut().enumerate() {
            *v = i as i64;
        }
        let mut out = Vec::new();
        encode(&longs, 8, &mut out).unwrap();

        let mut decoded = [0i64; BLOCK_SIZE];
        let mut input = Cursor::new(out.as_slice());
        decode(8, &mut input, &mut decoded).unwrap();
        assert_eq!(longs, decoded);
    }

    #[test]
    fn test_for_decode_roundtrip_bpv16() {
        let mut longs = [0i64; BLOCK_SIZE];
        for (i, v) in longs.iter_mut().enumerate() {
            *v = (i * 512) as i64;
        }
        let mut out = Vec::new();
        encode(&longs, 16, &mut out).unwrap();

        let mut decoded = [0i64; BLOCK_SIZE];
        let mut input = Cursor::new(out.as_slice());
        decode(16, &mut input, &mut decoded).unwrap();
        assert_eq!(longs, decoded);
    }

    #[test]
    fn test_for_decode_roundtrip_bpv20() {
        let mut longs = [0i64; BLOCK_SIZE];
        for (i, v) in longs.iter_mut().enumerate() {
            *v = (i * 8192) as i64;
        }
        let mut out = Vec::new();
        encode(&longs, 20, &mut out).unwrap();

        let mut decoded = [0i64; BLOCK_SIZE];
        let mut input = Cursor::new(out.as_slice());
        decode(20, &mut input, &mut decoded).unwrap();
        assert_eq!(longs, decoded);
    }

    #[test]
    fn test_for_decode_roundtrip_bpv0() {
        let longs = [0i64; BLOCK_SIZE];
        let mut decoded = [99i64; BLOCK_SIZE];
        let mut input = Cursor::new(&[][..]);
        decode(0, &mut input, &mut decoded).unwrap();
        assert_eq!(longs, decoded);
    }

    #[test]
    fn test_for_decode_roundtrip_bpv3() {
        let mut longs = [0i64; BLOCK_SIZE];
        for (i, v) in longs.iter_mut().enumerate() {
            *v = (i % 8) as i64;
        }
        let mut out = Vec::new();
        encode(&longs, 3, &mut out).unwrap();

        let mut decoded = [0i64; BLOCK_SIZE];
        let mut input = Cursor::new(out.as_slice());
        decode(3, &mut input, &mut decoded).unwrap();
        assert_eq!(longs, decoded);
    }

    #[test]
    fn test_for_decode_roundtrip_bpv10() {
        let mut longs = [0i64; BLOCK_SIZE];
        for (i, v) in longs.iter_mut().enumerate() {
            *v = (i * 8) as i64;
        }
        let mut out = Vec::new();
        encode(&longs, 10, &mut out).unwrap();

        let mut decoded = [0i64; BLOCK_SIZE];
        let mut input = Cursor::new(out.as_slice());
        decode(10, &mut input, &mut decoded).unwrap();
        assert_eq!(longs, decoded);
    }

    // --- PFOR decode round-trip tests ---

    #[test]
    fn test_pfor_decode_roundtrip_all_equal() {
        let mut longs = [42i64; BLOCK_SIZE];
        let mut out = Vec::new();
        pfor_encode(&mut longs, &mut out).unwrap();

        let mut decoded = [0i64; BLOCK_SIZE];
        let mut input = Cursor::new(out.as_slice());
        pfor_decode(&mut input, &mut decoded).unwrap();
        assert_eq!([42i64; BLOCK_SIZE], decoded);
    }

    #[test]
    fn test_pfor_decode_roundtrip_small_values() {
        let mut original = [0i64; BLOCK_SIZE];
        for (i, v) in original.iter_mut().enumerate() {
            *v = (i % 4) as i64;
        }
        let mut longs = original;
        let mut out = Vec::new();
        pfor_encode(&mut longs, &mut out).unwrap();

        let mut decoded = [0i64; BLOCK_SIZE];
        let mut input = Cursor::new(out.as_slice());
        pfor_decode(&mut input, &mut decoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_pfor_decode_roundtrip_with_exceptions() {
        let mut original = [0i64; BLOCK_SIZE];
        for (i, v) in original.iter_mut().enumerate() {
            *v = (i % 4) as i64;
        }
        original[10] = 500;
        original[50] = 1000;
        original[100] = 2000;

        let mut longs = original;
        let mut out = Vec::new();
        pfor_encode(&mut longs, &mut out).unwrap();

        let mut decoded = [0i64; BLOCK_SIZE];
        let mut input = Cursor::new(out.as_slice());
        pfor_decode(&mut input, &mut decoded).unwrap();
        assert_eq!(original, decoded);
    }

    // --- ForDelta decode round-trip tests ---

    #[test]
    fn test_for_delta_decode_roundtrip_bpv2() {
        let mut deltas = [1i32; BLOCK_SIZE];
        deltas[0] = 3;
        let bpv = for_delta_bits_required(&deltas);

        let mut out = Vec::new();
        for_delta_encode(bpv, &deltas, &mut out).unwrap();

        let base = 100;
        let mut decoded = [0i32; BLOCK_SIZE];
        let mut input = Cursor::new(out.as_slice());
        for_delta_decode(bpv, &mut input, base, &mut decoded).unwrap();

        // Verify: prefix-sum of deltas starting from base
        let mut expected = [0i32; BLOCK_SIZE];
        let mut sum = base;
        for (i, &d) in deltas.iter().enumerate() {
            sum += d;
            expected[i] = sum;
        }
        assert_eq!(expected, decoded);
    }

    #[test]
    fn test_for_delta_decode_roundtrip_bpv5() {
        let mut deltas = [1i32; BLOCK_SIZE];
        deltas[0] = 31;
        let bpv = for_delta_bits_required(&deltas);

        let mut out = Vec::new();
        for_delta_encode(bpv, &deltas, &mut out).unwrap();

        let base = 0;
        let mut decoded = [0i32; BLOCK_SIZE];
        let mut input = Cursor::new(out.as_slice());
        for_delta_decode(bpv, &mut input, base, &mut decoded).unwrap();

        let mut expected = [0i32; BLOCK_SIZE];
        let mut sum = base;
        for (i, &d) in deltas.iter().enumerate() {
            sum += d;
            expected[i] = sum;
        }
        assert_eq!(expected, decoded);
    }

    #[test]
    fn test_for_delta_decode_roundtrip_bpv12() {
        let mut deltas = [1i32; BLOCK_SIZE];
        deltas[0] = 2048;
        let bpv = for_delta_bits_required(&deltas);

        let mut out = Vec::new();
        for_delta_encode(bpv, &deltas, &mut out).unwrap();

        let base = 50;
        let mut decoded = [0i32; BLOCK_SIZE];
        let mut input = Cursor::new(out.as_slice());
        for_delta_decode(bpv, &mut input, base, &mut decoded).unwrap();

        let mut expected = [0i32; BLOCK_SIZE];
        let mut sum = base;
        for (i, &d) in deltas.iter().enumerate() {
            sum += d;
            expected[i] = sum;
        }
        assert_eq!(expected, decoded);
    }

    #[test]
    fn test_for_delta_decode_roundtrip_all_ones() {
        let deltas = [1i32; BLOCK_SIZE];
        let bpv = for_delta_bits_required(&deltas);

        let mut out = Vec::new();
        for_delta_encode(bpv, &deltas, &mut out).unwrap();

        let base = 0;
        let mut decoded = [0i32; BLOCK_SIZE];
        let mut input = Cursor::new(out.as_slice());
        for_delta_decode(bpv, &mut input, base, &mut decoded).unwrap();

        // Sequential doc IDs: 1, 2, 3, ..., 128
        let mut expected = [0i32; BLOCK_SIZE];
        for (i, v) in expected.iter_mut().enumerate() {
            *v = (i + 1) as i32;
        }
        assert_eq!(expected, decoded);
    }
}
