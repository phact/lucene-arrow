// SPDX-License-Identifier: Apache-2.0

//! DirectMonotonicWriter/Reader (bulk form) — monotone sequences in
//! `2^block_shift`-value blocks; 21 bytes of metadata per block
//! (`min: i64 LE`, `avgInc: f32 bits LE`, `offset: i64 LE relative to the
//! writer's base`, `bits: u8`) plus DirectWriter-packed deltas in the data
//! stream. Used for ord→doc maps (vectors), List offsets and term
//! addresses (P3). Write side mirrors Bearing/Java byte-for-byte.

use crate::direct;
use lucene_arrow_core::cursor::Cursor;
use lucene_arrow_core::{Error, Result};

pub const META_BYTES_PER_BLOCK: usize = 8 + 4 + 8 + 1;

/// Encode `values` (monotonically non-decreasing). Meta bytes append to
/// `meta`, packed deltas append to `data`; `data`'s length at entry is the
/// base the per-block offsets are relative to.
pub fn write(values: &[i64], block_shift: u32, meta: &mut Vec<u8>, data: &mut Vec<u8>) -> Result<()> {
    if values.windows(2).any(|w| w[0] > w[1]) {
        return Err(Error::invalid("DirectMonotonic input must be non-decreasing"));
    }
    let block_size = 1usize << block_shift;
    let base = data.len() as i64;

    for block in values.chunks(block_size) {
        let count = block.len();
        let avg_inc = (block[count - 1] - block[0]) as f64 / (count - 1).max(1) as f64;
        let avg_inc_f = avg_inc as f32;

        let mut deltas: Vec<i64> = block
            .iter()
            .enumerate()
            .map(|(i, &v)| v - (avg_inc_f * i as f32) as i64)
            .collect();
        let min = deltas.iter().copied().min().expect("non-empty block");
        let mut max_delta = 0i64;
        for d in &mut deltas {
            *d -= min;
            max_delta |= *d;
        }

        meta.extend_from_slice(&min.to_le_bytes());
        meta.extend_from_slice(&avg_inc_f.to_bits().to_le_bytes());
        meta.extend_from_slice(&(data.len() as i64 - base).to_le_bytes());
        if max_delta == 0 {
            meta.push(0);
        } else {
            let bits = direct::unsigned_bits_required(max_delta);
            direct::pack(&deltas, bits, data);
            meta.push(bits);
        }
    }
    Ok(())
}

/// One parsed meta block.
#[derive(Debug, Clone, Copy)]
pub struct MetaBlock {
    pub min: i64,
    pub avg_inc: f32,
    pub offset: u64,
    pub bits: u8,
}

/// Parse `num_values`-worth of meta blocks from the cursor.
pub fn read_meta(c: &mut Cursor<'_>, num_values: u64, block_shift: u32) -> Result<Vec<MetaBlock>> {
    let num_blocks = num_values.div_ceil(1u64 << block_shift);
    let mut blocks = Vec::with_capacity(num_blocks as usize);
    for _ in 0..num_blocks {
        let min = c.le_i64()?;
        let avg_inc = f32::from_bits(c.le_i32()? as u32);
        let offset = c.le_i64()?;
        let bits = c.u8()?;
        if offset < 0 {
            return Err(Error::corrupt("negative DirectMonotonic block offset"));
        }
        blocks.push(MetaBlock { min, avg_inc, offset: offset as u64, bits });
    }
    Ok(blocks)
}

/// Bulk-decode the whole sequence. `data` is the region starting at the
/// writer's base pointer.
pub fn decode(
    blocks: &[MetaBlock],
    data: &[u8],
    num_values: u64,
    block_shift: u32,
) -> Result<Vec<i64>> {
    let block_size = 1u64 << block_shift;
    let mut out = Vec::with_capacity(num_values as usize);
    for (bi, b) in blocks.iter().enumerate() {
        let count = (num_values - bi as u64 * block_size).min(block_size) as usize;
        if b.bits == 0 {
            for i in 0..count {
                out.push(b.min + (b.avg_inc * i as f32) as i64);
            }
        } else {
            let payload = data
                .get(b.offset as usize..)
                .ok_or_else(|| Error::corrupt("DirectMonotonic offset beyond data"))?;
            let mut i = 0usize;
            direct::for_each_unpacked(payload, b.bits, count, |x| {
                out.push(b.min + (b.avg_inc * i as f32) as i64 + x as i64);
                i += 1;
            })?;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_ord_to_doc_shapes() {
        for values in [
            (0..100_000i64).map(|i| i * 3 + (i % 17) * 17).scan(0, |acc, v| { *acc = v.max(*acc); Some(*acc) }).collect::<Vec<_>>(),
            vec![5; 1000],                       // constant (bits == 0)
            (0..70_000i64).collect::<Vec<_>>(),  // perfectly linear
        ] {
            let mut meta = Vec::new();
            let mut data = Vec::new();
            write(&values, 16, &mut meta, &mut data).unwrap();
            let mut c = Cursor::new(&meta);
            let blocks = read_meta(&mut c, values.len() as u64, 16).unwrap();
            assert_eq!(c.remaining(), 0);
            let decoded = decode(&blocks, &data, values.len() as u64, 16).unwrap();
            assert_eq!(decoded, values);
        }
    }
}
