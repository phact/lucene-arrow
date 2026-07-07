// SPDX-License-Identifier: Apache-2.0

//! Group-varint encoding.
//!
//! Groups of 4 integers are encoded with a flag byte (2 bits per int = byte
//! width - 1) followed by the ints in LE with variable byte widths. Remaining
//! values (< 4) are written as regular VInts.
//!
//! The decoder takes `&mut Cursor<&[u8]>` and reads directly from the cursor's
//! slice. Cursor state is undefined on error; the caller is expected to abort
//! the surrounding read.

use std::io;
use std::io::BufRead;
use std::io::Cursor;
use std::io::Write;

use crate::encoding::varint;

/// Writes integers using group-varint encoding.
pub fn write_group_vints(out: &mut dyn Write, values: &[i32], limit: usize) -> io::Result<()> {
    let mut read_pos = 0;
    let mut scratch = [0u8; 17]; // 1 flag + 4 * 4 bytes max

    while limit - read_pos >= 4 {
        let mut write_pos = 0;
        let n1m1 = num_bytes(values[read_pos]) - 1;
        let n2m1 = num_bytes(values[read_pos + 1]) - 1;
        let n3m1 = num_bytes(values[read_pos + 2]) - 1;
        let n4m1 = num_bytes(values[read_pos + 3]) - 1;
        let flag = (n1m1 << 6) | (n2m1 << 4) | (n3m1 << 2) | n4m1;
        scratch[write_pos] = flag as u8;
        write_pos += 1;

        // Write each int in LE, only the needed bytes
        for &nm1 in &[n1m1, n2m1, n3m1, n4m1] {
            let le = (values[read_pos] as u32).to_le_bytes();
            let len = nm1 as usize + 1;
            scratch[write_pos..write_pos + len].copy_from_slice(&le[..len]);
            write_pos += len;
            read_pos += 1;
        }

        out.write_all(&scratch[..write_pos])?;
    }

    // Tail values as regular VInts
    while read_pos < limit {
        varint::write_vint(out, values[read_pos])?;
        read_pos += 1;
    }

    Ok(())
}

/// Reads integers using group-varint encoding.
pub fn read_group_vints(
    cursor: &mut Cursor<&[u8]>,
    values: &mut [i32],
    limit: usize,
) -> io::Result<()> {
    let mut read_pos = 0;

    while limit - read_pos >= 4 {
        let consumed;
        {
            let buf = cursor.fill_buf()?;
            if buf.is_empty() {
                return Err(io::Error::from(io::ErrorKind::UnexpectedEof));
            }
            let flag = buf[0] as u32;
            let sizes = [
                (((flag >> 6) & 0x03) + 1) as usize,
                (((flag >> 4) & 0x03) + 1) as usize,
                (((flag >> 2) & 0x03) + 1) as usize,
                ((flag & 0x03) + 1) as usize,
            ];
            let total: usize = sizes.iter().sum();
            if buf.len() < 1 + total {
                return Err(io::Error::from(io::ErrorKind::UnexpectedEof));
            }

            let mut pos = 1;
            for &size in &sizes {
                let mut tmp = [0u8; 4];
                tmp[..size].copy_from_slice(&buf[pos..pos + size]);
                values[read_pos] = i32::from_le_bytes(tmp);
                read_pos += 1;
                pos += size;
            }
            consumed = 1 + total;
        }
        cursor.consume(consumed);
    }

    // Tail values as regular VInts
    while read_pos < limit {
        values[read_pos] = varint::read_vint_cursor(cursor)?;
        read_pos += 1;
    }

    Ok(())
}

/// Returns the number of bytes needed to represent a non-negative int (1-4).
fn num_bytes(v: i32) -> u32 {
    // 4 - (leading zeros / 8), but at least 1
    4 - ((v as u32 | 1).leading_zeros() >> 3)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Ported from org.apache.lucene.util.TestGroupVInt

    #[test]
    fn test_basic() {
        let mut buf = Vec::new();
        let values = [1, 2, 3, 4];
        write_group_vints(&mut buf, &values, 4).unwrap();
        // All values fit in 1 byte, so flag = 0b00_00_00_00 = 0x00
        // Then 1, 2, 3, 4 each as 1 byte
        assert_eq!(buf, [0x00, 1, 2, 3, 4]);
    }

    #[test]
    fn test_mixed_sizes() {
        let mut buf = Vec::new();
        let values = [1, 256, 1, 1]; // 256 needs 2 bytes
        write_group_vints(&mut buf, &values, 4).unwrap();
        // n1m1=0, n2m1=1, n3m1=0, n4m1=0 → flag = (0<<6)|(1<<4)|(0<<2)|0 = 0x10
        assert_eq!(buf[0], 0x10);
        assert_eq!(buf[1], 1); // value 1
        assert_eq!(buf[2], 0); // 256 LE low byte
        assert_eq!(buf[3], 1); // 256 LE high byte
        assert_eq!(buf[4], 1); // value 1
        assert_eq!(buf[5], 1); // value 1
    }

    #[test]
    fn test_with_tail() {
        let mut buf = Vec::new();
        let values = [1, 2, 3, 4, 5, 6];
        write_group_vints(&mut buf, &values, 6).unwrap();
        // First 4 as group, then 5 and 6 as VInts
        assert_eq!(buf[0], 0x00); // flag for [1,2,3,4]
        assert_eq!(&buf[1..5], &[1, 2, 3, 4]);
        assert_eq!(buf[5], 5); // VInt 5
        assert_eq!(buf[6], 6); // VInt 6
    }

    #[test]
    fn test_read_basic_roundtrip() {
        let values = [1, 2, 3, 4];
        let mut buf = Vec::new();
        write_group_vints(&mut buf, &values, 4).unwrap();
        let mut cursor = Cursor::new(&buf[..]);
        let mut decoded = [0i32; 4];
        read_group_vints(&mut cursor, &mut decoded, 4).unwrap();
        assert_eq!(decoded, values);
        assert_eq!(cursor.position() as usize, buf.len());
    }

    #[test]
    fn test_read_mixed_sizes_roundtrip() {
        let values = [1, 256, 1, 1];
        let mut buf = Vec::new();
        write_group_vints(&mut buf, &values, 4).unwrap();
        let mut cursor = Cursor::new(&buf[..]);
        let mut decoded = [0i32; 4];
        read_group_vints(&mut cursor, &mut decoded, 4).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_read_with_tail_roundtrip() {
        let values = [1, 2, 3, 4, 5, 6];
        let mut buf = Vec::new();
        write_group_vints(&mut buf, &values, 6).unwrap();
        let mut cursor = Cursor::new(&buf[..]);
        let mut decoded = [0i32; 6];
        read_group_vints(&mut cursor, &mut decoded, 6).unwrap();
        assert_eq!(decoded, values);
        assert_eq!(cursor.position() as usize, buf.len());
    }

    #[test]
    fn test_read_large_values_roundtrip() {
        let values = [0x01000000, 0x00010000, 0x00000100, 0x00000001];
        let mut buf = Vec::new();
        write_group_vints(&mut buf, &values, 4).unwrap();
        let mut cursor = Cursor::new(&buf[..]);
        let mut decoded = [0i32; 4];
        read_group_vints(&mut cursor, &mut decoded, 4).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_num_bytes() {
        assert_eq!(num_bytes(0), 1);
        assert_eq!(num_bytes(1), 1);
        assert_eq!(num_bytes(0xFF), 1);
        assert_eq!(num_bytes(0x100), 2);
        assert_eq!(num_bytes(0xFFFF), 2);
        assert_eq!(num_bytes(0x10000), 3);
        assert_eq!(num_bytes(0xFFFFFF), 3);
        assert_eq!(num_bytes(0x1000000), 4);
        assert_eq!(num_bytes(i32::MAX), 4);
    }
}
