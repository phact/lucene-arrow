// SPDX-License-Identifier: Apache-2.0

//! Index file output with checksum and position tracking.

use std::io;

use crate::store::DataOutput;

/// Trait for index file output with checksum and position tracking.
pub trait IndexOutput: DataOutput + Send {
    /// Returns the name of this output (the file name).
    fn name(&self) -> &str;

    /// Returns the current write position (byte offset).
    fn file_pointer(&self) -> u64;

    /// Returns the current CRC32 checksum of all bytes written so far.
    fn checksum(&self) -> u64;

    /// Aligns the file pointer to the given power-of-2 boundary by writing zero bytes.
    fn align_file_pointer(&mut self, alignment: usize) -> io::Result<u64> {
        let pos = self.file_pointer();
        let aligned = align_offset(pos, alignment);
        let padding = (aligned - pos) as usize;
        if padding > 0 {
            const ZEROS: [u8; 16] = [0u8; 16];
            self.write_all(&ZEROS[..padding])?;
        }
        Ok(aligned)
    }
}

/// Calculates the aligned offset for the given position and alignment.
pub(crate) fn align_offset(offset: u64, alignment: usize) -> u64 {
    let a = alignment as u64;
    (offset + a - 1) & !(a - 1)
}
