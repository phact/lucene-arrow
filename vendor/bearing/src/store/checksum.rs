// SPDX-License-Identifier: Apache-2.0

//! CRC32 checksum implementation compatible with `java.util.zip.CRC32`.

// ISO 3309 / ITU-T V.42, polynomial 0xEDB88320 (reflected representation)

/// CRC32 checksum compatible with java.util.zip.CRC32.
/// Uses the ISO 3309 polynomial (0xEDB88320 reflected).
pub struct CRC32 {
    crc: u32,
}

impl Default for CRC32 {
    fn default() -> Self {
        Self::new()
    }
}

impl CRC32 {
    const TABLE: [u32; 256] = Self::make_table();

    pub fn new() -> Self {
        Self { crc: 0xFFFFFFFF }
    }

    /// Updates the checksum with a single byte.
    pub fn update_byte(&mut self, b: u8) {
        self.crc = Self::TABLE[((self.crc ^ b as u32) & 0xFF) as usize] ^ (self.crc >> 8);
    }

    /// Updates the checksum with a slice of bytes.
    pub fn update(&mut self, buf: &[u8]) {
        for &b in buf {
            self.update_byte(b);
        }
    }

    /// Returns the current CRC32 value (finalized with XOR).
    pub fn value(&self) -> u64 {
        (self.crc ^ 0xFFFFFFFF) as u64
    }

    /// Resets the checksum to initial state.
    pub fn reset(&mut self) {
        self.crc = 0xFFFFFFFF;
    }

    const fn make_table() -> [u32; 256] {
        let mut table = [0u32; 256];
        let mut i = 0u32;
        while i < 256 {
            let mut crc = i;
            let mut j = 0;
            while j < 8 {
                if crc & 1 != 0 {
                    crc = (crc >> 1) ^ 0xEDB88320;
                } else {
                    crc >>= 1;
                }
                j += 1;
            }
            table[i as usize] = crc;
            i += 1;
        }
        table
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Ported from java.util.zip.CRC32 known values

    #[test]
    fn test_empty() {
        let crc = CRC32::new();
        assert_eq!(crc.value(), 0);
    }

    #[test]
    fn test_single_byte() {
        let mut crc = CRC32::new();
        crc.update_byte(0);
        // Java: new CRC32(); crc.update(0); => 3523407757L
        assert_eq!(crc.value(), 0xD202EF8D);
    }

    #[test]
    fn test_check_value() {
        // The "check value" for CRC32: CRC of bytes "123456789" = 0xCBF43926
        let mut crc = CRC32::new();
        crc.update(b"123456789");
        assert_eq!(crc.value(), 0xCBF43926);
    }

    #[test]
    fn test_hello() {
        // Java: CRC32 over "hello" bytes => 907060870 (0x3610A686)
        let mut crc = CRC32::new();
        crc.update(b"hello");
        assert_eq!(crc.value(), 0x3610A686);
    }

    #[test]
    fn test_reset() {
        let mut crc = CRC32::new();
        crc.update(b"hello");
        crc.reset();
        assert_eq!(crc.value(), 0);
    }

    #[test]
    fn test_incremental() {
        // CRC should be same whether computed in one call or incrementally
        let mut crc1 = CRC32::new();
        crc1.update(b"hello world");

        let mut crc2 = CRC32::new();
        crc2.update(b"hello ");
        crc2.update(b"world");

        assert_eq!(crc1.value(), crc2.value());
    }

    // Ported from org.apache.lucene.store.TestBufferedChecksum

    #[test]
    fn test_crc32_known_value() {
        // Ported from org.apache.lucene.store.TestBufferedChecksum.testSimple
        let mut crc = CRC32::new();
        crc.update_byte(1);
        crc.update_byte(2);
        crc.update_byte(3);
        assert_eq!(crc.value(), 1438416925);
    }

    #[test]
    fn test_crc32_batch_vs_incremental() {
        // Ported from org.apache.lucene.store.TestBufferedChecksum
        let mut batch = CRC32::new();
        batch.update(&[1, 2, 3]);

        let mut incremental = CRC32::new();
        incremental.update_byte(1);
        incremental.update_byte(2);
        incremental.update_byte(3);

        assert_eq!(batch.value(), incremental.value());
    }

    #[test]
    fn test_crc32_reset_clears_state() {
        let data = b"some bytes to checksum";

        let mut crc = CRC32::new();
        crc.update(data);
        crc.reset();
        crc.update(data);

        let mut fresh = CRC32::new();
        fresh.update(data);

        assert_eq!(crc.value(), fresh.value());
    }

    #[test]
    fn test_crc32_large_buffer() {
        let buf: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        let mut crc = CRC32::new();
        crc.update(&buf);
        assert_eq!(crc.value(), 1961098049);
    }

    #[test]
    fn test_crc32_all_zero_bytes() {
        let buf = [0u8; 256];
        let mut crc = CRC32::new();
        crc.update(&buf);
        assert_eq!(crc.value(), 227968344);
    }

    #[test]
    fn test_crc32_default() {
        let crc = CRC32::default();
        assert_eq!(crc.value(), 0);
    }
}
