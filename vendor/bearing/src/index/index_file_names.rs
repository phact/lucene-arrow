// SPDX-License-Identifier: Apache-2.0
//! Index file naming conventions and segment name generation.

/// Constructs a segment file name: `<segment>(_<suffix>)(.<ext>)`
pub fn segment_file_name(segment_name: &str, segment_suffix: &str, ext: &str) -> String {
    let mut name = segment_name.to_string();
    if !segment_suffix.is_empty() {
        name.push('_');
        name.push_str(segment_suffix);
    }
    if !ext.is_empty() {
        name.push('.');
        name.push_str(ext);
    }
    name
}

/// Converts a u64 to a radix-36 string (0-9, a-z).
/// Equivalent to Java's `Long.toString(n, Character.MAX_RADIX)`.
pub fn radix36(mut n: u64) -> String {
    if n == 0 {
        return "0".to_string();
    }
    let mut digits = Vec::new();
    while n > 0 {
        let d = (n % 36) as u8;
        let ch = if d < 10 { b'0' + d } else { b'a' + d - 10 };
        digits.push(ch as char);
        n /= 36;
    }
    digits.reverse();
    digits.into_iter().collect()
}

/// Strips the segment name prefix from a filename, returning the suffix portion.
///
/// Finds the second `_` (skipping the leading one) or the first `.`, whichever comes first,
/// and returns everything from that point onward.
///
/// Examples: `_0.fnm` → `.fnm`, `_0_Lucene90_0.dvd` → `_Lucene90_0.dvd`
pub fn strip_segment_name(filename: &str) -> &str {
    let bytes = filename.as_bytes();
    let mut underscore_count = 0;
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'.' {
            return &filename[i..];
        }
        if b == b'_' {
            underscore_count += 1;
            if underscore_count == 2 {
                return &filename[i..];
            }
        }
    }
    filename
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Constructs a file name from a base, extension, and generation.
    fn file_name_from_generation(base: &str, ext: &str, generation: i64) -> Option<String> {
        if generation == -1 {
            None
        } else if generation == 0 {
            Some(segment_file_name(base, "", ext))
        } else {
            let gen_str = radix36(generation as u64);
            let mut name = format!("{}_{}", base, gen_str);
            if !ext.is_empty() {
                name.push('.');
                name.push_str(ext);
            }
            Some(name)
        }
    }

    /// Extracts the file extension from a filename (after the first '.').
    fn get_extension(filename: &str) -> Option<&str> {
        filename.find('.').map(|idx| &filename[idx + 1..])
    }

    /// Parses the segment name from a filename.
    fn parse_segment_name(filename: &str) -> &str {
        let bytes = filename.as_bytes();
        let mut underscore_count = 0;
        for (i, &b) in bytes.iter().enumerate() {
            if b == b'.' {
                return &filename[..i];
            }
            if b == b'_' {
                underscore_count += 1;
                if underscore_count == 2 {
                    return &filename[..i];
                }
            }
        }
        filename
    }

    // Ported from org.apache.lucene.index.TestIndexFileNames

    #[test]
    fn test_segment_file_name() {
        assert_eq!(segment_file_name("_0", "", "cfs"), "_0.cfs");
        assert_eq!(segment_file_name("_0", "", "cfe"), "_0.cfe");
        assert_eq!(
            segment_file_name("_0", "Lucene90_0", "dvd"),
            "_0_Lucene90_0.dvd"
        );
    }

    #[test]
    fn test_segment_file_name_no_ext() {
        assert_eq!(segment_file_name("_0", "", ""), "_0");
        assert_eq!(segment_file_name("_0", "fnm", ""), "_0_fnm");
    }

    #[test]
    fn test_file_name_from_generation() {
        assert_eq!(file_name_from_generation("segments", "", -1), None);
        assert_eq!(
            file_name_from_generation("segments", "", 0),
            Some("segments".to_string())
        );
        assert_eq!(
            file_name_from_generation("segments", "", 1),
            Some("segments_1".to_string())
        );
    }

    #[test]
    fn test_radix36() {
        assert_eq!(radix36(0), "0");
        assert_eq!(radix36(1), "1");
        assert_eq!(radix36(10), "a");
        assert_eq!(radix36(35), "z");
        assert_eq!(radix36(36), "10");
    }

    #[test]
    fn test_get_extension() {
        assert_eq!(get_extension("_0.cfs"), Some("cfs"));
        assert_eq!(get_extension("_0"), None);
    }

    #[test]
    fn test_parse_segment_name() {
        assert_eq!(parse_segment_name("_0.cfs"), "_0");
        assert_eq!(parse_segment_name("_0_Lucene90_0.dvd"), "_0");
        assert_eq!(parse_segment_name("_a1.si"), "_a1");
    }

    // Ported from org.apache.lucene.index.TestIndexFileNames
    #[test]
    fn test_strip_segment_name() {
        assert_eq!(strip_segment_name("_0.fnm"), ".fnm");
        assert_eq!(strip_segment_name("_0_Lucene90_0.dvd"), "_Lucene90_0.dvd");
        assert_eq!(strip_segment_name("_0.cfs"), ".cfs");
        assert_eq!(strip_segment_name("_0.si"), ".si");
    }
}
