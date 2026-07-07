// SPDX-License-Identifier: Apache-2.0

//! Standard analysis: [`StandardTokenizer`] and [`StandardAnalyzer`].

use std::io::{self, Read};

use crate::analysis::chunk_reader::Utf8ChunkReader;
use crate::analysis::{Analyzer, AnalyzerFactory, Token};
use crate::document::TermOffset;

/// Maximum token length. Tokens longer than this are split.
/// Matches Java: StandardAnalyzer.DEFAULT_MAX_TOKEN_LENGTH = 255
const MAX_TOKEN_LENGTH: usize = 255;

/// Token classification helpers for UAX#29-like word break rules.
///
/// Token rules:
/// - Alphanumeric sequences are tokens
/// - Internal apostrophes (e.g., "don't") are kept as part of the token
/// - Everything else is a token separator
pub struct StandardTokenizer;

impl StandardTokenizer {
    /// Returns true if the character can be part of a word token.
    fn is_word_char(c: char) -> bool {
        c.is_alphanumeric() || c == '_'
    }

    /// Returns true if the character is an internal separator that can
    /// appear within a token (apostrophe, period, hyphen in certain contexts).
    fn is_internal_separator(c: char) -> bool {
        c == '\'' || c == '\u{2019}' // apostrophe and right single quotation mark
    }
}

/// Fast text analyzer with ASCII tokenization and lowercase normalization.
///
/// Splits on non-alphanumeric characters, keeps internal apostrophes
/// (e.g., "don't"), and lowercases ASCII. This is the default analyzer
/// and the fastest option for English text.
///
/// For correct handling of CJK, numeric grouping (`1,200`), and URLs,
/// use [`UnicodeAnalyzer`](super::unicode::UnicodeAnalyzer) instead.
#[derive(Default)]
pub struct StandardAnalyzer {
    chunk_reader: Option<Utf8ChunkReader>,
    /// Current chunk, ASCII-lowercased. Tokens borrow from this.
    current: String,
    /// Byte scan position in `current`.
    pos: usize,
    /// Buffer for tokens that span a chunk boundary.
    /// Also used for tokens truncated at MAX_TOKEN_LENGTH when the skip
    /// crosses a chunk boundary.
    boundary_buf: String,
    /// Total bytes consumed before the current chunk (for offset calculation).
    bytes_consumed: usize,
    /// Whether the chunk reader has been exhausted.
    eof: bool,
}

impl StandardAnalyzer {
    /// Creates a new analyzer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Loads the next chunk from the reader, ASCII-lowercases it, and resets
    /// the scan position.
    fn load_next_chunk(&mut self) -> io::Result<()> {
        self.bytes_consumed += self.current.len();
        if let Some(reader) = &mut self.chunk_reader {
            match reader.next_chunk()? {
                Some(mut chunk) => {
                    chunk.make_ascii_lowercase();
                    self.current = chunk;
                    self.pos = 0;
                }
                None => {
                    self.current.clear();
                    self.pos = 0;
                    self.eof = true;
                }
            }
        } else {
            self.eof = true;
        }
        Ok(())
    }

    /// Creates an analyzer with a custom chunk capacity (for testing).
    #[cfg(test)]
    fn with_capacity(capacity: usize, reader: Box<dyn Read + Send>) -> Self {
        Self {
            chunk_reader: Some(Utf8ChunkReader::with_capacity(capacity, reader)),
            ..Self::default()
        }
    }
}

impl Analyzer for StandardAnalyzer {
    fn set_reader(&mut self, reader: Box<dyn Read + Send>) {
        self.chunk_reader = Some(Utf8ChunkReader::new(reader));
        self.current.clear();
        self.pos = 0;
        self.boundary_buf.clear();
        self.bytes_consumed = 0;
        self.eof = false;
    }

    fn next_token(&mut self) -> io::Result<Option<Token<'_>>> {
        // --- Phase 1: Skip non-word characters ---
        'skip: loop {
            let bytes = self.current.as_bytes();
            while self.pos < bytes.len() {
                let b = bytes[self.pos];
                if b < 0x80 {
                    if StandardTokenizer::is_word_char(b as char) {
                        break 'skip;
                    }
                    self.pos += 1;
                } else {
                    let ch = self.current[self.pos..].chars().next().unwrap();
                    if StandardTokenizer::is_word_char(ch) {
                        break 'skip;
                    }
                    self.pos += ch.len_utf8();
                }
            }
            // Exhausted chunk.
            if self.eof {
                return Ok(None);
            }
            self.load_next_chunk()?;
        }

        let token_start_byte = self.bytes_consumed + self.pos;
        let scan_start = self.pos;
        let mut char_count: usize = 0;
        let mut spanning = false;

        // --- Phase 2: Scan word chars + internal separators ---
        // Pending separator: when we hit a separator that might cross a chunk
        // boundary, we break out of the inner loop, handle the boundary, then
        // continue. This avoids holding a `bytes` borrow across load_next_chunk.
        let mut pending_sep: Option<char> = None;

        'token: loop {
            // Handle a separator that was deferred from the previous iteration.
            if let Some(sep) = pending_sep.take() {
                let sep_len = sep.len_utf8();
                // We already advanced pos past the separator and possibly loaded
                // a new chunk. Now check what follows.
                if self.pos >= self.current.len() {
                    // EOF after separator — exclude it.
                    if !spanning {
                        // pos is past sep in now-empty chunk; we can't back up.
                        // But we saved to boundary_buf before breaking, so just
                        // emit from boundary_buf.
                        _ = sep_len;
                    }
                    break 'token;
                }
                let next_ch = self.current[self.pos..].chars().next().unwrap();
                if next_ch.is_alphanumeric() {
                    if spanning {
                        self.boundary_buf.push(sep);
                        self.boundary_buf.push(next_ch);
                    }
                    self.pos += next_ch.len_utf8();
                    char_count += 2;
                } else {
                    // Trailing separator — exclude it. If spanning, boundary_buf
                    // doesn't have the sep. If not spanning, pos is already past
                    // the sep but that's fine — we just won't include it.
                    if !spanning {
                        self.pos -= sep_len;
                    }
                    break 'token;
                }
            }

            // Tight inner loop: scan within current chunk.
            let bytes = self.current.as_bytes();
            while self.pos < bytes.len() && char_count < MAX_TOKEN_LENGTH {
                let b = bytes[self.pos];
                let ch = if b < 0x80 {
                    b as char
                } else {
                    self.current[self.pos..].chars().next().unwrap()
                };

                if StandardTokenizer::is_word_char(ch) {
                    if spanning {
                        self.boundary_buf.push(ch);
                    }
                    self.pos += ch.len_utf8();
                    char_count += 1;
                } else if StandardTokenizer::is_internal_separator(ch) {
                    let sep_len = ch.len_utf8();

                    // Will advancing past the separator leave the chunk?
                    if self.pos + sep_len >= bytes.len() && !self.eof {
                        // Save token text before crossing.
                        if !spanning {
                            self.boundary_buf.clear();
                            self.boundary_buf
                                .push_str(&self.current[scan_start..self.pos]);
                            spanning = true;
                        }
                        self.pos += sep_len;
                        // Need to load next chunk — break out of inner loop to
                        // avoid holding `bytes` borrow.
                        pending_sep = Some(ch);
                        break;
                    }

                    // Separator and next char both in this chunk.
                    if self.pos + sep_len < bytes.len() {
                        let next_ch = self.current[self.pos + sep_len..].chars().next().unwrap();
                        if next_ch.is_alphanumeric() {
                            if spanning {
                                self.boundary_buf.push(ch);
                                self.boundary_buf.push(next_ch);
                            }
                            self.pos += sep_len + next_ch.len_utf8();
                            char_count += 2;
                        } else {
                            break 'token;
                        }
                    } else {
                        // Separator is last byte(s) of chunk and eof is true.
                        break 'token;
                    }
                } else {
                    break 'token;
                }
            }

            // Exited inner loop: end of chunk, MAX_TOKEN_LENGTH, or pending_sep.
            if char_count >= MAX_TOKEN_LENGTH {
                break 'token;
            }
            if pending_sep.is_some() {
                // Load next chunk for the pending separator.
                self.load_next_chunk()?;
                continue 'token;
            }
            if self.eof {
                break 'token;
            }
            // End of chunk mid-token — save and continue in next chunk.
            if !spanning {
                self.boundary_buf.clear();
                self.boundary_buf
                    .push_str(&self.current[scan_start..self.pos]);
                spanning = true;
            }
            self.load_next_chunk()?;
        }

        // --- Phase 3: Handle MAX_TOKEN_LENGTH overflow ---
        if char_count >= MAX_TOKEN_LENGTH {
            if !spanning {
                self.boundary_buf.clear();
                self.boundary_buf
                    .push_str(&self.current[scan_start..self.pos]);
                spanning = true;
            }
            // Skip remaining word chars / separators.
            'skip_overflow: loop {
                let bytes = self.current.as_bytes();
                while self.pos < bytes.len() {
                    let ch = if bytes[self.pos] < 0x80 {
                        bytes[self.pos] as char
                    } else {
                        self.current[self.pos..].chars().next().unwrap()
                    };
                    if StandardTokenizer::is_word_char(ch)
                        || StandardTokenizer::is_internal_separator(ch)
                    {
                        self.pos += ch.len_utf8();
                    } else {
                        break 'skip_overflow;
                    }
                }
                if self.eof {
                    break;
                }
                self.load_next_chunk()?;
            }
        }

        // --- Phase 4: Emit token ---
        if spanning {
            Ok(Some(Token {
                text: &self.boundary_buf,
                offset: TermOffset {
                    start: token_start_byte as u32,
                    length: self.boundary_buf.len() as u16,
                },
                position_increment: 1,
            }))
        } else {
            let token_len = (self.bytes_consumed + self.pos) - token_start_byte;
            Ok(Some(Token {
                text: &self.current[scan_start..self.pos],
                offset: TermOffset {
                    start: token_start_byte as u32,
                    length: token_len as u16,
                },
                position_increment: 1,
            }))
        }
    }
}

/// Factory that creates [`StandardAnalyzer`] instances.
///
/// This is the default analyzer factory used by
/// [`IndexWriterConfig`](crate::index::config::IndexWriterConfig).
#[derive(Debug, Clone, Copy)]
pub struct StandardAnalyzerFactory;

impl AnalyzerFactory for StandardAnalyzerFactory {
    fn create(&self) -> Box<dyn Analyzer> {
        Box::new(StandardAnalyzer::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assertables::*;

    // --- Tokenizer tests (exercise the real StandardAnalyzer) ---

    /// Helper: tokenize via the real StandardAnalyzer, return (text, offset).
    fn tokenize(text: &str) -> Vec<(String, TermOffset)> {
        collect_tokens(text)
            .into_iter()
            .map(|(t, offset, _)| (t, offset))
            .collect()
    }

    #[test]
    fn test_standard_tokenizer_simple() {
        let tokens = tokenize("hello world");
        assert_len_eq_x!(&tokens, 2);
        assert_eq!(
            tokens[0],
            (
                "hello".to_string(),
                TermOffset {
                    start: 0,
                    length: 5
                }
            )
        );
        assert_eq!(
            tokens[1],
            (
                "world".to_string(),
                TermOffset {
                    start: 6,
                    length: 5
                }
            )
        );
    }

    #[test]
    fn test_standard_tokenizer_contraction() {
        let tokens = tokenize("don't stop");
        assert_len_eq_x!(&tokens, 2);
        assert_eq!(tokens[0].0, "don't");
        assert_eq!(tokens[1].0, "stop");
    }

    #[test]
    fn test_standard_tokenizer_numbers() {
        let tokens = tokenize("test123 456");
        assert_len_eq_x!(&tokens, 2);
        assert_eq!(tokens[0].0, "test123");
        assert_eq!(tokens[1].0, "456");
    }

    #[test]
    fn test_standard_tokenizer_punctuation() {
        let tokens = tokenize("hello, world! foo.");
        assert_len_eq_x!(&tokens, 3);
        assert_eq!(tokens[0].0, "hello");
        assert_eq!(tokens[1].0, "world");
        assert_eq!(tokens[2].0, "foo");
    }

    #[test]
    fn test_standard_tokenizer_empty() {
        let tokens = tokenize("");
        assert_is_empty!(tokens);
    }

    #[test]
    fn test_apostrophe_at_end_of_input() {
        let tokens = tokenize("don't");
        assert_len_eq_x!(&tokens, 1);
        assert_eq!(tokens[0].0, "don't");

        let tokens = tokenize("hello'");
        assert_len_eq_x!(&tokens, 1);
        assert_eq!(tokens[0].0, "hello");
    }

    #[test]
    fn test_apostrophe_followed_by_non_alpha() {
        let tokens = tokenize("it' s");
        assert_len_eq_x!(&tokens, 2);
        assert_eq!(tokens[0].0, "it");
        assert_eq!(tokens[1].0, "s");
    }

    #[test]
    fn test_token_exceeding_max_length() {
        let long_word: String = "a".repeat(255);
        let tokens = tokenize(&long_word);
        assert_len_eq_x!(&tokens, 1);
        assert_len_eq_x!(&tokens[0].0, 255);

        let too_long: String = "b".repeat(300);
        let tokens = tokenize(&too_long);
        assert_len_eq_x!(&tokens, 1);
        assert_len_eq_x!(&tokens[0].0, 255);

        let input = format!("{} short", "c".repeat(300));
        let tokens = tokenize(&input);
        assert_len_eq_x!(&tokens, 2);
        assert_len_eq_x!(&tokens[0].0, 255);
        assert_eq!(tokens[1].0, "short");
    }

    // --- StandardAnalyzer (pull-based) tests ---

    fn collect_tokens(text: &str) -> Vec<(String, TermOffset, i32)> {
        let mut analyzer = StandardAnalyzer::default();
        analyzer.set_reader(Box::new(io::Cursor::new(text.as_bytes().to_vec())));
        let mut result = Vec::new();

        while let Some(token) = analyzer.next_token().unwrap() {
            result.push((
                token.text.to_string(),
                token.offset,
                token.position_increment,
            ));
        }
        result
    }

    #[test]
    fn test_standard_analyzer() {
        let tokens = collect_tokens("The quick brown fox");
        let texts: Vec<&str> = tokens.iter().map(|t| t.0.as_str()).collect();
        assert_eq!(texts, vec!["the", "quick", "brown", "fox"]);
    }

    #[test]
    fn test_standard_analyzer_no_stop_words() {
        let tokens = collect_tokens("the quick and brown fox");
        assert_len_eq_x!(&tokens, 5);
        let texts: Vec<&str> = tokens.iter().map(|t| t.0.as_str()).collect();
        assert_eq!(texts, vec!["the", "quick", "and", "brown", "fox"]);
        for t in &tokens {
            assert_eq!(t.2, 1);
        }
    }

    #[test]
    fn test_lowercases_tokens() {
        let tokens = collect_tokens("Hello WORLD");
        assert_eq!(tokens[0].0, "hello");
        assert_eq!(tokens[1].0, "world");
    }

    #[test]
    fn test_empty_string_produces_no_tokens() {
        let tokens = collect_tokens("");
        assert_is_empty!(&tokens);
    }

    #[test]
    fn test_preserves_contractions() {
        let tokens = collect_tokens("don't");
        assert_len_eq_x!(&tokens, 1);
        assert_eq!(tokens[0].0, "don't");
    }

    #[test]
    fn test_offsets_are_correct() {
        let tokens = collect_tokens("hello world");
        assert_eq!(
            tokens[0].1,
            TermOffset {
                start: 0,
                length: 5
            }
        );
        assert_eq!(
            tokens[1].1,
            TermOffset {
                start: 6,
                length: 5
            }
        );
    }

    #[test]
    fn test_position_increments_are_one() {
        let tokens = collect_tokens("one two three");
        for t in &tokens {
            assert_eq!(t.2, 1);
        }
    }

    #[test]
    fn test_set_reader_allows_reuse() {
        let mut analyzer = StandardAnalyzer::default();

        // First field
        analyzer.set_reader(Box::new(io::Cursor::new(b"hello".to_vec())));
        let token = analyzer.next_token().unwrap();
        assert_some!(&token);
        let none = analyzer.next_token().unwrap();
        assert_none!(&none);

        // Set new reader and process second field
        analyzer.set_reader(Box::new(io::Cursor::new(b"world".to_vec())));
        let token = analyzer.next_token().unwrap();
        assert_some!(&token);
        assert_eq!(token.unwrap().text, "world");
    }

    // --- Chunk boundary tests (small capacity to force boundaries) ---

    fn collect_tokens_chunked(text: &str, capacity: usize) -> Vec<(String, TermOffset, i32)> {
        let reader: Box<dyn Read + Send> = Box::new(io::Cursor::new(text.as_bytes().to_vec()));
        let mut analyzer = StandardAnalyzer::with_capacity(capacity, reader);
        let mut result = Vec::new();
        while let Some(token) = analyzer.next_token().unwrap() {
            result.push((
                token.text.to_string(),
                token.offset,
                token.position_increment,
            ));
        }
        result
    }

    #[test]
    fn test_token_spanning_chunk_boundary() {
        let tokens = collect_tokens_chunked("hello world", 4);
        assert_len_eq_x!(&tokens, 2);
        assert_eq!(tokens[0].0, "hello");
        assert_eq!(tokens[1].0, "world");
    }

    #[test]
    fn test_contraction_spanning_boundary() {
        let tokens = collect_tokens_chunked("don't stop", 4);
        assert_len_eq_x!(&tokens, 2);
        assert_eq!(tokens[0].0, "don't");
        assert_eq!(tokens[1].0, "stop");
    }

    #[test]
    fn test_separator_at_exact_boundary() {
        // "ab'" is 3 bytes. With capacity 3, the apostrophe is last byte of chunk.
        // Next chunk starts with "cd" — should produce "ab'cd".
        let tokens = collect_tokens_chunked("ab'cd", 3);
        assert_len_eq_x!(&tokens, 1);
        assert_eq!(tokens[0].0, "ab'cd");
    }

    #[test]
    fn test_separator_at_boundary_followed_by_non_alpha() {
        let tokens = collect_tokens_chunked("ab' x", 3);
        assert_len_eq_x!(&tokens, 2);
        assert_eq!(tokens[0].0, "ab");
        assert_eq!(tokens[1].0, "x");
    }

    #[test]
    fn test_comprehensive_tiny_chunks_match_default() {
        let input = "The quick brown fox don't jump over the lazy dog's bed";
        let default_tokens = collect_tokens(input);
        let default_texts: Vec<&str> = default_tokens.iter().map(|t| t.0.as_str()).collect();
        let chunked_tokens = collect_tokens_chunked(input, 4);
        let chunked_texts: Vec<&str> = chunked_tokens.iter().map(|t| t.0.as_str()).collect();
        assert_eq!(chunked_texts, default_texts);
    }

    #[test]
    fn test_token_at_eof_no_trailing_whitespace() {
        let tokens = collect_tokens_chunked("hello", 3);
        assert_len_eq_x!(&tokens, 1);
        assert_eq!(tokens[0].0, "hello");
    }

    #[test]
    fn test_empty_input_chunked() {
        let tokens = collect_tokens_chunked("", 4);
        assert_is_empty!(&tokens);
    }

    #[test]
    fn test_set_reader_reuse_with_streaming() {
        let reader1: Box<dyn Read + Send> = Box::new(io::Cursor::new(b"hello".to_vec()));
        let mut analyzer = StandardAnalyzer::with_capacity(3, reader1);
        let token = analyzer.next_token().unwrap();
        assert_eq!(token.unwrap().text, "hello");
        let none = analyzer.next_token().unwrap();
        assert_none!(&none);
        analyzer.set_reader(Box::new(io::Cursor::new(b"world".to_vec())));
        let token = analyzer.next_token().unwrap();
        assert_eq!(token.unwrap().text, "world");
    }

    #[test]
    fn test_offsets_correct_across_chunks() {
        let tokens = collect_tokens_chunked("hello world", 4);
        assert_eq!(
            tokens[0].1,
            TermOffset {
                start: 0,
                length: 5
            }
        );
        assert_eq!(
            tokens[1].1,
            TermOffset {
                start: 6,
                length: 5
            }
        );
    }

    #[test]
    fn test_many_tokens_tiny_chunks() {
        let input = "a b c d e f g h i j";
        let tokens = collect_tokens_chunked(input, 3);
        let texts: Vec<&str> = tokens.iter().map(|t| t.0.as_str()).collect();
        assert_eq!(
            texts,
            vec!["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
        );
    }

    // --- MAX_TOKEN_LENGTH edge cases ---

    #[test]
    fn test_max_token_length_exact() {
        let input: String = "a".repeat(255);
        let tokens = collect_tokens(&input);
        assert_len_eq_x!(&tokens, 1);
        assert_len_eq_x!(&tokens[0].0, 255);
    }

    #[test]
    fn test_max_token_length_exceeded() {
        let input: String = "b".repeat(300);
        let tokens = collect_tokens(&input);
        assert_len_eq_x!(&tokens, 1);
        assert_len_eq_x!(&tokens[0].0, 255);
    }

    #[test]
    fn test_max_token_length_with_following_token() {
        let input = format!("{} short", "c".repeat(300));
        let tokens = collect_tokens(&input);
        assert_len_eq_x!(&tokens, 2);
        assert_len_eq_x!(&tokens[0].0, 255);
        assert_eq!(tokens[1].0, "short");
    }

    // --- Smart quote (U+2019) edge cases ---

    #[test]
    fn test_smart_quote_contraction_at_boundary() {
        // U+2019 is 3 bytes. With capacity 4, chunk boundary falls inside
        // the multi-byte sequence — utf8-zero handles this by not splitting
        // the codepoint, but the token must still be preserved.
        let input = "don\u{2019}t";
        let tokens = collect_tokens_chunked(input, 4);
        assert_len_eq_x!(&tokens, 1);
        assert_eq!(tokens[0].0, "don\u{2019}t");
    }
}
