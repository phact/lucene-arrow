// SPDX-License-Identifier: Apache-2.0

//! Unicode-aware analysis: [`UnicodeAnalyzer`] and [`UnicodeAnalyzerFactory`].
//!
//! Uses the `unicode-segmentation` crate for UAX#29 word boundary detection.
//! Produces correct tokens for CJK ideographs, numeric grouping (`1,200`),
//! URLs, and accented text. Slower than [`StandardAnalyzer`](super::StandardAnalyzer)
//! (~57 MB/s vs ~212 MB/s) but significantly more accurate for non-ASCII text.

use std::io::{self, Read};

use unicode_segmentation::UnicodeSegmentation;

use crate::analysis::chunk_reader::Utf8ChunkReader;
use crate::analysis::{Analyzer, AnalyzerFactory, Token};
use crate::document::TermOffset;

/// Unicode-aware text analyzer using UAX#29 word boundary rules.
///
/// Uses `unicode-segmentation` for word boundary detection with full Unicode
/// lowercase normalization. Matches Java Lucene's `StandardAnalyzer` output
/// for all token categories except emoji sequences.
///
/// For most English-only workloads, [`StandardAnalyzer`](super::StandardAnalyzer)
/// is faster and produces identical results. Use `UnicodeAnalyzer` when
/// correctness for CJK, numeric grouping, or URLs matters.
#[derive(Default)]
pub struct UnicodeAnalyzer {
    chunk_reader: Option<Utf8ChunkReader>,
    /// Accumulated text for segmentation. Contains lowercased content from
    /// one or more chunks. After segmenting, the trailing unsegmented portion
    /// (which might be an incomplete token) is retained for the next fill.
    buf: String,
    /// Word boundary segments computed from `buf`.
    segments: Vec<(usize, usize)>,
    /// Index into `segments` for the next token to return.
    seg_idx: usize,
    /// Total bytes consumed before `buf` (for offset calculation).
    bytes_consumed: usize,
    /// Whether the chunk reader has been exhausted.
    eof: bool,
}

impl UnicodeAnalyzer {
    /// Creates a new analyzer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Reads the next chunk into `buf`, lowercases it, and computes segments.
    /// If `buf` has a trailing fragment from a previous chunk, it is preserved
    /// and the new chunk is appended. Segments are computed on the combined
    /// buffer, but the last segment is withheld if it touches the end (it might
    /// be incomplete).
    fn fill_and_segment(&mut self) -> io::Result<()> {
        // Read next chunk and append to buf.
        if let Some(reader) = &mut self.chunk_reader {
            match reader.next_chunk()? {
                Some(chunk) => {
                    let lowered = chunk.to_lowercase();
                    self.buf.push_str(&lowered);
                }
                None => {
                    self.eof = true;
                }
            }
        } else {
            self.eof = true;
        }

        // Compute word boundary segments on the full buffer.
        self.segments.clear();
        self.seg_idx = 0;
        let mut offset = 0;
        for segment in self.buf.split_word_bounds() {
            let len = segment.len();
            if segment.chars().any(|c| c.is_alphanumeric()) {
                self.segments.push((offset, offset + len));
            }
            offset += len;
        }

        // If not at EOF, withhold the last word segment — it might be
        // incomplete at the chunk boundary. The trailing portion of buf
        // (from the last segment's start onward) is kept for the next fill.
        if !self.eof && !self.segments.is_empty() {
            self.segments.pop();
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

impl Analyzer for UnicodeAnalyzer {
    fn set_reader(&mut self, reader: Box<dyn Read + Send>) {
        self.chunk_reader = Some(Utf8ChunkReader::new(reader));
        self.buf.clear();
        self.segments.clear();
        self.seg_idx = 0;
        self.bytes_consumed = 0;
        self.eof = false;
    }

    fn next_token(&mut self) -> io::Result<Option<Token<'_>>> {
        loop {
            if self.seg_idx < self.segments.len() {
                let (start, end) = self.segments[self.seg_idx];
                self.seg_idx += 1;
                let token_start_byte = self.bytes_consumed + start;
                let token_len = (self.bytes_consumed + end) - token_start_byte;
                return Ok(Some(Token {
                    text: &self.buf[start..end],
                    offset: TermOffset {
                        start: token_start_byte as u32,
                        length: token_len as u16,
                    },
                    position_increment: 1,
                }));
            }

            if self.eof {
                return Ok(None);
            }

            // All safe segments consumed. Trim buf: keep only the trailing
            // portion that might be an incomplete token.
            let trim_to = if let Some(&(_, end)) = self.segments.last() {
                end
            } else if !self.buf.is_empty() {
                // No safe segments — entire buf is potentially incomplete.
                0
            } else {
                self.buf.len()
            };

            // The content we withheld might include non-word separators between
            // the last safe segment and the withheld word segment. We need to
            // keep all of it for correct re-segmentation.
            self.bytes_consumed += trim_to;
            self.buf = self.buf[trim_to..].to_string();
            self.segments.clear();
            self.seg_idx = 0;

            self.fill_and_segment()?;
        }
    }
}

/// Factory that creates [`UnicodeAnalyzer`] instances.
#[derive(Debug, Clone, Copy)]
pub struct UnicodeAnalyzerFactory;

impl AnalyzerFactory for UnicodeAnalyzerFactory {
    fn create(&self) -> Box<dyn Analyzer> {
        Box::new(UnicodeAnalyzer::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assertables::*;

    fn collect_tokens(text: &str) -> Vec<(String, TermOffset, i32)> {
        let mut analyzer = UnicodeAnalyzer::default();
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

    fn collect_tokens_chunked(text: &str, capacity: usize) -> Vec<(String, TermOffset, i32)> {
        let reader: Box<dyn Read + Send> = Box::new(io::Cursor::new(text.as_bytes().to_vec()));
        let mut analyzer = UnicodeAnalyzer::with_capacity(capacity, reader);
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

    fn texts(tokens: &[(String, TermOffset, i32)]) -> Vec<&str> {
        tokens.iter().map(|t| t.0.as_str()).collect()
    }

    // --- Basic tokenization (same as StandardAnalyzer tests) ---

    #[test]
    fn test_simple_english() {
        let tokens = collect_tokens("The quick brown fox");
        assert_eq!(texts(&tokens), vec!["the", "quick", "brown", "fox"]);
    }

    #[test]
    fn test_lowercases_tokens() {
        let tokens = collect_tokens("Hello WORLD");
        assert_eq!(tokens[0].0, "hello");
        assert_eq!(tokens[1].0, "world");
    }

    #[test]
    fn test_empty_string() {
        let tokens = collect_tokens("");
        assert_is_empty!(&tokens);
    }

    #[test]
    fn test_position_increments() {
        let tokens = collect_tokens("one two three");
        for t in &tokens {
            assert_eq!(t.2, 1);
        }
    }

    #[test]
    fn test_set_reader_reuse() {
        let mut analyzer = UnicodeAnalyzer::default();
        analyzer.set_reader(Box::new(io::Cursor::new(b"hello".to_vec())));
        let token = analyzer.next_token().unwrap();
        assert_eq!(token.unwrap().text, "hello");
        let none = analyzer.next_token().unwrap();
        assert_none!(&none);

        analyzer.set_reader(Box::new(io::Cursor::new(b"world".to_vec())));
        let token = analyzer.next_token().unwrap();
        assert_eq!(token.unwrap().text, "world");
    }

    // --- UAX#29 token categories (validated against Java Lucene StandardAnalyzer) ---

    #[test]
    fn test_numeric_grouping() {
        let tokens = collect_tokens("1,200");
        assert_len_eq_x!(&tokens, 1);
        assert_eq!(tokens[0].0, "1,200");
    }

    #[test]
    fn test_decimal_numbers() {
        let tokens = collect_tokens("12.1 3.14159");
        assert_len_eq_x!(&tokens, 2);
        assert_eq!(tokens[0].0, "12.1");
        assert_eq!(tokens[1].0, "3.14159");
    }

    #[test]
    fn test_urls() {
        let tokens = collect_tokens("www.gutenberg.org www.pgdp.net");
        assert_len_eq_x!(&tokens, 2);
        assert_eq!(tokens[0].0, "www.gutenberg.org");
        assert_eq!(tokens[1].0, "www.pgdp.net");
    }

    #[test]
    fn test_email_domain() {
        // Java splits on @ but keeps the domain with periods
        let tokens = collect_tokens("user@example.com");
        let t = texts(&tokens);
        assert_eq!(t, vec!["user", "example.com"]);
    }

    #[test]
    fn test_contractions() {
        let tokens = collect_tokens("don't can't it's they're");
        assert_eq!(texts(&tokens), vec!["don't", "can't", "it's", "they're"]);
    }

    #[test]
    fn test_possessives() {
        let tokens = collect_tokens("Todd's dog's");
        assert_eq!(texts(&tokens), vec!["todd's", "dog's"]);
    }

    #[test]
    fn test_cjk_ideographs_split() {
        // Each CJK ideograph is a separate token (matches Java)
        let tokens = collect_tokens("宋史");
        assert_len_eq_x!(&tokens, 2);
        assert_eq!(tokens[0].0, "宋");
        assert_eq!(tokens[1].0, "史");
    }

    #[test]
    fn test_cjk_multiple_sequences() {
        let tokens = collect_tokens("中文测试 東京");
        assert_eq!(texts(&tokens), vec!["中", "文", "测", "试", "東", "京"]);
    }

    #[test]
    fn test_hiragana_split() {
        // Each hiragana character is a separate token (matches Java)
        let tokens = collect_tokens("ひらがな");
        assert_eq!(texts(&tokens), vec!["ひ", "ら", "が", "な"]);
    }

    #[test]
    fn test_katakana_kept() {
        // Katakana sequences stay together (matches Java)
        let tokens = collect_tokens("カタカナ");
        assert_len_eq_x!(&tokens, 1);
        assert_eq!(tokens[0].0, "カタカナ");
    }

    #[test]
    fn test_hangul_kept() {
        // Hangul sequences stay together (matches Java)
        let tokens = collect_tokens("한국어");
        assert_len_eq_x!(&tokens, 1);
        assert_eq!(tokens[0].0, "한국어");
    }

    #[test]
    fn test_accented_text() {
        let tokens = collect_tokens("boïelle société café résumé naïve");
        assert_eq!(
            texts(&tokens),
            vec!["boïelle", "société", "café", "résumé", "naïve"]
        );
    }

    #[test]
    fn test_unicode_lowercase() {
        // Full Unicode lowercase, not just ASCII
        let tokens = collect_tokens("Société Café");
        assert_eq!(tokens[0].0, "société");
        assert_eq!(tokens[1].0, "café");
    }

    #[test]
    fn test_smart_quote_contraction() {
        let tokens = collect_tokens("don\u{2019}t");
        assert_len_eq_x!(&tokens, 1);
        assert_eq!(tokens[0].0, "don\u{2019}t");
    }

    #[test]
    fn test_hyphenated_words_split() {
        // Hyphens break tokens (matches Java)
        let tokens = collect_tokens("well-known state-of-the-art");
        assert_eq!(
            texts(&tokens),
            vec!["well", "known", "state", "of", "the", "art"]
        );
    }

    #[test]
    fn test_underscore_words() {
        let tokens = collect_tokens("foo_bar _private");
        assert_eq!(texts(&tokens), vec!["foo_bar", "_private"]);
    }

    #[test]
    fn test_mixed_alphanumeric() {
        let tokens = collect_tokens("test123 abc456def");
        assert_eq!(texts(&tokens), vec!["test123", "abc456def"]);
    }

    // --- Comprehensive Java-validated test ---

    #[test]
    fn test_full_uax29_document() {
        // Input and expected tokens validated against Java Lucene 10.3.2
        // StandardAnalyzer (no stop words). Emoji tokens excluded (known gap).
        let input = "Simple English words here.\n\
            Numbers like 1,200 and 12.1 and 3.14159 should stay together.\n\
            URLs like www.gutenberg.org and www.pgdp.net are single tokens.\n\
            Email test user@example.com should be one token.\n\
            Contractions: don't can't it's they're\n\
            Possessives: Todd's dog's\n\
            CJK ideographs: 宋史 中文测试 東京\n\
            Hiragana: ひらがな\n\
            Katakana: カタカナ\n\
            Hangul: 한국어\n\
            Accented: boïelle société d\u{2019}académie café résumé naïve\n\
            Smart quote contraction: don\u{2019}t\n\
            Mixed: test123 abc456def\n\
            Hyphenated: well-known state-of-the-art\n\
            Underscore: foo_bar _private\n";

        let expected = vec![
            "simple",
            "english",
            "words",
            "here",
            "numbers",
            "like",
            "1,200",
            "and",
            "12.1",
            "and",
            "3.14159",
            "should",
            "stay",
            "together",
            "urls",
            "like",
            "www.gutenberg.org",
            "and",
            "www.pgdp.net",
            "are",
            "single",
            "tokens",
            "email",
            "test",
            "user",
            "example.com",
            "should",
            "be",
            "one",
            "token",
            "contractions",
            "don't",
            "can't",
            "it's",
            "they're",
            "possessives",
            "todd's",
            "dog's",
            "cjk",
            "ideographs",
            "宋",
            "史",
            "中",
            "文",
            "测",
            "试",
            "東",
            "京",
            "hiragana",
            "ひ",
            "ら",
            "が",
            "な",
            "katakana",
            "カタカナ",
            "hangul",
            "한국어",
            "accented",
            "boïelle",
            "société",
            "d\u{2019}académie",
            "café",
            "résumé",
            "naïve",
            "smart",
            "quote",
            "contraction",
            "don\u{2019}t",
            "mixed",
            "test123",
            "abc456def",
            "hyphenated",
            "well",
            "known",
            "state",
            "of",
            "the",
            "art",
            "underscore",
            "foo_bar",
            "_private",
        ];

        let tokens = collect_tokens(input);
        let actual = texts(&tokens);
        assert_eq!(actual, expected);
    }

    // --- Chunk boundary tests ---

    #[test]
    fn test_token_spanning_chunk_boundary() {
        let tokens = collect_tokens_chunked("hello world", 4);
        assert_eq!(texts(&tokens), vec!["hello", "world"]);
    }

    #[test]
    fn test_contraction_spanning_boundary() {
        let tokens = collect_tokens_chunked("don't stop", 4);
        assert_eq!(texts(&tokens), vec!["don't", "stop"]);
    }

    #[test]
    fn test_numeric_grouping_spanning_boundary() {
        // "1,200" is 5 bytes. With capacity 3, the comma falls at boundary.
        let tokens = collect_tokens_chunked("1,200 test", 3);
        assert_eq!(texts(&tokens), vec!["1,200", "test"]);
    }

    #[test]
    fn test_url_spanning_boundary() {
        let tokens = collect_tokens_chunked("www.gutenberg.org test", 6);
        assert_eq!(texts(&tokens), vec!["www.gutenberg.org", "test"]);
    }

    #[test]
    fn test_cjk_spanning_boundary() {
        // Each CJK char is 3 bytes. With capacity 4, boundary falls mid-sequence.
        let tokens = collect_tokens_chunked("宋史", 4);
        assert_eq!(texts(&tokens), vec!["宋", "史"]);
    }

    #[test]
    fn test_tiny_chunks_match_default() {
        let input = "The quick brown fox don't jump over the lazy dog's bed";
        let default_tokens = collect_tokens(input);
        let chunked_tokens = collect_tokens_chunked(input, 4);
        assert_eq!(texts(&chunked_tokens), texts(&default_tokens));
    }

    #[test]
    fn test_empty_input_chunked() {
        let tokens = collect_tokens_chunked("", 4);
        assert_is_empty!(&tokens);
    }

    #[test]
    fn test_many_tokens_tiny_chunks() {
        let input = "a b c d e f g h i j";
        let tokens = collect_tokens_chunked(input, 3);
        assert_eq!(
            texts(&tokens),
            vec!["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
        );
    }
}
