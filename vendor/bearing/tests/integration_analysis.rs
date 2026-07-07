//! Integration tests for the bearing analysis public API.
//!
//! Tests the StandardAnalyzer pull-based tokenization interface.

#[macro_use]
extern crate assertables;

use std::io::Cursor;

use bearing::analysis::{Analyzer, StandardAnalyzer};
use bearing::document::TermOffset;

/// Collects all tokens from an analyzer into (text, offset, pos_inc) tuples.
fn collect_tokens(text: &str) -> Vec<(String, TermOffset, i32)> {
    let mut analyzer = StandardAnalyzer::new();
    analyzer.set_reader(Box::new(Cursor::new(text.as_bytes().to_vec())));
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

/// Collects just the token texts.
fn collect_texts(text: &str) -> Vec<String> {
    collect_tokens(text).into_iter().map(|t| t.0).collect()
}

// ---------------------------------------------------------------------------
// StandardAnalyzer tokenization
// ---------------------------------------------------------------------------

#[test]
fn standard_analyzer_basic_tokenization() {
    let texts = collect_texts("the quick brown fox");
    assert_eq!(texts, vec!["the", "quick", "brown", "fox"]);
}

#[test]
fn standard_analyzer_lowercases() {
    let texts = collect_texts("Hello WORLD FoO");
    assert_eq!(texts, vec!["hello", "world", "foo"]);
}

#[test]
fn standard_analyzer_empty_input() {
    let tokens = collect_tokens("");
    assert_is_empty!(tokens);
}

#[test]
fn standard_analyzer_whitespace_only() {
    let tokens = collect_tokens("   \t\n  ");
    assert_is_empty!(tokens);
}

#[test]
fn standard_analyzer_punctuation_splitting() {
    let texts = collect_texts("hello, world! how are you?");
    assert_eq!(texts, vec!["hello", "world", "how", "are", "you"]);
}

#[test]
fn standard_analyzer_offsets() {
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
fn standard_analyzer_position_increments() {
    let tokens = collect_tokens("one two three");
    for token in &tokens {
        assert_eq!(token.2, 1);
    }
}

#[test]
fn standard_analyzer_set_reader_allows_reuse() {
    let mut analyzer = StandardAnalyzer::new();

    // First field
    analyzer.set_reader(Box::new(Cursor::new(b"hello".to_vec())));
    let token = analyzer.next_token().unwrap();
    assert_some!(&token);
    let none = analyzer.next_token().unwrap();
    assert_none!(&none);

    // Set new reader and process second field
    analyzer.set_reader(Box::new(Cursor::new(b"world".to_vec())));
    let token = analyzer.next_token().unwrap();
    assert_some!(&token);
    assert_eq!(token.unwrap().text, "world");
}

// ---------------------------------------------------------------------------
// Unicode handling
// ---------------------------------------------------------------------------

#[test]
fn unicode_basic_latin_extended() {
    let texts = collect_texts("café résumé naïve");
    assert_len_eq_x!(&texts, 3);
    assert_eq!(texts[0], "café");
    assert_eq!(texts[1], "résumé");
    assert_eq!(texts[2], "naïve");
}

#[test]
fn unicode_cjk_characters() {
    let texts = collect_texts("hello 世界");
    assert_not_empty!(texts);
    assert_eq!(texts[0], "hello");
}

#[test]
fn unicode_emoji() {
    // Should not panic on emoji input
    let _tokens = collect_tokens("hello 🌍 world");
}

#[test]
fn single_character_tokens() {
    let texts = collect_texts("a b c");
    assert_eq!(texts, vec!["a", "b", "c"]);
}

#[test]
fn numeric_text_tokenization() {
    let texts = collect_texts("version 3 release 42");
    assert_contains!(texts, &"version".to_string());
    assert_contains!(texts, &"release".to_string());
    assert_contains!(texts, &"42".to_string());
}
