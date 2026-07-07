# Bearing

[![Crates.io](https://img.shields.io/crates/v/bearing)](https://crates.io/crates/bearing)
[![Docs](https://img.shields.io/docsrs/bearing)](https://docs.rs/bearing/)
[![CI](https://github.com/toddfeak/bearing/actions/workflows/ci.yml/badge.svg)](https://github.com/toddfeak/bearing/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![MSRV](https://img.shields.io/badge/MSRV-1.88-orange.svg)](https://blog.rust-lang.org/)

A Rust port of Apache Lucene - the full-text search library.

## Status

Bearing is in **alpha**. The API will change.

- **Indexing**: Multi-threaded `IndexWriter` with fourteen field types, five doc values types, term vectors, sparse fields, and postings with positions and offsets
- **Querying**: TermQuery and BooleanQuery with BM25 scoring — arbitrary numbers of MUST, SHOULD, and MUST_NOT clauses, including `minShouldMatch`
- **Codec**: Lucene103 — indexes are readable by Java Lucene and vice versa
- **Correctness**: Query results are cross-validated against Java Lucene across multiple corpus sizes

## API Documentation

Full API documentation is available on [docs.rs](https://docs.rs/bearing/).

## About

Bearing is a port of [Apache Lucene 10.3.2](https://lucene.apache.org/) to Rust. It writes and reads Lucene-compatible indexes using the Lucene103 codec, producing byte-identical results to the Java implementation.

This project exists at the intersection of learning and building. It's an exercise in learning Rust and exploring AI-assisted development with Claude Code, while producing something that might actually be useful. The code is AI-generated and a work in progress — it won't be perfect, and it moves at its own pace. This is a personal project: no pull requests, no issue tracker, no collaborators. If it interests you, fork it and make it your own.

## Quick Start

### Indexing

```rust
use bearing::prelude::{
    DocumentBuilder, FSDirectory, IndexWriter, IndexWriterConfig,
    keyword, text,
};

let directory = FSDirectory::open(std::path::Path::new("/tmp/my-index")).unwrap();
let writer = IndexWriter::new(IndexWriterConfig::default(), directory);

let doc = DocumentBuilder::new()
    .add_field(text("body").value("the quick brown fox"))
    .add_field(keyword("category").value("animals"))
    .build();
writer.add_document(doc).unwrap();

writer.commit().unwrap();
```

### Searching

```rust
use bearing::prelude::{DirectoryReader, FSDirectory, IndexSearcher, TermQuery};

let directory = FSDirectory::open(std::path::Path::new("/tmp/my-index")).unwrap();
let reader = DirectoryReader::open(&*directory).unwrap();
let searcher = IndexSearcher::new(&reader);

let query = TermQuery::new("body", b"fox");
let top_docs = searcher.search(&query, 10).unwrap();

for hit in &top_docs.score_docs {
    println!("doc={} score={}", hit.doc, hit.score);
}
```

See the [`prelude`](https://docs.rs/bearing/latest/bearing/prelude/) module for all available types.

## Why "Bearing"?

The name is a play on words. A bearing gives direction — fitting for a search library. Bearings also carry load — as this project aims to do for indexing workloads. And of course, bearings are made of steel, which isn't too far from Rust.

## Roadmap

See [PLAN.md](docs/PLAN.md) for detailed progress and next steps.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

This project is derived from [Apache Lucene](https://lucene.apache.org/).
