// SPDX-License-Identifier: Apache-2.0

//! Write a small real Lucene103 segment (via Bearing) to poke at with
//! `segment_info`. Usage: `cargo run -p lucene-arrow-codec --example
//! make_demo_segment -- /tmp/demo-segment`

use bearing::prelude::{
    DocumentBuilder, FSDirectory, IndexWriter, IndexWriterConfig, binary_dv, keyword, numeric_dv,
    sorted_dv, sorted_numeric_dv,
};

fn main() {
    let path = std::env::args().nth(1).expect("usage: make_demo_segment <output-dir>");
    std::fs::create_dir_all(&path).expect("create output dir");

    let directory = FSDirectory::open(std::path::Path::new(&path)).expect("open directory");
    let config = IndexWriterConfig::default().num_threads(1).use_compound_file(false);
    let writer = IndexWriter::new(config, directory);

    for i in 0..1000i64 {
        let mut doc = DocumentBuilder::new()
            .add_field(numeric_dv("price").value(1000 + i * 25))
            .add_field(sorted_dv("category").value(format!("cat-{}", i % 7).into_bytes()))
            .add_field(sorted_numeric_dv("scores").value(vec![i, i * 2]))
            .add_field(binary_dv("payload").value(vec![i as u8, 0xAB]))
            .add_field(keyword("tag").value(format!("tag-{}", i % 3)));
        if i % 6 == 1 {
            doc = doc.add_field(numeric_dv("rare").value(i * -13));
        }
        writer.add_document(doc.build()).expect("add document");
    }
    writer.commit().expect("commit");
    println!("wrote 1000-doc Lucene103 segment to {path}");
}
