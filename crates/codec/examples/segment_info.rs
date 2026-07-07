// SPDX-License-Identifier: Apache-2.0

//! P0 "Done" criterion (SPEC §13): print a real segment's field inventory.
//!
//! ```text
//! cargo run -p lucene-arrow-codec --example segment_info -- /path/to/index
//! ```

use lucene_arrow_codec::SegmentDirectory;

fn flags(field: &lucene_arrow_codec::FieldMeta) -> String {
    let mut out = Vec::new();
    if field.indexed {
        out.push("indexed");
    }
    if field.has_norms {
        out.push("norms");
    }
    if field.has_points {
        out.push("points");
    }
    if field.has_vectors {
        out.push("vectors");
    }
    out.join(",")
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: segment_info <lucene-segment-directory>");
        std::process::exit(2);
    });

    let dir = match SegmentDirectory::open(&path) {
        Ok(dir) => dir,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

    println!(
        "{path}: generation {} — {} segment(s)",
        dir.generation(),
        dir.segments().len()
    );

    for seg in dir.segments() {
        println!();
        println!(
            "segment {} (ord {})  codec={}  max_doc={}  del_count={}  compound={}",
            seg.name, seg.ord, seg.codec, seg.max_doc, seg.del_count, seg.is_compound
        );
        println!("  id: {}", hex(&seg.id));
        println!("  files ({}):", seg.files.len());
        for f in &seg.files {
            println!("    {f}");
        }
        println!("  fields ({}):", seg.fields.len());
        println!(
            "    {:>4}  {:<24} {:<15} flags",
            "#", "name", "doc_values"
        );
        for field in &seg.fields {
            println!(
                "    {:>4}  {:<24} {:<15} {}",
                field.number,
                field.name,
                field.doc_values.as_str(),
                flags(field)
            );
            for (k, v) in &field.attributes {
                println!("          {k} = {v}");
            }
        }
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
