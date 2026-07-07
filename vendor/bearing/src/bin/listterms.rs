// SPDX-License-Identifier: Apache-2.0

//! Dumps all terms for a field in iteration (lexicographic) order.
//!
//! Usage: `listterms -index <path> -field <name> [-segment <prefix>]`

use std::collections::BTreeSet;
use std::env;
use std::path::Path;
use std::process;

use bearing::index::directory_reader::DirectoryReader;
use bearing::index::terms::Terms;
use bearing::store::FSDirectory;

fn main() {
    let args = parse_args();

    let dir = FSDirectory::open(Path::new(&args.index_path)).unwrap_or_else(|e| {
        eprintln!("Failed to open index directory '{}': {e}", args.index_path);
        process::exit(1);
    });

    let reader = DirectoryReader::open(&dir).unwrap_or_else(|e| {
        eprintln!("Failed to open index: {e}");
        process::exit(1);
    });

    let mut all_terms = BTreeSet::new();

    for leaf in reader.leaves() {
        let seg = &leaf.reader;
        let seg_name = seg.segment_name();

        if let Some(ref prefix) = args.segment_prefix
            && !seg_name.starts_with(prefix.as_str())
        {
            continue;
        }

        let Some(terms) = seg.terms(&args.field_name) else {
            eprintln!(
                "No terms for field '{}' in segment {seg_name}",
                args.field_name
            );
            continue;
        };

        let mut te = terms.iterator().unwrap_or_else(|e| {
            eprintln!("Failed to create terms iterator for segment {seg_name}: {e}");
            process::exit(1);
        });

        while let Ok(Some(term)) = te.next() {
            all_terms.insert(term.to_vec());
        }
    }

    for term in &all_terms {
        let text = String::from_utf8_lossy(term);
        println!("{text}");
    }

    eprintln!("{} terms", all_terms.len());
}

struct Args {
    index_path: String,
    field_name: String,
    segment_prefix: Option<String>,
}

fn parse_args() -> Args {
    let args: Vec<String> = env::args().collect();
    let mut index_path = None;
    let mut field_name = None;
    let mut segment_prefix = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-index" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Missing value for -index");
                    process::exit(1);
                }
                index_path = Some(args[i].clone());
            }
            "-field" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Missing value for -field");
                    process::exit(1);
                }
                field_name = Some(args[i].clone());
            }
            "-segment" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Missing value for -segment");
                    process::exit(1);
                }
                segment_prefix = Some(args[i].clone());
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                print_usage();
                process::exit(1);
            }
        }
        i += 1;
    }

    let index_path = index_path.unwrap_or_else(|| {
        print_usage();
        process::exit(1);
    });
    let field_name = field_name.unwrap_or_else(|| {
        print_usage();
        process::exit(1);
    });

    Args {
        index_path,
        field_name,
        segment_prefix,
    }
}

fn print_usage() {
    eprintln!("Usage: listterms -index <index_path> -field <field_name> [-segment <prefix>]");
    eprintln!("\nDumps all terms for a field in lexicographic order.");
}
