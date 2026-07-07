// SPDX-License-Identifier: Apache-2.0

//! Indexes documents from a directory into a Lucene-compatible index.

#[cfg(feature = "jemalloc")]
#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process;
use std::time::{Instant, UNIX_EPOCH};

use log::error;

use bearing::prelude::{
    Document, DocumentBuilder, FSDirectory, IndexWriter, IndexWriterConfig, TermVectorOptions,
    TokenizedIndexOptions, binary_dv, double_field, double_range, feature, float_field,
    float_range, int_field, int_range, keyword, lat_lon, long_field, long_range, numeric_dv,
    sorted_dv, sorted_numeric_dv, sorted_set_dv, stored, string, text,
};

struct CliArgs {
    index_path: String,
    docs_path: String,
    max_buffered_docs: i32,
    ram_buffer_size_mb: f64,
    num_threads: usize,
    use_compound_file: bool,
}

fn parse_args() -> CliArgs {
    let args: Vec<String> = env::args().collect();

    let mut index_path = String::from("index");
    let mut docs_path: Option<String> = None;
    let mut max_buffered_docs: i32 = -1;
    let mut ram_buffer_size_mb: f64 = -1.0;
    let mut num_threads: usize = 8;
    let mut use_compound_file: bool = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-index" => {
                i += 1;
                if i >= args.len() {
                    error!("-index requires a value");
                    process::exit(1);
                }
                index_path = args[i].clone();
            }
            "-docs" => {
                i += 1;
                if i >= args.len() {
                    error!("-docs requires a value");
                    process::exit(1);
                }
                docs_path = Some(args[i].clone());
            }
            "--max-buffered-docs" => {
                i += 1;
                if i >= args.len() {
                    error!("--max-buffered-docs requires a value");
                    process::exit(1);
                }
                max_buffered_docs = match args[i].parse() {
                    Ok(n) => n,
                    Err(_) => {
                        error!("--max-buffered-docs must be an integer");
                        process::exit(1);
                    }
                };
            }
            "--ram-buffer-size" => {
                i += 1;
                if i >= args.len() {
                    error!("--ram-buffer-size requires a value");
                    process::exit(1);
                }
                ram_buffer_size_mb = match args[i].parse() {
                    Ok(n) => n,
                    Err(_) => {
                        error!("--ram-buffer-size must be a number (MB)");
                        process::exit(1);
                    }
                };
            }
            "--threads" => {
                i += 1;
                if i >= args.len() {
                    error!("--threads requires a value");
                    process::exit(1);
                }
                num_threads = match args[i].parse() {
                    Ok(n) if n >= 1 => n,
                    _ => {
                        error!("--threads must be a positive integer");
                        process::exit(1);
                    }
                };
            }
            "--compound" => {
                use_compound_file = true;
            }
            other => {
                error!("Unknown parameter: {other}");
                print_usage();
                process::exit(1);
            }
        }
        i += 1;
    }

    let docs_path = match docs_path {
        Some(p) => p,
        None => {
            print_usage();
            process::exit(1);
        }
    };

    CliArgs {
        index_path,
        docs_path,
        max_buffered_docs,
        ram_buffer_size_mb,
        num_threads,
        use_compound_file,
    }
}

fn main() {
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Warn)
        .env()
        .init()
        .unwrap();

    let args = parse_args();

    let doc_dir = PathBuf::from(&args.docs_path);
    if !doc_dir.is_dir() {
        error!(
            "Document directory '{}' does not exist or is not readable",
            doc_dir.display()
        );
        process::exit(1);
    }

    // Remove any existing index files in the output directory
    let index_dir = Path::new(&args.index_path);
    if index_dir.is_dir()
        && let Err(e) = remove_index_files(index_dir)
    {
        error!("Error cleaning index directory '{}': {e}", args.index_path);
        process::exit(1);
    }

    println!(
        "Creating index in '{}' from documents in '{}'...",
        args.index_path, args.docs_path
    );

    let start = Instant::now();

    // Collect document paths
    let mut doc_paths = Vec::new();
    walk_docs(&doc_dir, &mut doc_paths);

    let mut config = IndexWriterConfig::default()
        .num_threads(args.num_threads)
        .max_buffered_docs(args.max_buffered_docs)
        .use_compound_file(args.use_compound_file);
    if args.ram_buffer_size_mb >= 0.0 {
        config = config.ram_buffer_size_mb(args.ram_buffer_size_mb);
    }

    let directory = FSDirectory::open_with_file_handles(index_dir).unwrap();
    let writer = IndexWriter::new(config, directory);

    // Threading is handled internally by IndexWriter via num_threads config.
    for path in &doc_paths {
        let doc = make_document(path);
        writer.add_document(doc).unwrap();
        log::info!("indexed: {}", path.display());
    }

    let segments = writer.commit().unwrap();

    let elapsed = start.elapsed();
    let total_docs: i32 = segments.iter().map(|s| s.doc_count).sum();

    println!();
    println!("Indexed {total_docs} documents in {elapsed:.2?}");
    println!("Wrote {} segment(s) to {}", segments.len(), args.index_path);

    // Report index file sizes
    if let Ok(entries) = fs::read_dir(index_dir) {
        let mut files: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_file())
            .collect();
        files.sort_by_key(|e| e.file_name());
        for entry in &files {
            let name = entry.file_name();
            match entry.metadata() {
                Ok(meta) => log::info!("{}: {} bytes", name.to_string_lossy(), meta.len()),
                Err(_) => log::info!("{}", name.to_string_lossy()),
            }
        }
    }

    println!("\nIndex written to '{}'", args.index_path);
}

/// Creates a Document from a file path with all supported field types.
fn make_document(path: &Path) -> Document {
    let path_str = path.to_string_lossy().to_string();
    let file_name = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let title = match file_name.rfind('.') {
        Some(dot) => file_name[..dot].to_string(),
        None => file_name,
    };

    let metadata = fs::metadata(path).ok();
    let file_size = metadata.as_ref().map(|m| m.len()).unwrap_or(0);
    let modified = metadata
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);

    let contents_field = text("contents")
        .with_index_options(TokenizedIndexOptions::PositionsAndOffsets)
        .with_term_vectors(TermVectorOptions::PositionsAndOffsets)
        .value(path.to_path_buf());

    let lat = 40.7128 + (file_size % 10) as f64 * 0.01;
    let lon = -74.006 + (file_size % 10) as f64 * 0.01;

    let mut builder = DocumentBuilder::new();

    builder = builder
        .add_field(keyword("path").stored().value(&path_str))
        .add_field(long_field("modified").value(modified))
        .add_field(contents_field)
        .add_field(string("title").stored().value(&title))
        .add_field(int_field("size").stored().value(file_size as i32))
        .add_field(
            float_field("score")
                .stored()
                .value((file_size % 100) as f32 / 10.0),
        )
        .add_field(
            double_field("rating")
                .stored()
                .value(file_size as f64 * 1.5),
        );

    // Stored-only fields
    builder = builder
        .add_field(stored("notes").string("indexed by Rust"))
        .add_field(stored("extra_int").int((file_size % 1000) as i32))
        .add_field(stored("extra_float").float((file_size % 100) as f32 / 3.0))
        .add_field(stored("extra_double").double(file_size as f64 * 0.123));

    // LatLonPoint
    builder = builder.add_field(lat_lon("location").value(lat, lon));

    // Range fields
    builder = builder
        .add_field(int_range("int_range").value(&[file_size as i32], &[file_size as i32 + 100]))
        .add_field(long_range("long_range").value(&[file_size as i64], &[file_size as i64 + 1000]))
        .add_field(
            float_range("float_range")
                .value(&[file_size as f32 / 10.0], &[file_size as f32 / 10.0 + 1.0]),
        )
        .add_field(
            double_range("double_range")
                .value(&[file_size as f64 * 0.1], &[file_size as f64 * 0.1 + 1.0]),
        );

    // FeatureField
    builder = builder
        .add_field(feature("features").value("pagerank", (file_size % 100) as f32 / 10.0 + 0.5))
        .add_field(feature("features").value("freshness", (file_size % 50) as f32 / 5.0 + 1.0));

    // Doc-values-only fields
    builder = builder
        .add_field(numeric_dv("dv_count").value(file_size as i64))
        .add_field(binary_dv("dv_hash").value(format!("{:016x}", file_size).into_bytes()))
        .add_field(sorted_dv("dv_category").value(title.as_bytes().to_vec()))
        .add_field(sorted_set_dv("dv_tag").value(vec![title.as_bytes().to_vec()]))
        .add_field(sorted_numeric_dv("dv_priority").value(vec![(file_size % 10) as i64]));

    // Sparse doc values — only even-numbered docs
    if let Some(doc_num) = parse_doc_num(&title)
        && doc_num % 2 == 0
    {
        builder = builder.add_field(numeric_dv("sparse_count").value((doc_num * 100) as i64));
    }

    builder.build()
}

/// Extracts a doc number from a title like "doc_003" or "science".
/// Returns `None` if no number can be parsed.
fn parse_doc_num(title: &str) -> Option<i32> {
    let suffix = title.rsplit('_').next()?;
    suffix.parse().ok()
}

/// Recursively collects file paths from a directory.
fn walk_docs(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) => {
            error!("Error reading directory '{}': {e}", dir.display());
            return;
        }
    };

    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                error!("Error reading entry: {e}");
                continue;
            }
        };

        let path = entry.path();
        if path.is_dir() {
            walk_docs(&path, out);
        } else {
            out.push(path);
        }
    }
}

/// Removes known Lucene index files from a directory.
fn remove_index_files(dir: &Path) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n,
            None => continue,
        };
        if is_index_file(name) {
            fs::remove_file(&path)?;
        }
    }
    Ok(())
}

/// Returns true if the filename looks like a Lucene index file.
fn is_index_file(name: &str) -> bool {
    if name.starts_with("segments_") || name.starts_with("pending_segments_") {
        return true;
    }
    if name.starts_with('_') && name.contains('.') {
        return true;
    }
    false
}

fn print_usage() {
    eprintln!(
        "Usage: indexfiles [-index INDEX_PATH] -docs DOCS_PATH [OPTIONS]\n\n\
         Indexes documents with all supported field types.\n\n\
         Options:\n\
         \t--max-buffered-docs N  Flush after N docs per segment (-1 = disabled)\n\
         \t--ram-buffer-size MB   RAM buffer size in MB (default: 64.0)\n\
         \t--threads N            Number of indexing threads (default: 8)\n\
         \t--compound             Package segment files into .cfs/.cfe (default: non-compound)"
    );
}
