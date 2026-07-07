//! Deterministic Zipfian markdown-ish corpus, one doc per line — shared
//! input for the JVM and Rust ingest benches.
//! Usage: gen_corpus <path> <num_docs>

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("path");
    let n: usize = args.next().expect("num docs").parse().unwrap();
    let len_scale: usize = args.next().map(|v| v.parse().unwrap()).unwrap_or(1);
    let vocab = 200_000u64;
    let mut out = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
    use std::io::Write;
    let mut state = 0x9E3779B97F4A7C15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for d in 0..n {
        let len = (40 + (next() % 80) as usize) * len_scale;
        write!(out, "# doc{d}").unwrap();
        for _ in 0..len {
            // Zipf-ish: rank = vocab^u for uniform u -> heavy head.
            let u = (next() % 1_000_000) as f64 / 1_000_000.0;
            let rank = ((vocab as f64).powf(u)) as u64 % vocab;
            write!(out, " w{rank}").unwrap();
        }
        writeln!(out).unwrap();
    }
}
