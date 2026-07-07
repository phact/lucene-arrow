// SPDX-License-Identifier: Apache-2.0

//! P9a acceptance: real markdown (this repo's own docs) → our analyzer +
//! CPU aggregation → Bearing's block-tree writer + our norms/.fnm/.si →
//! a complete segment that (1) our P7 reader round-trips exactly,
//! (2) Java CheckIndex accepts, (3) Java BM25 TermQuery scores match the
//! formula computed from our own aggregation.

use lucene_arrow_postings::build::IndexBuilder;
use lucene_arrow_postings::segment::write_postings_files;
use lucene_arrow_postings::text::{byte4_to_int, int_to_byte4, tokenize};
use lucene_arrow_postings::walk::{FieldTraits, walk_terms};
use lucene_arrow_postings::{parse_tmd, root_block};

use lucene_arrow_codec::norms::NormsFilesBuilder;
use lucene_arrow_codec::writer::{
    WriteField, commit_segments, random_segment_id, write_segment_files_full,
};

fn corpus() -> Vec<String> {
    // Real markdown: every tracked .md in the repo root + memory docs.
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut docs = Vec::new();
    for name in ["README.md", "SPEC.md"] {
        if let Ok(text) = std::fs::read_to_string(root.join(name)) {
            // Split into sections so we get a real multi-doc corpus.
            for chunk in text.split("\n## ") {
                if chunk.trim().len() > 40 {
                    docs.push(chunk.to_string());
                }
            }
        }
    }
    assert!(docs.len() >= 10, "need a real corpus");
    docs
}

#[test]
fn markdown_to_bm25_segment_all_gates() {
    let docs = corpus();
    let mut b = IndexBuilder::new();
    for d in &docs {
        b.add_doc(d);
    }
    let num_docs = b.num_docs();
    let inv = b.finish();
    let (num_terms, total_rows) = (inv.num_terms(), inv.docs.len());
    eprintln!("corpus: {num_docs} docs, {num_terms} terms, {total_rows} postings");

    let tmp = tempfile::tempdir().unwrap();
    let id = random_segment_id();

    // Postings via Bearing's writer.
    let postings_files =
        write_postings_files(tmp.path(), "_0", &id, "body", 0, &inv).unwrap();

    // Norms (ours).
    let mut nb = NormsFilesBuilder::new(&id, "");
    nb.add_dense_field(0, &inv.norms).unwrap();
    let (nvm, nvd) = nb.finish();

    // Empty doc values + the rest of the segment (ours).
    let dv = lucene_arrow_docvalues::file::DocValuesFileBuilder::new(&id, "Lucene90_0");
    let (dvm, dvd) = dv.finish();
    let field = WriteField {
        name: "body".into(),
        number: 0,
        doc_values_type: 0,
        vector_dim: 0,
        vector_encoding: 0,
        vector_similarity: 0,
        index_options: 2, // DOCS_AND_FREQS
    };
    let extra = [("_0.nvm".to_string(), nvm.as_slice()), ("_0.nvd".to_string(), nvd.as_slice())];
    let seg = write_segment_files_full(
        tmp.path(), "_0", id, &[field], num_docs, &dvm, &dvd, &extra, &postings_files,
    )
    .unwrap();
    commit_segments(tmp.path(), &[seg]).unwrap();

    // Gate 1: our P7 reader round-trips every term + posting.
    let tmd = std::fs::read(tmp.path().join("_0_Lucene103_0.tmd")).unwrap();
    let tim = std::fs::read(tmp.path().join("_0_Lucene103_0.tim")).unwrap();
    let tip = std::fs::read(tmp.path().join("_0_Lucene103_0.tip")).unwrap();
    let docf = std::fs::read(tmp.path().join("_0_Lucene103_0.doc")).unwrap();
    let traits = FieldTraits { has_freqs: true, has_positions: false, has_offsets: false };
    let m = &parse_tmd(&tmd, |_| true).unwrap()[0];
    assert_eq!(m.num_terms as usize, num_terms);
    assert_eq!(m.sum_total_term_freq, inv.sum_total_term_freq);
    let tip_slice = &tip[m.index_start_fp as usize..m.index_end_fp as usize];
    let root = root_block(tip_slice, m.trie_root_fp).unwrap();
    let mut ord = 0usize;
    walk_terms(&tim, root.fp, traits, |term, df, ttf, tm| {
        assert_eq!(term, inv.term(ord), "term ord {ord}");
        let (docs, freqs) = inv.postings(ord);
        assert_eq!(df as usize, docs.len());
        assert_eq!(ttf, freqs.iter().map(|&f| f as i64).sum::<i64>());
        let mut i = 0usize;
        lucene_arrow_postings::doc::scan_postings(&docf, df, ttf, tm, traits, |d, f| {
            assert_eq!((d, f), (docs[i], freqs[i]), "posting {i} of ord {ord}");
            i += 1;
            Ok(())
        })
        .unwrap();
        ord += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(ord, num_terms);
    eprintln!("gate 1: P7 reader round-trips all {total_rows} postings");

    // Gates 2+3 need Java.
    let java = std::path::Path::new("/home/tato/.jbang/cache/jdks/21/bin/java");
    let harness = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../harness");
    let jar = harness.join("lib/lucene-core-10.3.2.jar");
    if !java.exists() || !jar.exists() {
        eprintln!("skipping Java gates");
        return;
    }
    let output = std::process::Command::new(java)
        .args(["--add-modules", "jdk.incubator.vector", "-cp"])
        .arg(&jar)
        .arg("org.apache.lucene.index.CheckIndex")
        .arg(tmp.path())
        .args(["-level", "2"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success() && stdout.contains("No problems were detected"),
        "CheckIndex failed:\n{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("test: terms, freq"), "postings not validated:\n{stdout}");
    eprintln!("gate 2: CheckIndex clean (postings + norms validated)");

    // Gate 3: BM25 parity. Pick a mid-df term; compute the expected score
    // for its highest-freq doc from OUR aggregation, Lucene-style.
    let probe_ord = (0..num_terms)
        .find(|&t| {
            let (d, _) = inv.postings(t);
            d.len() >= 3 && d.len() <= num_docs as usize / 2
        })
        .expect("probe term");
    let probe = String::from_utf8(inv.term(probe_ord).to_vec()).unwrap();
    let (pdocs, pfreqs) = inv.postings(probe_ord);
    let (k1, bp) = (1.2f64, 0.75f64);
    let n = num_docs as f64;
    let df = pdocs.len() as f64;
    let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
    // avgdl over DECODED norms (Lucene computes avg field length as
    // sumTotalTermFreq / docCount).
    let avgdl = inv.sum_total_term_freq as f64 / n;
    let expected: Vec<f64> = pdocs
        .iter()
        .zip(pfreqs)
        .map(|(&d, &f)| {
            let dl = byte4_to_int(inv.norms[d as usize] as u8) as f64;
            idf * f as f64 / (f as f64 + k1 * (1.0 - bp + bp * dl / avgdl))
        })
        .collect();

    let build = harness.join("build");
    std::process::Command::new(java.parent().unwrap().join("javac"))
        .args(["-cp"]).arg(&jar).args(["-d"]).arg(&build)
        .arg(harness.join("src/BM25Parity.java"))
        .status()
        .unwrap();
    let output = std::process::Command::new(java)
        .args(["--add-modules", "jdk.incubator.vector", "-cp"])
        .arg(format!("{}:{}", jar.display(), build.display()))
        .arg("BM25Parity")
        .arg(tmp.path())
        .args(["body", &probe, &pdocs.len().to_string()])
        .output()
        .unwrap();
    assert!(output.status.success(), "BM25Parity: {}", String::from_utf8_lossy(&output.stderr));
    // Java prints doc,score sorted by docid.
    let mut got: Vec<(u32, f64)> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|l| {
            let (d, s) = l.split_once(',')?;
            Some((d.parse().ok()?, s.parse().ok()?))
        })
        .collect();
    got.sort_by_key(|&(d, _)| d);
    assert_eq!(got.len(), pdocs.len(), "hit count for {probe}");
    for ((gd, gs), (ed, es)) in got.iter().zip(pdocs.iter().zip(&expected)) {
        assert_eq!(gd, ed);
        assert!(
            (gs - es).abs() < 1e-4 * es.max(1e-9),
            "score mismatch doc {gd}: java {gs} vs ours {es} (term {probe})"
        );
    }
    eprintln!("gate 3: BM25 parity on term {probe:?} across {} docs", got.len());
    let _ = int_to_byte4(0);
    let _ = tokenize("");
}
