// SPDX-License-Identifier: Apache-2.0

//! P7 gate: full-enumeration walk of a REAL Java Lucene 10.3 terms dict
//! (harness/golden/deletes: `id` StringField, 2000 unique terms, df=1
//! each — exercises singleton-RLE stats, inline singleton docids, floor
//! blocks, and suffix compression on true Java bytes).

use lucene_arrow_postings::walk::{FieldTraits, walk_terms};
use lucene_arrow_postings::{parse_tmd, root_block};

#[test]
fn walks_java_golden_terms_exactly() {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../harness/golden/deletes");
    if !dir.exists() {
        eprintln!("golden segments absent; run harness/run.sh golden");
        return;
    }
    let tmd = std::fs::read(dir.join("_0_Lucene103_0.tmd")).unwrap();
    let tim = std::fs::read(dir.join("_0_Lucene103_0.tim")).unwrap();
    let tip = std::fs::read(dir.join("_0_Lucene103_0.tip")).unwrap();

    // `id` is a StringField (IndexOptions.DOCS → no freqs).
    let metas = parse_tmd(&tmd, |_| false).unwrap();
    assert_eq!(metas.len(), 1, "only `id` is indexed");
    let m = &metas[0];
    assert_eq!(m.num_terms, 2000);
    assert_eq!(m.doc_count, 2000);
    assert_eq!(m.min_term, b"0");
    assert_eq!(m.max_term, b"999");

    let tip_slice = &tip[m.index_start_fp as usize..m.index_end_fp as usize];
    let root = root_block(tip_slice, m.trie_root_fp).unwrap();
    let traits = FieldTraits { has_freqs: false, has_positions: false, has_offsets: false };
    let mut got: Vec<(Vec<u8>, u32, i64, i64)> = Vec::new();
    walk_terms(&tim, root.fp, traits, |term, df, ttf, tm| {
        got.push((term.to_vec(), df, ttf, tm.singleton_doc_id));
        Ok(())
    })
    .unwrap();

    let mut expected: Vec<Vec<u8>> =
        (0..2000).map(|i| i.to_string().into_bytes()).collect();
    expected.sort();
    assert_eq!(got.len(), 2000);
    for (i, (term, df, ttf, singleton)) in got.iter().enumerate() {
        assert_eq!(term, &expected[i], "term ord {i}");
        assert_eq!(*df, 1);
        assert_eq!(*ttf, 1);
        let docid: i64 = String::from_utf8(term.clone()).unwrap().parse().unwrap();
        assert_eq!(*singleton, docid, "singleton docid for term {i}");
    }
}

/// P7 postings gate: tokenized text golden ("common" df=3000 freq=i%3+1,
/// "mod0".."mod6" df≈429, 3000 unique "uN") — exercises FOR/CONSECUTIVE
/// doc blocks, PFor freq blocks, group-varint tails, and non-singleton
/// metadata chains, all on real Java bytes.
#[test]
fn scans_java_golden_postings_exactly() {
    use lucene_arrow_postings::doc::scan_postings;

    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../harness/golden/text");
    if !dir.exists() {
        eprintln!("text golden absent; run harness/run.sh golden");
        return;
    }
    let tmd = std::fs::read(dir.join("_0_Lucene103_0.tmd")).unwrap();
    let tim = std::fs::read(dir.join("_0_Lucene103_0.tim")).unwrap();
    let tip = std::fs::read(dir.join("_0_Lucene103_0.tip")).unwrap();
    let doc = std::fs::read(dir.join("_0_Lucene103_0.doc")).unwrap();

    let n = 3000u32;
    // TextField → DOCS_AND_FREQS_AND_POSITIONS.
    let traits = FieldTraits { has_freqs: true, has_positions: true, has_offsets: false };
    let metas = parse_tmd(&tmd, |_| true).unwrap();
    assert_eq!(metas.len(), 1);
    let m = &metas[0];
    assert_eq!(m.num_terms as u32, 1 + 7 + n);

    let tip_slice = &tip[m.index_start_fp as usize..m.index_end_fp as usize];
    let root = root_block(tip_slice, m.trie_root_fp).unwrap();

    let mut n_terms = 0u32;
    let mut checked_postings = 0u64;
    walk_terms(&tim, root.fp, traits, |term, df, ttf, tm| {
        n_terms += 1;
        let name = String::from_utf8(term.to_vec()).unwrap();
        // Expected doc list + freqs from the generator formula.
        let expected: Vec<(u32, u32)> = if name == "common" {
            (0..n).map(|i| (i, i % 3 + 1)).collect()
        } else if let Some(k) = name.strip_prefix("mod").and_then(|k| k.parse::<u32>().ok()) {
            (0..n).filter(|i| i % 7 == k).map(|i| (i, 1)).collect()
        } else if let Some(i) = name.strip_prefix("u").and_then(|i| i.parse::<u32>().ok()) {
            vec![(i, 1)]
        } else {
            panic!("unexpected term {name}");
        };
        assert_eq!(df as usize, expected.len(), "df for {name}");
        assert_eq!(ttf, expected.iter().map(|&(_, f)| f as i64).sum::<i64>(), "ttf for {name}");

        if std::env::var("P7_DEBUG").is_ok() {
            eprintln!("term {name} df={df} fp={} sing={}", tm.doc_start_fp, tm.singleton_doc_id);
        }
        let mut got = Vec::with_capacity(df as usize);
        scan_postings(&doc, df, ttf, tm, traits, |d, f| {
            got.push((d, f));
            Ok(())
        })
        .unwrap();
        assert_eq!(got, expected, "postings for {name}");
        checked_postings += got.len() as u64;
        Ok(())
    })
    .unwrap();
    assert_eq!(n_terms, 1 + 7 + n);
    eprintln!("verified {checked_postings} postings across {n_terms} terms");
}

/// Level-1 skip walk + CSR assembly gate: textbig golden ("all" df=10000
/// spans two level-1 entries; "even" df=5000 spans one; 10000 uniques).
#[test]
fn level1_and_csr_on_textbig() {
    use lucene_arrow_postings::coo::read_csr;

    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../harness/golden/textbig");
    if !dir.exists() {
        eprintln!("textbig golden absent; run harness/run.sh golden");
        return;
    }
    let tmd = std::fs::read(dir.join("_0_Lucene103_0.tmd")).unwrap();
    let tim = std::fs::read(dir.join("_0_Lucene103_0.tim")).unwrap();
    let tip = std::fs::read(dir.join("_0_Lucene103_0.tip")).unwrap();
    let doc = std::fs::read(dir.join("_0_Lucene103_0.doc")).unwrap();

    let n = 10_000u32;
    let traits = FieldTraits { has_freqs: true, has_positions: true, has_offsets: false };
    let m = &parse_tmd(&tmd, |_| true).unwrap()[0];
    assert_eq!(m.num_terms as u32, 2 + n);
    let tip_slice = &tip[m.index_start_fp as usize..m.index_end_fp as usize];
    let root = root_block(tip_slice, m.trie_root_fp).unwrap();

    let csr = read_csr(&tim, &doc, root.fp, traits).unwrap();
    assert_eq!(csr.num_terms() as u32, 2 + n);
    assert_eq!(csr.num_rows() as u32, n + n / 2 + n); // all + even + uniques

    // Term 0 is "all": docs 0..n, freq 1.
    assert_eq!(csr.term(0), b"all");
    let span = csr.term_offsets[0] as usize..csr.term_offsets[1] as usize;
    assert_eq!(span.len() as u32, n);
    for (i, d) in csr.docs[span.clone()].iter().enumerate() {
        assert_eq!(*d, i as u32, "all doc {i}");
    }
    assert!(csr.freqs[span].iter().all(|&f| f == 1));

    // Term 1 is "even": docs 0,2,4,...
    assert_eq!(csr.term(1), b"even");
    let span = csr.term_offsets[1] as usize..csr.term_offsets[2] as usize;
    assert_eq!(span.len() as u32, n / 2);
    for (i, d) in csr.docs[span].iter().enumerate() {
        assert_eq!(*d, 2 * i as u32, "even doc {i}");
    }

    // COO materialization is consistent.
    let ords = csr.term_ords();
    assert_eq!(ords.len(), csr.num_rows());
    assert_eq!(ords[0], 0);
    assert_eq!(*ords.last().unwrap() as usize, csr.num_terms() - 1);
}
