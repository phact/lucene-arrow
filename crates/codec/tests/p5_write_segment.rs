// SPDX-License-Identifier: Apache-2.0

//! Write-path acceptance (SPEC §10.5): a complete segment commit produced
//! by **our** writers (doc values, `.fnm`, `.si`, stored-fields framing;
//! `segments_N` via Bearing's public commit) must:
//!  1. open through `SegmentDirectory` (Bearing's readers — cross-parser),
//!  2. decode back to the source values,
//!  3. pass Java `CheckIndex` (run when a JDK 21 is present; skipped not
//!     failed otherwise, same policy as the golden tests).

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;

use lucene_arrow_codec::writer::{WriteField, random_segment_id, write_segment_commit};
use lucene_arrow_codec::{DocValuesKind, SegmentDirectory};
use lucene_arrow_cpu::decode_numeric;
use lucene_arrow_docvalues::file::DocValuesFileBuilder;
use lucene_arrow_docvalues::read::{DvField, DvKind, plan_doc_values};

const NUM_DOCS: u32 = 20_000;

fn per_doc(field: usize, d: u32) -> Option<i64> {
    match field {
        0 => Some(1_000_000 + d as i64 * 25),                       // gcd
        1 => Some([7i64, -3, 1 << 41][d as usize % 3]),             // table
        _ => (d % 6 != 2).then_some(d as i64 * -11),                // sparse
    }
}

#[test]
fn our_segment_opens_decodes_and_passes_checkindex() {
    let tmp = tempfile::tempdir().unwrap();
    let segment_id = random_segment_id();

    let mut builder = DocValuesFileBuilder::new(&segment_id, "Lucene90_0");
    let names = ["price", "bucket", "rare"];
    for (i, _) in names.iter().enumerate() {
        let (docs, values): (Vec<u32>, Vec<i64>) = (0..NUM_DOCS)
            .filter_map(|d| per_doc(i, d).map(|v| (d, v)))
            .unzip();
        builder.add_numeric(i as i32, &docs, &values, NUM_DOCS).unwrap();
    }
    // A SORTED column too (300 terms, sparse) — CheckIndex validates the
    // terms dict + reverse index our writer emits.
    {
        use lucene_arrow_docvalues::write::CpuEncoder;
        let (docs, terms): (Vec<u32>, Vec<Vec<u8>>) = (0..NUM_DOCS)
            .filter(|d| d % 5 != 4)
            .map(|d| (d, format!("cat-{:04}", d * 11 % 300).into_bytes()))
            .unzip();
        let term_refs: Vec<&[u8]> = terms.iter().map(|t| t.as_slice()).collect();
        builder.add_sorted_with(&CpuEncoder, 3, &docs, &term_refs, NUM_DOCS).unwrap();
    }
    let (dvm, dvd) = builder.finish();

    let mut fields: Vec<WriteField> = names
        .iter()
        .enumerate()
        .map(|(i, n)| WriteField::doc_values(n.to_string(), i as u32, 1))
        .collect();
    fields.push(WriteField::doc_values("category", 3, 3));
    write_segment_commit(tmp.path(), "_0", segment_id, &fields, NUM_DOCS, &dvm, &dvd).unwrap();

    // 1 + 2: open through Bearing's readers and decode back.
    let dir = SegmentDirectory::open(tmp.path()).unwrap();
    assert_eq!(dir.segments().len(), 1);
    let seg = &dir.segments()[0];
    assert_eq!(seg.max_doc, NUM_DOCS as i32);
    assert_eq!(seg.field("price").unwrap().doc_values, DocValuesKind::Numeric);

    let mut dv_fields: Vec<DvField> = names
        .iter()
        .enumerate()
        .map(|(i, n)| DvField {
            number: i as i32,
            name: n.to_string(),
            kind: DvKind::Numeric,
            has_skip_index: false,
        })
        .collect();
    dv_fields.push(DvField {
        number: 3,
        name: "category".into(),
        kind: DvKind::Sorted,
        has_skip_index: false,
    });
    let dvm_name = seg.files.iter().find(|f| f.ends_with(".dvm")).unwrap();
    let dvd_name = seg.files.iter().find(|f| f.ends_with(".dvd")).unwrap();
    let dvm_r = dir.open_input(&seg.name, dvm_name).unwrap();
    let dvd_r = dir.open_input(&seg.name, dvd_name).unwrap();
    let plans = plan_doc_values(
        dvm_r.slice(0, dvm_r.len()).unwrap(),
        dvd_r.slice(0, dvd_r.len()).unwrap(),
        &dv_fields,
        NUM_DOCS,
        dvd_name,
    )
    .unwrap();
    for (i, name) in names.iter().enumerate() {
        let plan = plans.plans.iter().find(|p| p.column.name == *name).unwrap();
        let array = decode_numeric(plan, dvd_r.slice(0, dvd_r.len()).unwrap()).unwrap();
        let ints = array.as_primitive::<Int64Type>();
        for d in 0..NUM_DOCS {
            match per_doc(i, d) {
                Some(v) => assert_eq!(ints.value(d as usize), v, "{name} doc {d}"),
                None => assert!(ints.is_null(d as usize), "{name} doc {d}"),
            }
        }
    }

    // 3: the CheckIndex gate (SPEC §10.5 acceptance).
    let java = std::path::Path::new("/home/tato/.jbang/cache/jdks/21/bin/java");
    let jar = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../harness/lib/lucene-core-10.3.2.jar");
    if !java.exists() || !jar.exists() {
        eprintln!("skipping CheckIndex gate: JDK 21 / lucene-core jar not present");
        return;
    }
    let output = std::process::Command::new(java)
        .args(["--add-modules", "jdk.incubator.vector", "-cp"])
        .arg(&jar)
        .args(["org.apache.lucene.index.CheckIndex"])
        .arg(tmp.path())
        .args(["-level", "2"])
        .output()
        .expect("run CheckIndex");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success() && stdout.contains("No problems were detected"),
        "CheckIndex failed:\n{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
