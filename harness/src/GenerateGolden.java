// SPDX-License-Identifier: Apache-2.0
//
// Golden-file generator (SPEC §12.1): writes Lucene 10.3.2 segments with
// known values and an expected.json alongside. Requires JDK 21+.
//
// Shapes covered (numeric focus for P1; extend per milestone):
//   - dense gcd-friendly, dense constant, dense ≤256-distinct (table),
//     wide/negative values
//   - sparse (~1/6 docs) and sparse-dense (>4095 per 65536 block)
//   - "big" field with 100k jumpy values → Java picks multi-block encoding
//     (the case Bearing's writer never emits — our reader must handle it)
//   - a segment with deletes (tests .liv + positional/compact row modes)
//   - a multi-segment commit (tests cross-segment planning)

import java.io.IOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.document.Field;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.index.*;
import org.apache.lucene.store.FSDirectory;

public final class GenerateGolden {

    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            System.err.println("usage: GenerateGolden <output-dir>");
            System.exit(2);
        }
        Path out = Paths.get(args[0]);
        Files.createDirectories(out);

        Map<String, Object> expected = new LinkedHashMap<>();
        expected.put("lucene_version", org.apache.lucene.util.Version.LATEST.toString());

        expected.put("numerics", writeNumerics(out.resolve("numerics")));
        expected.put("multiblock", writeMultiBlock(out.resolve("multiblock")));
        expected.put("deletes", writeDeletes(out.resolve("deletes")));
        expected.put("multisegment", writeMultiSegment(out.resolve("multisegment")));
        expected.put("vectors", writeVectors(out.resolve("vectors")));
        expected.put("keywords", writeKeywords(out.resolve("keywords")));
        expected.put("text", writeText(out.resolve("text")));
        expected.put("textbig", writeTextBig(out.resolve("textbig")));

        try (Writer w = Files.newBufferedWriter(out.resolve("expected.json"), StandardCharsets.UTF_8)) {
            w.write(Json.render(expected));
        }
        System.out.println("golden segments + expected.json written to " + out);
    }

    /** One segment, five numeric fields mirroring crates/codec/tests/p1_docvalues.rs. */
    static Map<String, Object> writeNumerics(Path dir) throws IOException {
        int numDocs = 5000;
        Map<String, List<Long>> perField = new LinkedHashMap<>(); // null = absent
        List<Long> price = new ArrayList<>(), flag = new ArrayList<>(),
                bucket = new ArrayList<>(), rare = new ArrayList<>(), common = new ArrayList<>();
        long[] buckets = {-1_000_000_007L, 3L, 900_719_925_474L};
        for (int i = 0; i < numDocs; i++) {
            price.add(1000L + i * 25L);
            flag.add(7L);
            bucket.add(buckets[i % 3]);
            rare.add(i % 6 == 1 ? i * -13L : null);
            common.add(i % 10 != 0 ? (long) i : null);
        }
        perField.put("price", price);
        perField.put("flag", flag);
        perField.put("bucket", bucket);
        perField.put("rare", rare);
        perField.put("common", common);

        try (IndexWriter w = writer(dir)) {
            for (int i = 0; i < numDocs; i++) {
                Document doc = new Document();
                for (var e : perField.entrySet()) {
                    Long v = e.getValue().get(i);
                    if (v != null) doc.add(new NumericDocValuesField(e.getKey(), v));
                }
                w.addDocument(doc);
            }
            w.commit();
        }
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("num_docs", numDocs);
        m.put("fields", perField);
        return m;
    }

    /** 100k jumpy values in one field: forces Java's multi-block numeric mode. */
    static Map<String, Object> writeMultiBlock(Path dir) throws IOException {
        int numDocs = 100_000;
        List<Long> values = new ArrayList<>(numDocs);
        Random rnd = new Random(42);
        for (int i = 0; i < numDocs; i++) {
            // Per-16384 block regimes with wildly different ranges → blockwise wins.
            long base = (i >> 14) % 2 == 0 ? 0 : 1L << 40;
            values.add(base + rnd.nextInt(1 << (4 + (i >> 14) % 3 * 8)));
        }
        try (IndexWriter w = writer(dir)) {
            for (long v : values) {
                Document doc = new Document();
                doc.add(new NumericDocValuesField("jumpy", v));
                w.addDocument(doc);
            }
            w.commit();
        }
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("num_docs", numDocs);
        m.put("fields", Map.of("jumpy", values));
        return m;
    }

    /** Segment with tombstones: docs i % 9 == 4 deleted after commit. */
    static Map<String, Object> writeDeletes(Path dir) throws IOException {
        int numDocs = 2000;
        List<Long> values = new ArrayList<>();
        List<Integer> deleted = new ArrayList<>();
        try (IndexWriter w = writer(dir)) {
            for (int i = 0; i < numDocs; i++) {
                Document doc = new Document();
                doc.add(new StringField("id", Integer.toString(i), Field.Store.NO));
                doc.add(new NumericDocValuesField("val", i * 3L));
                values.add(i * 3L);
                w.addDocument(doc);
            }
            w.commit();
            for (int i = 0; i < numDocs; i++) {
                if (i % 9 == 4) {
                    w.deleteDocuments(new Term("id", Integer.toString(i)));
                    deleted.add(i);
                }
            }
            w.commit();
        }
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("num_docs", numDocs);
        m.put("deleted_docids", deleted);
        m.put("fields", Map.of("val", values));
        return m;
    }

    /**
     * Tokenized text with df>1 terms for the P7 postings walker:
     * "common" in every doc (freq = i%3+1), "modK" (K = i%7) once per
     * doc, and a unique "uI" per doc. 3000 docs keeps every docFreq
     * under 4096 (no level-1 skip entries).
     */
    static Map<String, Object> writeText(Path dir) throws IOException {
        int numDocs = 3000;
        try (IndexWriter w = writer(dir)) {
            for (int i = 0; i < numDocs; i++) {
                StringBuilder body = new StringBuilder();
                for (int r = 0; r <= i % 3; r++) body.append("common ");
                body.append("mod").append(i % 7).append(' ');
                body.append('u').append(i);
                Document doc = new Document();
                doc.add(new TextField("body", body.toString(), Field.Store.NO));
                w.addDocument(doc);
            }
            w.commit();
        }
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("num_docs", numDocs);
        return m;
    }

    /**
     * Large-df terms for the level-1 skip walk: "all" in every doc
     * (df=10000 → two level-1 entries), "even" in even docs (df=5000 →
     * one), plus uniques.
     */
    static Map<String, Object> writeTextBig(Path dir) throws IOException {
        int numDocs = 10000;
        try (IndexWriter w = writer(dir)) {
            for (int i = 0; i < numDocs; i++) {
                StringBuilder body = new StringBuilder("all ");
                if (i % 2 == 0) body.append("even ");
                body.append('u').append(i);
                Document doc = new Document();
                doc.add(new TextField("body", body.toString(), Field.Store.NO));
                w.addDocument(doc);
            }
            w.commit();
        }
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("num_docs", numDocs);
        return m;
    }

    /** Three segments in one commit (no merge). */
    static Map<String, Object> writeMultiSegment(Path dir) throws IOException {
        List<List<Long>> segments = new ArrayList<>();
        try (IndexWriter w = writer(dir)) {
            long v = 0;
            for (int s = 0; s < 3; s++) {
                List<Long> seg = new ArrayList<>();
                for (int i = 0; i < 1000; i++) {
                    Document doc = new Document();
                    doc.add(new NumericDocValuesField("val", v));
                    seg.add(v++);
                    w.addDocument(doc);
                }
                w.flush(); // cut a segment without merging
                segments.add(seg);
            }
            w.commit();
        }
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("segments", segments);
        return m;
    }

    /**
     * Vector golden: values follow a formula recomputed by the Rust test
     * (exact-representable floats, so no JSON precision issues):
     *   emb[doc][k]   = ((doc*31 + k*7) % 1009) * 0.25f - 100.0f   (dim 64,
     *                    sparse: docs where doc % 5 != 1, EUCLIDEAN)
     *   bytes[doc][k] = (byte)((doc*7 + k*3) % 256 - 128)          (dim 16,
     *                    dense, DOT_PRODUCT)
     */
    static Map<String, Object> writeVectors(Path dir) throws IOException {
        int numDocs = 3000, dim = 64, byteDim = 16;
        try (IndexWriter w = writer(dir)) {
            for (int d = 0; d < numDocs; d++) {
                Document doc = new Document();
                if (d % 5 != 1) {
                    float[] v = new float[dim];
                    for (int k = 0; k < dim; k++) v[k] = ((d * 31 + k * 7) % 1009) * 0.25f - 100.0f;
                    doc.add(new KnnFloatVectorField("emb", v,
                            org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN));
                }
                byte[] b = new byte[byteDim];
                for (int k = 0; k < byteDim; k++) b[k] = (byte) ((d * 7 + k * 3) % 256 - 128);
                doc.add(new KnnByteVectorField("bytes", b,
                        org.apache.lucene.index.VectorSimilarityFunction.DOT_PRODUCT));
                w.addDocument(doc);
            }
            w.commit();
        }
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("num_docs", numDocs);
        m.put("dim", dim);
        m.put("byte_dim", byteDim);
        return m;
    }

    /**
     * Dictionary golden: values follow formulas recomputed by the Rust test.
     *   cat  (SORTED, sparse: doc % 5 != 2):  "cat-%04d" of (doc*7 % 501)  — 501 terms
     *   tags (SORTED_SET, doc % 4 values):    "tag-%03d" of ((doc + j*37) % 211), j in 0..doc%4
     *   nums (SORTED_NUMERIC, 1 + doc % 3 values): doc*5, doc*5 - 100, 7
     */
    static Map<String, Object> writeKeywords(Path dir) throws IOException {
        int numDocs = 3000;
        try (IndexWriter w = writer(dir)) {
            for (int d = 0; d < numDocs; d++) {
                Document doc = new Document();
                if (d % 5 != 2) {
                    doc.add(new SortedDocValuesField("cat",
                            new BytesRef(String.format("cat-%04d", d * 7 % 501))));
                }
                for (int j = 0; j < d % 4; j++) {
                    doc.add(new SortedSetDocValuesField("tags",
                            new BytesRef(String.format("tag-%03d", (d + j * 37) % 211))));
                }
                for (int j = 0; j < 1 + d % 3; j++) {
                    long v = j == 0 ? d * 5L : (j == 1 ? d * 5L - 100 : 7L);
                    doc.add(new SortedNumericDocValuesField("nums", v));
                }
                w.addDocument(doc);
                if (d % 1000 == 999) {
                    w.flush(); // 3 segments: per-segment dicts differ → OrdinalMap test
                }
            }
            w.commit();
        }
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("num_docs", numDocs);
        return m;
    }

    static IndexWriter writer(Path dir) throws IOException {
        IndexWriterConfig cfg = new IndexWriterConfig();
        cfg.setUseCompoundFile(false);
        cfg.setMergePolicy(NoMergePolicy.INSTANCE); // SPEC: merges are out of scope
        return new IndexWriter(FSDirectory.open(dir), cfg);
    }

    /** Tiny JSON writer (avoids a dependency; values are numbers/strings/lists/maps/null). */
    static final class Json {
        static String render(Object o) {
            StringBuilder sb = new StringBuilder();
            write(sb, o);
            return sb.toString();
        }
        static void write(StringBuilder sb, Object o) {
            if (o == null) sb.append("null");
            else if (o instanceof String s) sb.append('"').append(s.replace("\\", "\\\\").replace("\"", "\\\"")).append('"');
            else if (o instanceof Number n) sb.append(n);
            else if (o instanceof Map<?, ?> m) {
                sb.append('{');
                boolean first = true;
                for (var e : m.entrySet()) {
                    if (!first) sb.append(',');
                    first = false;
                    write(sb, e.getKey().toString());
                    sb.append(':');
                    write(sb, e.getValue());
                }
                sb.append('}');
            } else if (o instanceof Iterable<?> it) {
                sb.append('[');
                boolean first = true;
                for (Object e : it) {
                    if (!first) sb.append(',');
                    first = false;
                    write(sb, e);
                }
                sb.append(']');
            } else throw new IllegalArgumentException(o.getClass().toString());
        }
    }
}
