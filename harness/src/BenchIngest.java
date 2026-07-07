// SPDX-License-Identifier: Apache-2.0
//
// Kill-criterion baseline (c), SPEC §15.4: JVM IndexWriter bulk ingest of
// the same data our DoPut writes. Also generates the big index that
// BenchScan and our scan benches read (same field shapes as the Rust
// write bench: dense-gcd, dense-20bit, dense-64bit, sparse-16bit).
//
// Usage: BenchIngest <output-dir> <numDocs>

import java.nio.file.Paths;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.store.FSDirectory;

public final class BenchIngest {
    public static void main(String[] args) throws Exception {
        var path = Paths.get(args[0]);
        int numDocs = Integer.parseInt(args[1]);

        IndexWriterConfig cfg = new IndexWriterConfig();
        cfg.setUseCompoundFile(false);
        cfg.setMergePolicy(NoMergePolicy.INSTANCE);
        cfg.setRAMBufferSizeMB(1024); // few, large segments
        long t0 = System.nanoTime();
        try (IndexWriter w = new IndexWriter(FSDirectory.open(path), cfg)) {
            Document doc = new Document();
            NumericDocValuesField f0 = new NumericDocValuesField("f0", 0);
            NumericDocValuesField f1 = new NumericDocValuesField("f1", 0);
            NumericDocValuesField f2 = new NumericDocValuesField("f2", 0);
            NumericDocValuesField f3 = new NumericDocValuesField("f3", 0);
            for (int d = 0; d < numDocs; d++) {
                doc.clear();
                f0.setLongValue(1_000_000L + (long) (d % 4096) * 25);
                doc.add(f0);
                f1.setLongValue((d * 0x9E37L) & 0xF_FFFF);
                doc.add(f1);
                f2.setLongValue(d * 0x9E3779B97F4A7C15L);
                doc.add(f2);
                if (d % 4 != 3) {
                    f3.setLongValue(d & 0xFFFF);
                    doc.add(f3);
                }
                w.addDocument(doc);
            }
            w.commit();
        }
        double secs = (System.nanoTime() - t0) / 1e9;
        System.out.printf("ingest: %d docs in %.2f s = %.2f Mdocs/s%n",
                numDocs, secs, numDocs / secs / 1e6);
    }
}
