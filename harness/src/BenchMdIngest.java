// SPDX-License-Identifier: Apache-2.0
//
// P9b JVM baseline: index a line-per-doc text corpus with IndexWriter
// (TextField, NoMergePolicy, single flush) and report docs/s.
//
// Usage: BenchMdIngest <corpus.txt> <output-dir>

import java.io.BufferedReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.store.FSDirectory;

public final class BenchMdIngest {
    public static void main(String[] args) throws Exception {
        IndexWriterConfig cfg = new IndexWriterConfig();
        cfg.setUseCompoundFile(false);
        cfg.setMergePolicy(NoMergePolicy.INSTANCE);
        cfg.setRAMBufferSizeMB(4096);
        long t0 = System.nanoTime();
        long docs = 0;
        try (IndexWriter w = new IndexWriter(FSDirectory.open(Paths.get(args[1])), cfg);
             BufferedReader r = Files.newBufferedReader(Paths.get(args[0]), StandardCharsets.UTF_8)) {
            String line;
            while ((line = r.readLine()) != null) {
                Document doc = new Document();
                doc.add(new TextField("body", line, Field.Store.NO));
                w.addDocument(doc);
                docs++;
            }
            w.commit();
        }
        double secs = (System.nanoTime() - t0) / 1e9;
        System.out.printf("jvm md ingest: %d docs in %.2f s = %.0f kdocs/s%n",
                docs, secs, docs / secs / 1e3);
    }
}
