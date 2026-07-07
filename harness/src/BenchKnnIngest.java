// SPDX-License-Identifier: Apache-2.0
//
// P6d baseline: JVM flush-time HNSW indexing of KnnFloatVectorField —
// the competing vector-write workflow (SPEC §15.4 flavor of §10.4).
// Vector formula matches the Rust bench (100 clusters + jitter).
//
// Usage: BenchKnnIngest <output-dir> <numDocs> <dim>

import java.nio.file.Paths;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.FSDirectory;

public final class BenchKnnIngest {
    public static void main(String[] args) throws Exception {
        var path = Paths.get(args[0]);
        int numDocs = Integer.parseInt(args[1]);
        int dim = Integer.parseInt(args[2]);

        IndexWriterConfig cfg = new IndexWriterConfig();
        cfg.setUseCompoundFile(false);
        cfg.setMergePolicy(NoMergePolicy.INSTANCE);
        cfg.setRAMBufferSizeMB(4096);
        long t0 = System.nanoTime();
        try (IndexWriter w = new IndexWriter(FSDirectory.open(path), cfg)) {
            float[] v = new float[dim];
            for (int d = 0; d < numDocs; d++) {
                int cluster = d % 100;
                for (int k = 0; k < dim; k++) {
                    long hc = (cluster ^ ((long) k << 32)) * 0x9E3779B97F4A7C15L;
                    long hj = (d ^ ((long) k << 32) ^ 0xABCD) * 0xC2B2AE3D27D4EB4FL;
                    float center = (float) ((hc >>> 11) / (double) (1L << 53) * 2.0 - 1.0);
                    float jitter = (float) ((hj >>> 11) / (double) (1L << 53) * 2.0 - 1.0) * 0.05f;
                    v[k] = center + jitter;
                }
                Document doc = new Document();
                doc.add(new KnnFloatVectorField("emb", v.clone(), VectorSimilarityFunction.EUCLIDEAN));
                w.addDocument(doc);
            }
            w.commit();
        }
        double secs = (System.nanoTime() - t0) / 1e9;
        System.out.printf("jvm knn ingest: %d x %d in %.2f s = %.0f kdocs/s%n",
                numDocs, dim, secs, numDocs / secs / 1e3);
    }
}
