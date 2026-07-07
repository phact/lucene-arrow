// SPDX-License-Identifier: Apache-2.0
//
// Search-throughput + recall over the same data for:
//   - jvector      : jVector reading OUR OnDiskGraphIndex file
//   - lucene_hnsw  : Lucene reading OUR .vem/.vex segment
//   - jvector_native : a graph built by jVector's OWN GraphIndexBuilder
//   - lucene_native  : a graph built by Lucene's OWN IndexWriter
// against a ground-truth top-k (exact, from our GPU FlatKnn). This
// isolates "is our graph bad in general, or bad specifically for jVector's
// search?" — build each engine's home-turf graph and compare.
//
// Usage: BenchVectorSearch <jvFile> <lucDir> <queries.bin> <gt.bin>
//                          <dim> <k> <ef> <vectors.bin>

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.FSDirectory;

public final class BenchVectorSearch {
    static final VectorTypeSupport VTS = VectorizationProvider.getInstance().getVectorTypeSupport();

    public static void main(String[] args) throws Exception {
        String jvFile = args[0], lucDir = args[1], qFile = args[2], gtFile = args[3];
        int dim = Integer.parseInt(args[4]), k = Integer.parseInt(args[5]);
        int ef = args.length > 6 ? Integer.parseInt(args[6]) : 100;
        float[][] queries = readVecs(qFile, dim);
        int[][] gt = readInts(gtFile, k);

        benchJvectorFile(jvFile, queries, gt, k, ef);
        benchLuceneDir(lucDir, queries, gt, k, ef, "lucene_hnsw");

        if (args.length > 7) {
            float[][] vecs = readVecs(args[7], dim);
            benchJvectorNative(vecs, queries, gt, dim, k, ef);
            benchLuceneNative(vecs, queries, gt, k, ef);
        }
    }

    // --- jVector reading OUR OnDiskGraphIndex file ---
    static void benchJvectorFile(String file, float[][] q, int[][] gt, int k, int ef) throws Exception {
        try (ReaderSupplier rs = ReaderSupplierFactory.open(Path.of(file))) {
            OnDiskGraphIndex index = OnDiskGraphIndex.load(rs);
            try (var view = index.getView(); GraphSearcher searcher = new GraphSearcher(index)) {
                searchJvector(searcher, view, q, gt, k, ef, "jvector");
            }
        }
    }

    // --- jVector building its OWN graph on the same data ---
    static void benchJvectorNative(float[][] vecs, float[][] q, int[][] gt, int dim, int k, int ef)
            throws Exception {
        List<VectorFloat<?>> list = new ArrayList<>(vecs.length);
        for (float[] v : vecs) list.add(VTS.createFloatVector((Object) v));
        var ravv = new ListRandomAccessVectorValues(list, dim);
        // M=16 (layer-0 degree 32, matching ours), beamWidth=100, default
        // overflow/alpha, addHierarchy=true (jVector 4.x multi-layer).
        var builder = new GraphIndexBuilder(
                ravv, VectorSimilarityFunction.EUCLIDEAN, 16, 100, 1.2f, 1.2f, true);
        GraphIndex graph = builder.build(ravv);
        try (GraphSearcher searcher = new GraphSearcher(graph)) {
            searchJvector(searcher, ravv, q, gt, k, ef, "jvector_native");
        }
    }

    static void searchJvector(GraphSearcher searcher, RandomAccessVectorValues ravv,
            float[][] q, int[][] gt, int k, int ef, String label) throws Exception {
        double best = Double.MAX_VALUE, recall = 0;
        for (int round = 0; round < 3; round++) {
            long t0 = System.nanoTime();
            int hit = 0;
            for (int qi = 0; qi < q.length; qi++) {
                VectorFloat<?> qv = VTS.createFloatVector((Object) q[qi]);
                var ssp = DefaultSearchScoreProvider.exact(qv, VectorSimilarityFunction.EUCLIDEAN, ravv);
                SearchResult r = searcher.search(ssp, ef, Bits.ALL);
                SearchResult.NodeScore[] ns = r.getNodes();
                int n = Math.min(k, ns.length);
                int[] got = new int[n];
                for (int i = 0; i < n; i++) got[i] = ns[i].node;
                hit += overlap(got, gt[qi]);
            }
            best = Math.min(best, (System.nanoTime() - t0) / 1e9);
            recall = hit / (double) (q.length * k);
        }
        System.out.printf("%s,%.0f,%.3f%n", label, q.length / best, recall);
    }

    // --- Lucene reading OUR segment, or a native-built one ---
    static void benchLuceneDir(String dir, float[][] q, int[][] gt, int k, int ef, String label)
            throws Exception {
        try (DirectoryReader reader = DirectoryReader.open(FSDirectory.open(Path.of(dir)))) {
            IndexSearcher s = new IndexSearcher(reader);
            double best = Double.MAX_VALUE, recall = 0;
            for (int round = 0; round < 3; round++) {
                long t0 = System.nanoTime();
                int hit = 0;
                for (int qi = 0; qi < q.length; qi++) {
                    TopDocs td = s.search(new KnnFloatVectorQuery("emb", q[qi], ef), ef);
                    int n = Math.min(k, td.scoreDocs.length);
                    int[] got = new int[n];
                    for (int i = 0; i < n; i++) got[i] = td.scoreDocs[i].doc;
                    hit += overlap(got, gt[qi]);
                }
                best = Math.min(best, (System.nanoTime() - t0) / 1e9);
                recall = hit / (double) (q.length * k);
            }
            System.out.printf("%s,%.0f,%.3f%n", label, q.length / best, recall);
        }
    }

    static void benchLuceneNative(float[][] vecs, float[][] q, int[][] gt, int k, int ef)
            throws Exception {
        Path dir = Files.createTempDirectory("lucnat");
        IndexWriterConfig cfg = new IndexWriterConfig();
        cfg.setUseCompoundFile(false);
        cfg.setMergePolicy(NoMergePolicy.INSTANCE);
        cfg.setRAMBufferSizeMB(2048);
        try (IndexWriter w = new IndexWriter(FSDirectory.open(dir), cfg)) {
            for (float[] v : vecs) {
                Document d = new Document();
                d.add(new KnnFloatVectorField(
                        "emb", v, org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN));
                w.addDocument(d);
            }
            w.commit();
        }
        benchLuceneDir(dir.toString(), q, gt, k, ef, "lucene_native");
    }

    static int overlap(int[] got, int[] groundTruth) {
        HashSet<Integer> s = new HashSet<>();
        for (int g : groundTruth) s.add(g);
        int c = 0;
        for (int x : got) if (s.contains(x)) c++;
        return c;
    }

    static float[][] readVecs(String f, int dim) throws IOException {
        FloatBuffer fb = ByteBuffer.wrap(Files.readAllBytes(Path.of(f)))
                .order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        int q = fb.remaining() / dim;
        float[][] out = new float[q][dim];
        for (int i = 0; i < q; i++) fb.get(out[i]);
        return out;
    }

    static int[][] readInts(String f, int k) throws IOException {
        IntBuffer ib = ByteBuffer.wrap(Files.readAllBytes(Path.of(f)))
                .order(ByteOrder.LITTLE_ENDIAN).asIntBuffer();
        int q = ib.remaining() / k;
        int[][] out = new int[q][k];
        for (int i = 0; i < q; i++) ib.get(out[i]);
        return out;
    }
}
