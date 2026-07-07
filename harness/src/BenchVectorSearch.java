// SPDX-License-Identifier: Apache-2.0
//
// Search-throughput baseline for graph ANN over files lucene-arrow wrote
// from one GPU-built graph: jVector's OnDiskGraphIndex vs Lucene's HNSW
// segment. Both read externally-supplied query vectors and report QPS +
// recall@k against a ground-truth top-k (exact, from our GPU FlatKnn).
// Node/doc ids share the ordinal space across all three engines.
//
// Usage: BenchVectorSearch <jvectorFile> <luceneDir> <queries.bin> <gt.bin> <dim> <k>
//   queries.bin = Q*dim float32 LE ; gt.bin = Q*k int32 LE

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;

import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.FSDirectory;

public final class BenchVectorSearch {
    public static void main(String[] args) throws Exception {
        String jvFile = args[0], lucDir = args[1], qFile = args[2], gtFile = args[3];
        int dim = Integer.parseInt(args[4]), k = Integer.parseInt(args[5]);
        int ef = args.length > 6 ? Integer.parseInt(args[6]) : 100; // search beam width
        float[][] queries = readVecs(qFile, dim);
        int[][] gt = readInts(gtFile, k);
        benchJvector(jvFile, queries, gt, k, ef);
        benchLucene(lucDir, queries, gt, k, ef);
    }

    static void benchJvector(String file, float[][] queries, int[][] gt, int k, int ef)
            throws Exception {
        VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        try (ReaderSupplier rs = ReaderSupplierFactory.open(Path.of(file))) {
            OnDiskGraphIndex index = OnDiskGraphIndex.load(rs);
            try (var view = index.getView(); GraphSearcher searcher = new GraphSearcher(index)) {
                double best = Double.MAX_VALUE, recall = 0;
                for (int round = 0; round < 3; round++) {
                    long t0 = System.nanoTime();
                    int hit = 0;
                    for (int qi = 0; qi < queries.length; qi++) {
                        VectorFloat<?> qv = vts.createFloatVector((Object) queries[qi]);
                        var ssp = DefaultSearchScoreProvider.exact(
                                qv, VectorSimilarityFunction.EUCLIDEAN, view);
                        SearchResult r = searcher.search(ssp, ef, Bits.ALL);
                        SearchResult.NodeScore[] ns = r.getNodes();
                        int[] got = new int[Math.min(k, ns.length)];
                        for (int i = 0; i < got.length; i++) got[i] = ns[i].node;
                        hit += overlap(got, gt[qi]);
                    }
                    best = Math.min(best, (System.nanoTime() - t0) / 1e9);
                    recall = hit / (double) (queries.length * k);
                }
                System.out.printf("jvector,%.0f,%.3f%n", queries.length / best, recall);
            }
        }
    }

    static void benchLucene(String dir, float[][] queries, int[][] gt, int k, int ef)
            throws Exception {
        try (DirectoryReader reader = DirectoryReader.open(FSDirectory.open(Path.of(dir)))) {
            IndexSearcher s = new IndexSearcher(reader);
            double best = Double.MAX_VALUE, recall = 0;
            for (int round = 0; round < 3; round++) {
                long t0 = System.nanoTime();
                int hit = 0;
                for (int qi = 0; qi < queries.length; qi++) {
                    TopDocs td = s.search(new KnnFloatVectorQuery("emb", queries[qi], ef), ef);
                    int n = Math.min(k, td.scoreDocs.length);
                    int[] got = new int[n];
                    for (int i = 0; i < n; i++) got[i] = td.scoreDocs[i].doc;
                    hit += overlap(got, gt[qi]);
                }
                best = Math.min(best, (System.nanoTime() - t0) / 1e9);
                recall = hit / (double) (queries.length * k);
            }
            System.out.printf("lucene_hnsw,%.0f,%.3f%n", queries.length / best, recall);
        }
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
