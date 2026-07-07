// SPDX-License-Identifier: Apache-2.0
// P9c JVM baseline: timed BM25 OR-queries. Usage: BenchBM25Query <index> <queries.txt> <k>

import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.List;

import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.store.FSDirectory;

public final class BenchBM25Query {
    public static void main(String[] args) throws Exception {
        List<String> queries = Files.readAllLines(Paths.get(args[1]), StandardCharsets.UTF_8);
        int k = Integer.parseInt(args[2]);
        try (DirectoryReader r = DirectoryReader.open(FSDirectory.open(Paths.get(args[0])))) {
            IndexSearcher s = new IndexSearcher(r);
            for (int round = 0; round < 3; round++) {
                long t0 = System.nanoTime();
                long hits = 0;
                for (String q : queries) {
                    BooleanQuery.Builder b = new BooleanQuery.Builder();
                    for (String term : q.split(" "))
                        b.add(new TermQuery(new Term("body", term)), BooleanClause.Occur.SHOULD);
                    TopDocs top = s.search(b.build(), k);
                    hits += top.scoreDocs.length;
                }
                double secs = (System.nanoTime() - t0) / 1e9;
                System.out.printf("round %d: %d queries in %.3f s = %.0f qps (hits %d)%n",
                        round, queries.size(), secs, queries.size() / secs, hits);
            }
        }
    }
}
