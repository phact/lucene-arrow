// SPDX-License-Identifier: Apache-2.0
//
// P8 acceptance: open a jVector OnDiskGraphIndex file written by
// lucene-arrow (Rust, GPU-built graph) with the REAL jVector library and
// run a graph search. Prints "ord,score" per hit.
//
// Usage: VerifyJVector <indexFile> <dim> <queryOrd> <k>

import java.nio.file.Path;

import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

public final class VerifyJVector {
    public static void main(String[] args) throws Exception {
        Path path = Path.of(args[0]);
        int queryOrd = Integer.parseInt(args[2]);
        int k = Integer.parseInt(args[3]);

        try (ReaderSupplier rs = ReaderSupplierFactory.open(path)) {
            OnDiskGraphIndex index = OnDiskGraphIndex.load(rs);
            try (var view = index.getView()) {
                var query = view.getVector(queryOrd);
                var ssp = DefaultSearchScoreProvider.exact(
                        query, VectorSimilarityFunction.EUCLIDEAN, view);
                try (GraphSearcher searcher = new GraphSearcher(index)) {
                    SearchResult r = searcher.search(ssp, k, Bits.ALL);
                    for (SearchResult.NodeScore ns : r.getNodes()) {
                        System.out.println(ns.node + "," + ns.score);
                    }
                }
            }
        }
    }
}
