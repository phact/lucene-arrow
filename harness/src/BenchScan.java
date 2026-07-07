// SPDX-License-Identifier: Apache-2.0
//
// Kill-criterion baseline (a), SPEC §15.4: JVM Lucene NumericDocValues
// full-column scan — the fastest possible JVM read of this data, and the
// floor cost of any scroll/export workflow (export = this scan + serialize
// + write + re-read elsewhere).
//
// Usage: BenchScan <index-dir> <field...>

import java.nio.file.Paths;

import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.FSDirectory;

public final class BenchScan {
    public static void main(String[] args) throws Exception {
        var dir = FSDirectory.open(Paths.get(args[0]));
        try (DirectoryReader reader = DirectoryReader.open(dir)) {
            long rowsPerPass = 0;
            long sink = 0;
            double best = Double.MAX_VALUE;
            for (int pass = 0; pass < 5; pass++) {
                rowsPerPass = 0;
                long t0 = System.nanoTime();
                for (int f = 1; f < args.length; f++) {
                    for (LeafReaderContext leaf : reader.leaves()) {
                        NumericDocValues dv = leaf.reader().getNumericDocValues(args[f]);
                        if (dv == null) continue;
                        while (dv.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
                            sink += dv.longValue();
                            rowsPerPass++;
                        }
                    }
                }
                best = Math.min(best, (System.nanoTime() - t0) / 1e9);
            }
            System.out.printf("scan: %d values/pass, best %.3f s = %.2f Mvals/s (sink=%d)%n",
                    rowsPerPass, best, rowsPerPass / best / 1e6, sink == 42 ? 1 : 0);
        }
    }
}
