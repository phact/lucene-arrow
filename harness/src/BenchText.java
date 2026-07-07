// SPDX-License-Identifier: Apache-2.0
//
// P7 §15-style baseline for the postings relation.
//   ingest: BenchText ingest <dir> <numDocs>   (3 tokens/doc, ~111k vocab)
//   scan:   BenchText scan <dir>               (TermsEnum + PostingsEnum full sweep)

import java.nio.file.Paths;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.FSDirectory;

public final class BenchText {
    public static void main(String[] args) throws Exception {
        if (args[0].equals("ingest")) ingest(args[1], Integer.parseInt(args[2]));
        else scan(args[1]);
    }

    static void ingest(String dir, int numDocs) throws Exception {
        IndexWriterConfig cfg = new IndexWriterConfig();
        cfg.setUseCompoundFile(false);
        cfg.setMergePolicy(NoMergePolicy.INSTANCE);
        cfg.setRAMBufferSizeMB(4096);
        long t0 = System.nanoTime();
        try (IndexWriter w = new IndexWriter(FSDirectory.open(Paths.get(dir)), cfg)) {
            StringBuilder sb = new StringBuilder(48);
            for (int i = 0; i < numDocs; i++) {
                sb.setLength(0);
                sb.append('a').append(i % 1000).append(" b").append(i % 9973)
                  .append(" c").append(i % 100000);
                Document doc = new Document();
                doc.add(new TextField("body", sb.toString(), Field.Store.NO));
                w.addDocument(doc);
            }
            w.commit();
        }
        System.out.printf("ingest: %d docs in %.1f s%n", numDocs, (System.nanoTime() - t0) / 1e9);
    }

    static void scan(String dir) throws Exception {
        try (DirectoryReader r = DirectoryReader.open(FSDirectory.open(Paths.get(dir)))) {
            for (int round = 0; round < 3; round++) {
                long t0 = System.nanoTime();
                long postings = 0, terms = 0, sum = 0;
                for (var leafCtx : r.leaves()) {
                    LeafReader leaf = leafCtx.reader();
                    Terms t = leaf.terms("body");
                    TermsEnum te = t.iterator();
                    PostingsEnum pe = null;
                    while (te.next() != null) {
                        terms++;
                        pe = te.postings(pe, PostingsEnum.FREQS);
                        while (pe.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
                            sum += pe.docID() + pe.freq();
                            postings++;
                        }
                    }
                }
                double secs = (System.nanoTime() - t0) / 1e9;
                System.out.printf(
                        "scan round %d: %d terms, %d postings in %.2f s = %.1f Mpostings/s (sum %d)%n",
                        round, terms, postings, secs, postings / secs / 1e6, sum);
            }
        }
    }
}
