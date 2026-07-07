// SPDX-License-Identifier: Apache-2.0
//
// P9 gate: run a BM25 TermQuery over a lucene-arrow-written text segment
// and print "docid,score" for all hits.
//
// Usage: BM25Parity <indexDir> <field> <term> <k>

import java.nio.file.Paths;

import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.FSDirectory;

public final class BM25Parity {
    public static void main(String[] args) throws Exception {
        try (DirectoryReader r = DirectoryReader.open(FSDirectory.open(Paths.get(args[0])))) {
            IndexSearcher s = new IndexSearcher(r);
            TopDocs top = s.search(new TermQuery(new Term(args[1], args[2])),
                    Integer.parseInt(args[3]));
            for (ScoreDoc sd : top.scoreDocs) {
                System.out.println(sd.doc + "," + sd.score);
            }
        }
    }
}
