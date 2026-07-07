// SPDX-License-Identifier: Apache-2.0
//
// P6c acceptance (SPEC §13): run a Java Lucene KNN query over a segment
// whose vectors + HNSW graph were written by lucene-arrow. Query = the
// stored vector of docid <queryDoc>; prints "docid,score" per hit.
//
// Usage: VerifyKnn <indexDir> <field> <queryDoc> <k>

import java.nio.file.Paths;

import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.FSDirectory;

public final class VerifyKnn {
    public static void main(String[] args) throws Exception {
        var dir = FSDirectory.open(Paths.get(args[0]));
        String field = args[1];
        int queryDoc = Integer.parseInt(args[2]);
        int k = Integer.parseInt(args[3]);
        try (DirectoryReader reader = DirectoryReader.open(dir)) {
            FloatVectorValues values = reader.leaves().get(0).reader().getFloatVectorValues(field);
            float[] query = values.vectorValue(queryDoc).clone();

            IndexSearcher searcher = new IndexSearcher(reader);
            TopDocs top = searcher.search(new KnnFloatVectorQuery(field, query, k), k);
            for (ScoreDoc sd : top.scoreDocs) {
                System.out.println(sd.doc + "," + sd.score);
            }
        }
    }
}
