// Query a Lucene index with a list of query strings and report per-query timing.
//
// Each line of the queries file is a single JSON object:
//
//   {"q": "<query string in Lucene syntax>", "msm": <int, optional, default 0>}
//
// `msm` (if > 0 and the parsed query is a BooleanQuery with at least msm SHOULD clauses)
// applies `BooleanQuery.Builder.setMinimumNumberShouldMatch(msm)` to that query.
//
// Usage (via gradle task):
//   ./tests/java/gradlew -p tests/java -q queryIndex \
//       -PindexDir=/tmp/index -PqueriesFile=/tmp/queries.jsonl [-PoutputFile=/tmp/results.txt]

import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopScoreDocCollectorManager;
import org.apache.lucene.store.FSDirectory;

public class QueryIndex {

    /** Parsed queries-file entry. */
    private static final class Entry {
        final String raw;   // original JSON line, used for the result line
        final String q;
        final int msm;
        Entry(String raw, String q, int msm) {
            this.raw = raw;
            this.q = q;
            this.msm = msm;
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: QueryIndex <index-dir> <queries-file> [output-file]");
            System.exit(1);
        }

        String indexDir = args[0];
        String queriesFile = args[1];
        String outputFile = args.length > 2 ? args[2] : null;

        List<Entry> queries = readQueries(queriesFile);

        try (DirectoryReader reader = DirectoryReader.open(FSDirectory.open(Paths.get(indexDir)))) {
            IndexSearcher searcher = new IndexSearcher(reader);
            QueryParser parser = new QueryParser("contents", new StandardAnalyzer(CharArraySet.EMPTY_SET));

            // Warm up: run all queries once to let JIT compile hot paths
            for (Entry e : queries) {
                Query query = applyMsm(parser.parse(e.q), e.msm);
                searcher.search(query, new TopScoreDocCollectorManager(10, Integer.MAX_VALUE));
            }

            // Collect results in memory — no I/O during timed section
            String[] results = new String[queries.size()];
            long startNanos = System.nanoTime();

            for (int i = 0; i < queries.size(); i++) {
                Entry e = queries.get(i);
                Query query = applyMsm(parser.parse(e.q), e.msm);
                TopDocs topDocs = searcher.search(query, new TopScoreDocCollectorManager(10, Integer.MAX_VALUE));

                StringBuilder sb = new StringBuilder();
                sb.append(String.format("%-40s hits=%-6d", e.raw, topDocs.totalHits.value()));
                for (ScoreDoc sd : topDocs.scoreDocs) {
                    sb.append(String.format("  doc=%-5d score=%.5f", sd.doc, sd.score));
                }
                results[i] = sb.toString();
            }

            long elapsedNanos = System.nanoTime() - startNanos;
            double elapsedMs = elapsedNanos / 1_000_000.0;
            double avgUs = elapsedNanos / 1_000.0 / queries.size();

            if (outputFile != null) {
                try (PrintWriter pw = new PrintWriter(outputFile)) {
                    for (String line : results) {
                        pw.println(line);
                    }
                }
            } else {
                for (String line : results) {
                    System.out.println(line);
                }
            }

            System.out.printf("Queried %d queries in %.2f ms%n", queries.size(), elapsedMs);
            System.out.printf("Average: %.2f µs/query%n", avgUs);
        }
    }

    private static List<Entry> readQueries(String path) throws Exception {
        List<Entry> out = new ArrayList<>();
        for (String raw : Files.readAllLines(Paths.get(path))) {
            String trimmed = raw.trim();
            if (trimmed.isEmpty()) continue;
            JsonObject obj = JsonParser.parseString(trimmed).getAsJsonObject();
            String q = obj.get("q").getAsString();
            int msm = obj.has("msm") ? obj.get("msm").getAsInt() : 0;
            out.add(new Entry(trimmed, q, msm));
        }
        return out;
    }

    /**
     * Applies a minimum-should-match constraint to a parsed query when it is a BooleanQuery
     * with at least `msm` SHOULD clauses. Other query shapes are returned unchanged.
     */
    private static Query applyMsm(Query parsed, int msm) {
        if (msm <= 0 || !(parsed instanceof BooleanQuery)) {
            return parsed;
        }
        BooleanQuery bq = (BooleanQuery) parsed;
        int shouldCount = 0;
        for (BooleanClause c : bq) {
            if (c.occur() == BooleanClause.Occur.SHOULD) shouldCount++;
        }
        if (shouldCount < msm) {
            return parsed;
        }
        BooleanQuery.Builder b = new BooleanQuery.Builder();
        for (BooleanClause c : bq) {
            b.add(c);
        }
        b.setMinimumNumberShouldMatch(msm);
        return b.build();
    }
}
