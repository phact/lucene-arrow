// SPDX-License-Identifier: Apache-2.0

import java.io.IOException;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.BinaryDocValuesField;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.DoubleField;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FloatField;
import org.apache.lucene.document.IntField;
import org.apache.lucene.document.KeywordField;
import org.apache.lucene.document.LongField;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.document.DoubleRange;
import org.apache.lucene.document.FeatureField;
import org.apache.lucene.document.FloatRange;
import org.apache.lucene.document.IntRange;
import org.apache.lucene.document.LatLonPoint;
import org.apache.lucene.document.LongRange;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.FSDirectory;

/**
 * Indexes documents with ALL field types for cross-validation with the Rust indexer.
 *
 * Usage: java IndexAllFields <docs-dir> <index-dir> [--threads N]
 */
public class IndexAllFields {

    /** FieldType matching Rust's text_field_with_term_vectors: indexed with positions and offsets, term vectors with positions and offsets. */
    private static final FieldType TV_TYPE;
    static {
        TV_TYPE = new FieldType();
        TV_TYPE.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS);
        TV_TYPE.setTokenized(true);
        TV_TYPE.setStoreTermVectors(true);
        TV_TYPE.setStoreTermVectorPositions(true);
        TV_TYPE.setStoreTermVectorOffsets(true);
        TV_TYPE.freeze();
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: java IndexAllFields <docs-dir> <index-dir> [--threads N]");
            System.exit(1);
        }

        Path docsDir = Paths.get(args[0]);
        Path indexDir = Paths.get(args[1]);
        int numThreads = 1;
        boolean useCompoundFile = false;

        for (int i = 2; i < args.length; i++) {
            if ("--threads".equals(args[i]) && i + 1 < args.length) {
                numThreads = Integer.parseInt(args[++i]);
            } else if ("--compound".equals(args[i])) {
                useCompoundFile = true;
            }
        }

        if (!Files.isDirectory(docsDir)) {
            System.err.println("Document directory '" + docsDir + "' does not exist");
            System.exit(1);
        }

        // Clean up existing index
        if (Files.exists(indexDir)) {
            Files.walk(indexDir)
                .filter(Files::isRegularFile)
                .forEach(p -> { try { Files.delete(p); } catch (IOException e) { /* ignore */ } });
        }
        Files.createDirectories(indexDir);

        System.out.println("Indexing to '" + indexDir + "' from '" + docsDir + "'"
            + (numThreads > 1 ? " (" + numThreads + " threads)" : "") + "...");

        // Collect all file paths first
        List<Path> filePaths = new ArrayList<>();
        Files.walkFileTree(docsDir, new SimpleFileVisitor<Path>() {
            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) {
                filePaths.add(file);
                return FileVisitResult.CONTINUE;
            }
        });

        IndexWriterConfig config = new IndexWriterConfig(new StandardAnalyzer());
        config.setUseCompoundFile(useCompoundFile);
        try (IndexWriter writer = new IndexWriter(FSDirectory.open(indexDir), config)) {
            if (numThreads <= 1) {
                for (Path file : filePaths) {
                    indexDoc(writer, file);
                }
            } else {
                ExecutorService executor = Executors.newFixedThreadPool(numThreads);
                for (Path file : filePaths) {
                    executor.submit(() -> {
                        try {
                            indexDoc(writer, file);
                        } catch (IOException e) {
                            System.err.println("Error indexing " + file + ": " + e.getMessage());
                        }
                    });
                }
                executor.shutdown();
                executor.awaitTermination(10, TimeUnit.MINUTES);
            }
        }

        System.out.println("Indexed " + filePaths.size() + " documents. Done.");
    }

    static void indexDoc(IndexWriter writer, Path file) throws IOException {
        String contents = Files.readString(file);
        long modified = Files.getLastModifiedTime(file).toMillis();
        long fileSize = Files.size(file);
        String fileName = file.getFileName().toString();
        String title = fileName.contains(".") ? fileName.substring(0, fileName.lastIndexOf('.')) : fileName;

        Document doc = new Document();

        // Same 3 fields as the standard indexer
        doc.add(new KeywordField("path", file.toString(), Field.Store.YES));
        doc.add(new LongField("modified", modified, Field.Store.NO));
        doc.add(new Field("contents", contents, TV_TYPE));

        // New field types
        doc.add(new StringField("title", title, Field.Store.YES));
        doc.add(new IntField("size", (int) fileSize, Field.Store.YES));
        doc.add(new FloatField("score", (fileSize % 100) / 10.0f, Field.Store.YES));
        doc.add(new DoubleField("rating", fileSize * 1.5, Field.Store.YES));
        doc.add(new StoredField("notes", "indexed by Java"));
        doc.add(new StoredField("extra_int", (int) (fileSize % 1000)));
        doc.add(new StoredField("extra_float", (float) (fileSize % 100) / 3.0f));
        doc.add(new StoredField("extra_double", fileSize * 0.123));

        // LatLonPoint
        double lat = 40.7128 + (fileSize % 10) * 0.01;
        double lon = -74.006 + (fileSize % 10) * 0.01;
        doc.add(new LatLonPoint("location", lat, lon));

        // Range fields
        doc.add(new IntRange("int_range", new int[]{(int) fileSize}, new int[]{(int) fileSize + 100}));
        doc.add(new LongRange("long_range", new long[]{fileSize}, new long[]{fileSize + 1000}));
        doc.add(new FloatRange("float_range", new float[]{fileSize / 10.0f}, new float[]{fileSize / 10.0f + 1.0f}));
        doc.add(new DoubleRange("double_range", new double[]{fileSize * 0.1}, new double[]{fileSize * 0.1 + 1.0}));

        // FeatureField
        doc.add(new FeatureField("features", "pagerank", (fileSize % 100) / 10.0f + 0.5f));
        doc.add(new FeatureField("features", "freshness", (fileSize % 50) / 5.0f + 1.0f));

        // Doc-values-only fields
        doc.add(new NumericDocValuesField("dv_count", fileSize));
        doc.add(new BinaryDocValuesField("dv_hash",
            new BytesRef(String.format("%016x", fileSize))));
        doc.add(new SortedDocValuesField("dv_category", new BytesRef(title)));
        doc.add(new SortedSetDocValuesField("dv_tag", new BytesRef(title)));
        doc.add(new SortedNumericDocValuesField("dv_priority", fileSize % 10));

        // Sparse doc values — only even-numbered docs (parsed from filename like "doc_003")
        int docNum = parseDocNum(title);
        if (docNum >= 0 && docNum % 2 == 0) {
            doc.add(new NumericDocValuesField("sparse_count", docNum * 100L));
        }

        writer.addDocument(doc);
        System.out.println("  indexed: " + file);
    }

    /**
     * Extracts a doc number from a title like "doc_003" or "science".
     * Returns -1 if no number can be parsed.
     */
    static int parseDocNum(String title) {
        int underscoreIdx = title.lastIndexOf('_');
        if (underscoreIdx < 0 || underscoreIdx == title.length() - 1) {
            return -1;
        }
        try {
            return Integer.parseInt(title.substring(underscoreIdx + 1));
        } catch (NumberFormatException e) {
            return -1;
        }
    }
}
