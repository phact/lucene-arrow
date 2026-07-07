// SPDX-License-Identifier: Apache-2.0

import java.nio.file.Paths;
import java.util.List;

import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.Impact;
import org.apache.lucene.index.Impacts;
import org.apache.lucene.index.ImpactsEnum;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.FSDirectory;

/**
 * Verifies that a Lucene index contains proper competitive impact data.
 *
 * For terms with docFreq >= 128 (at least one full block), reads impacts via
 * ImpactsEnum and checks that blocks contain proper norm values and multiple
 * impacts where expected.
 *
 * Usage: java VerifyImpacts <index-dir>
 *
 * Exits 0 if impacts look correct, non-zero if all blocks have exactly 1
 * impact with norm=1 (indicating the writer only emits minimal impacts).
 */
public class VerifyImpacts {

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("Usage: java VerifyImpacts <index-dir>");
            System.exit(1);
        }

        String indexDir = args[0];
        System.out.println("=== VerifyImpacts: opening " + indexDir + " ===");

        int termsChecked = 0;
        int termsWithMultipleImpacts = 0;
        int termsWithNonOneNorm = 0;
        int totalBlocks = 0;
        int blocksWithMultipleImpacts = 0;
        int blocksWithNonOneNorm = 0;

        try (IndexReader reader = DirectoryReader.open(FSDirectory.open(Paths.get(indexDir)))) {
            for (LeafReaderContext ctx : reader.leaves()) {
                LeafReader leaf = ctx.reader();
                FieldInfos fieldInfos = leaf.getFieldInfos();

                for (FieldInfo fi : fieldInfos) {
                    // Only check fields that have frequencies
                    if (fi.getIndexOptions().compareTo(IndexOptions.DOCS_AND_FREQS) < 0) {
                        continue;
                    }

                    Terms terms = leaf.terms(fi.name);
                    if (terms == null) {
                        continue;
                    }

                    TermsEnum termsEnum = terms.iterator();
                    while (termsEnum.next() != null) {
                        int docFreq = termsEnum.docFreq();
                        // Only check terms with at least one full 128-doc block
                        if (docFreq < 128) {
                            continue;
                        }

                        termsChecked++;
                        boolean termHasMultipleImpacts = false;
                        boolean termHasNonOneNorm = false;

                        // Request FREQS so the postings reader creates an impacts-capable enum
                        ImpactsEnum impactsEnum = termsEnum.impacts(PostingsEnum.FREQS);

                        // Position at the first doc, then walk block boundaries
                        int doc = impactsEnum.nextDoc();
                        while (doc != DocIdSetIterator.NO_MORE_DOCS) {
                            impactsEnum.advanceShallow(doc);
                            Impacts impacts = impactsEnum.getImpacts();

                            for (int level = 0; level < impacts.numLevels(); level++) {
                                List<Impact> levelImpacts = impacts.getImpacts(level);
                                totalBlocks++;

                                if (levelImpacts.size() > 1) {
                                    blocksWithMultipleImpacts++;
                                    termHasMultipleImpacts = true;
                                }

                                for (Impact impact : levelImpacts) {
                                    if (impact.norm != 1) {
                                        blocksWithNonOneNorm++;
                                        termHasNonOneNorm = true;
                                        break;
                                    }
                                }
                            }

                            // Jump past this block to the next one
                            int docIdUpTo = impacts.getDocIdUpTo(0);
                            if (docIdUpTo == DocIdSetIterator.NO_MORE_DOCS) {
                                break;
                            }
                            doc = impactsEnum.advance(docIdUpTo + 1);
                        }

                        if (termHasMultipleImpacts) {
                            termsWithMultipleImpacts++;
                        }
                        if (termHasNonOneNorm) {
                            termsWithNonOneNorm++;
                        }
                    }
                }
            }
        }

        System.out.println("Terms checked (docFreq >= 128): " + termsChecked);
        System.out.println("Terms with multiple impacts in any block: " + termsWithMultipleImpacts);
        System.out.println("Terms with non-1 norm in any block: " + termsWithNonOneNorm);
        System.out.println("Total blocks examined: " + totalBlocks);
        System.out.println("Blocks with multiple impacts: " + blocksWithMultipleImpacts);
        System.out.println("Blocks with non-1 norm: " + blocksWithNonOneNorm);

        if (termsChecked == 0) {
            System.out.println("SKIPPED: no terms with docFreq >= 128 found (segments too small to verify impacts).");
            System.exit(0);
        }

        // The defect: all blocks have exactly 1 impact with norm=1
        // A correct writer should produce non-1 norms for fields that have norms,
        // since different documents have different field lengths.
        if (blocksWithNonOneNorm == 0) {
            System.err.println("FAIL: all " + totalBlocks + " blocks have norm=1 only.");
            System.err.println("This indicates the writer emits minimal impacts without real norms.");
            System.exit(1);
        }

        System.out.println("VerifyImpacts: PASSED");
    }
}
