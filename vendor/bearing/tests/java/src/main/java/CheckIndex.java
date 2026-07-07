// SPDX-License-Identifier: Apache-2.0

import java.io.PrintStream;
import java.nio.file.Paths;

import org.apache.lucene.store.FSDirectory;

/**
 * Validates index structural integrity using Lucene's built-in CheckIndex.
 *
 * Unlike VerifyIndex (which checks field-specific expectations), this only
 * validates that the index is structurally sound and readable by Lucene.
 *
 * Usage: CheckIndex <index-dir>
 * Exit code 0 = clean, non-zero = corrupt or unreadable.
 */
public class CheckIndex {
    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            System.err.println("Usage: CheckIndex <index-dir>");
            System.exit(1);
        }

        try (FSDirectory dir = FSDirectory.open(Paths.get(args[0]));
             org.apache.lucene.index.CheckIndex checker = new org.apache.lucene.index.CheckIndex(dir)) {

            checker.setInfoStream(new PrintStream(System.out));
            org.apache.lucene.index.CheckIndex.Status status = checker.checkIndex();

            if (status.clean) {
                System.out.println("\nCheckIndex: OK");
            } else {
                System.err.println("\nCheckIndex: FAILED — index is corrupt");
                System.exit(1);
            }
        }
    }
}
