#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Unified E2E test for the Bearing indexer.
#
# Uses testdata/impact-docs (150 docs) — enough for impact blocks (128+ threshold)
# and all field types. Flow:
#   1. Build Rust indexfiles binary
#   2. Index impact-docs → verify expected files on disk
#   3. Re-index same directory (idempotency check)
#   4. Index with --compound → verify .cfs/.cfe exist
#   5. Run VerifyIndex on Rust index
#   6. Run Lucene CheckIndex on Rust index
#   7. Run VerifyImpacts on Rust index
#   8. Multi-segment, multi-thread, and compound variants with VerifyIndex
#   9. Golden summary test
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCS_DIR="$PROJECT_DIR/testdata/impact-docs"
DOC_COUNT=150

GRADLE="$SCRIPT_DIR/java/gradlew --project-dir=$SCRIPT_DIR/java --quiet"

INDEX_DIR="$(mktemp -d)"
COMPOUND_DIR="$(mktemp -d)"
trap 'rm -rf "$INDEX_DIR" "$COMPOUND_DIR"' EXIT

PASSED=0
FAILED=0
FAILURES=()

run_test() {
    local name="$1"
    echo ""
    echo "========================================"
    echo "  $name"
    echo "========================================"
}

pass() {
    echo "PASSED"
    PASSED=$((PASSED + 1))
}

fail() {
    echo "FAILED: $1"
    FAILED=$((FAILED + 1))
    FAILURES+=("$2")
}

# --- Build ---
echo "Building Rust indexfiles binary..."
cargo build --bin indexfiles --manifest-path "$PROJECT_DIR/Cargo.toml" 2>&1
INDEXFILES="$PROJECT_DIR/target/debug/indexfiles"

# --- 1. Index impact-docs ---
run_test "Index impact-docs (non-compound)"
"$INDEXFILES" -docs "$DOCS_DIR" -index "$INDEX_DIR"

EXPECTED_FILES="segments_1 _0.si _0.fnm _0_Lucene103_0.doc _0_Lucene103_0.pos _0_Lucene103_0.pay _0_Lucene103_0.tim _0_Lucene103_0.tip _0.fdt _0.fdm _0.fdx _0.kdm _0.kdi _0.kdd _0.nvm _0.nvd _0.tvd _0.tvx _0.tvm"
ALL_FOUND=true
for f in $EXPECTED_FILES; do
    if [ ! -f "$INDEX_DIR/$f" ]; then
        fail "expected file '$f' not found" "Index files on disk"
        ALL_FOUND=false
        break
    fi
    echo "  found: $f ($(stat --format='%s' "$INDEX_DIR/$f") bytes)"
done
# Verify compound files do NOT exist
if $ALL_FOUND; then
    for f in _0.cfs _0.cfe; do
        if [ -f "$INDEX_DIR/$f" ]; then
            fail "compound file '$f' should not exist in non-compound mode" "Index files on disk"
            ALL_FOUND=false
            break
        fi
    done
fi
if $ALL_FOUND; then
    pass
fi

# --- 2. Re-index (idempotency) ---
run_test "Re-index over existing index"
"$INDEXFILES" -docs "$DOCS_DIR" -index "$INDEX_DIR"
ALL_FOUND=true
for f in $EXPECTED_FILES; do
    if [ ! -f "$INDEX_DIR/$f" ]; then
        fail "expected file '$f' not found after re-index" "Re-index idempotency"
        ALL_FOUND=false
        break
    fi
done
if $ALL_FOUND; then
    pass
fi

# --- 3. Compound format ---
run_test "Index with --compound"
"$INDEXFILES" -docs "$DOCS_DIR" -index "$COMPOUND_DIR" --compound
ALL_FOUND=true
for f in _0.cfs _0.cfe; do
    if [ ! -f "$COMPOUND_DIR/$f" ]; then
        fail "expected compound file '$f' not found" "Compound format"
        ALL_FOUND=false
        break
    fi
    echo "  found: $f ($(stat --format='%s' "$COMPOUND_DIR/$f") bytes)"
done
if $ALL_FOUND; then
    pass
fi

# --- 4. VerifyIndex ---
run_test "Java VerifyIndex on Rust index"
if $GRADLE verifyIndex -PindexDir="$INDEX_DIR" -PdocCount="$DOC_COUNT" 2>&1; then
    pass
else
    fail "VerifyIndex failed" "VerifyIndex"
fi

# --- 5. CheckIndex ---
run_test "Lucene CheckIndex on Rust index"
if $GRADLE runJava -PmainClass=CheckIndex -Pargs="$INDEX_DIR" 2>&1; then
    pass
else
    fail "CheckIndex failed" "CheckIndex"
fi

# --- 6. VerifyImpacts ---
run_test "Java VerifyImpacts on Rust index"
if $GRADLE verifyImpacts -PindexDir="$INDEX_DIR" 2>&1; then
    pass
else
    fail "VerifyImpacts failed" "VerifyImpacts"
fi

# --- 7. Configuration variants with VerifyIndex ---
run_variant() {
    local name="$1"
    shift
    local args=("$@")

    run_test "$name"

    local idx_dir
    idx_dir="$(mktemp -d)"

    "$INDEXFILES" -docs "$DOCS_DIR" -index "$idx_dir" "${args[@]}"

    if [ ! -f "$idx_dir/segments_1" ]; then
        fail "segments_1 not found" "$name"
        rm -rf "$idx_dir"
        return
    fi

    if $GRADLE verifyIndex -PindexDir="$idx_dir" -PdocCount="$DOC_COUNT" 2>&1; then
        pass
    else
        fail "VerifyIndex rejected the index" "$name"
    fi

    rm -rf "$idx_dir"
}

run_variant "Multi-segment" --max-buffered-docs 10
run_variant "Multi-thread" --threads 2
run_variant "Multi-thread + multi-segment" --threads 2 --max-buffered-docs 10
run_variant "Compound + multi-segment" --max-buffered-docs 10 --compound

# --- 8. Golden Summary ---
run_test "Golden index summary"
if "$SCRIPT_DIR/e2e_golden.sh" 2>&1; then
    pass
else
    fail "Golden summary mismatch" "GoldenSummary"
fi

# --- Summary ---
echo ""
echo "========================================"
echo "  E2E Summary: $PASSED passed, $FAILED failed"
if [ $FAILED -gt 0 ]; then
    echo "  Failed:"
    for f in "${FAILURES[@]}"; do
        echo "    - $f"
    done
fi
echo "========================================"

exit $FAILED
