#!/bin/bash
# Golden summary test: verifies Rust's generate_summary produces identical output
# to the checked-in golden summary when reading indexes from both Java and Rust,
# in both non-compound and compound file modes.
#
# Sub-tests:
#   1. Java write (non-compound) → Rust read
#   2. Rust write (non-compound) → Rust read
#   3. Java write (compound) → Rust read
#   4. Rust write (compound) → Rust read
#
# Usage: ./tests/e2e_golden.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
GOLDEN_DOCS="$PROJECT_DIR/testdata/golden-docs"
GOLDEN_DOCS_ABS=$(cd "$GOLDEN_DOCS" && pwd)
GOLDEN_SUMMARY="$PROJECT_DIR/testdata/golden-summary.json"
JAVA_DIR="$PROJECT_DIR/tests/java"

FAILED=0
TEMP_DIRS=()

cleanup() {
    for d in "${TEMP_DIRS[@]}"; do
        rm -rf "$d"
    done
}
trap cleanup EXIT

make_temp_dir() {
    local d
    d=$(mktemp -d)
    TEMP_DIRS+=("$d")
    echo "$d"
}

# Compare a generated summary against the golden file
check_summary() {
    local label="$1"
    local index_dir="$2"
    local actual
    actual=$(mktemp)
    TEMP_DIRS+=("$actual")

    "$PROJECT_DIR/target/debug/generate_summary" -index "$index_dir" > "$actual"

    if diff -u "$GOLDEN_SUMMARY" "$actual" > /dev/null 2>&1; then
        echo "PASSED: $label"
    else
        echo "FAILED: $label"
        echo ""
        diff -u "$GOLDEN_SUMMARY" "$actual" || true
        FAILED=1
    fi
}

echo "Building Rust binaries..."
cargo build --bin indexfiles --bin generate_summary --manifest-path "$PROJECT_DIR/Cargo.toml" 2>&1 | tail -1

echo ""
echo "========================================"
echo "  Golden Summary Test"
echo "========================================"

# --- Java write (non-compound) → Rust read ---

echo ""
echo "--- Java write (non-compound) → Rust read ---"
INDEX_DIR=$(make_temp_dir)
"$JAVA_DIR/gradlew" -p "$JAVA_DIR" -q indexAllFields \
    -PdocsDir="$GOLDEN_DOCS_ABS" \
    -PindexDir="$INDEX_DIR" \
    > /dev/null 2>&1
check_summary "Java non-compound" "$INDEX_DIR"

# --- Rust write (non-compound) → Rust read ---

echo ""
echo "--- Rust write (non-compound) → Rust read ---"
INDEX_DIR=$(make_temp_dir)
"$PROJECT_DIR/target/debug/indexfiles" \
    -docs "$GOLDEN_DOCS" \
    -index "$INDEX_DIR" \
    --threads 1 \
    > /dev/null 2>&1
check_summary "Rust non-compound" "$INDEX_DIR"

# --- Java write (compound) → Rust read ---

echo ""
echo "--- Java write (compound) → Rust read ---"
INDEX_DIR=$(make_temp_dir)
"$JAVA_DIR/gradlew" -p "$JAVA_DIR" -q indexAllFields \
    -PdocsDir="$GOLDEN_DOCS_ABS" \
    -PindexDir="$INDEX_DIR" \
    -Pcompound=true \
    > /dev/null 2>&1
check_summary "Java compound" "$INDEX_DIR"

# --- Rust write (compound) → Rust read ---

echo ""
echo "--- Rust write (compound) → Rust read ---"
INDEX_DIR=$(make_temp_dir)
"$PROJECT_DIR/target/debug/indexfiles" \
    -docs "$GOLDEN_DOCS" \
    -index "$INDEX_DIR" \
    --compound \
    --threads 1 \
    > /dev/null 2>&1
check_summary "Rust compound" "$INDEX_DIR"

# --- Result ---

echo ""
if [ $FAILED -eq 0 ]; then
    echo "PASSED"
    exit 0
else
    echo "FAILED"
    exit 1
fi
