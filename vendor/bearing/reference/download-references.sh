#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Downloads reference source code into this directory.
# - Apache Lucene 10.3.2: canonical source for the Rust port
# - Assertables 9.8.6: assertion macros used in tests

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Apache Lucene -----------------------------------------------------------

LUCENE_VERSION="10.3.2"
LUCENE_DIR="$SCRIPT_DIR/lucene-${LUCENE_VERSION}"
LUCENE_TARBALL="lucene-${LUCENE_VERSION}-src.tgz"
LUCENE_URL="https://archive.apache.org/dist/lucene/java/${LUCENE_VERSION}/${LUCENE_TARBALL}"

if [ -d "$LUCENE_DIR" ]; then
    echo "Lucene $LUCENE_VERSION already exists at $LUCENE_DIR"
else
    echo "Downloading Lucene $LUCENE_VERSION source..."
    curl -fSL -o "$SCRIPT_DIR/$LUCENE_TARBALL" "$LUCENE_URL"
    echo "Extracting..."
    tar -xzf "$SCRIPT_DIR/$LUCENE_TARBALL" -C "$SCRIPT_DIR"
    rm "$SCRIPT_DIR/$LUCENE_TARBALL"
    echo "Lucene $LUCENE_VERSION source is ready."
fi

# --- Assertables --------------------------------------------------------------

ASSERTABLES_VERSION="9.8.6"
ASSERTABLES_DIR="$SCRIPT_DIR/assertables"
ASSERTABLES_TARBALL="assertables-${ASSERTABLES_VERSION}.crate"
ASSERTABLES_URL="https://static.crates.io/crates/assertables/${ASSERTABLES_VERSION}/download"

if [ -d "$ASSERTABLES_DIR" ]; then
    echo "Assertables $ASSERTABLES_VERSION already exists at $ASSERTABLES_DIR"
else
    echo "Downloading Assertables $ASSERTABLES_VERSION source..."
    curl -fSL -o "$SCRIPT_DIR/$ASSERTABLES_TARBALL" "$ASSERTABLES_URL"
    echo "Extracting..."
    mkdir -p "$ASSERTABLES_DIR"
    tar -xzf "$SCRIPT_DIR/$ASSERTABLES_TARBALL" -C "$ASSERTABLES_DIR" --strip-components=1
    rm "$SCRIPT_DIR/$ASSERTABLES_TARBALL"
    echo "Assertables $ASSERTABLES_VERSION source is ready."
fi

echo "Done. All reference sources are ready."
