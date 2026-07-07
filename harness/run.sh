#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Golden-file generation + CheckIndex gate (SPEC §12). Requires JDK 21+.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
LUCENE_VERSION=10.3.2
LIB="$HERE/lib"
CORE_JAR="$LIB/lucene-core-$LUCENE_VERSION.jar"

usage() {
    echo "usage: $0 golden <output-dir> | check <segment-dir>" >&2
    exit 2
}

[ $# -eq 2 ] || usage

java_major=$(java -version 2>&1 | sed -nE 's/.*version "([0-9]+).*/\1/p' | head -1)
if [ "${java_major:-0}" -lt 21 ]; then
    echo "error: Lucene $LUCENE_VERSION requires JDK 21+, found ${java_major:-none}" >&2
    exit 1
fi

if [ ! -f "$CORE_JAR" ]; then
    mkdir -p "$LIB"
    echo "fetching lucene-core $LUCENE_VERSION from Maven Central..."
    curl -fsSL -o "$CORE_JAR" \
        "https://repo1.maven.org/maven2/org/apache/lucene/lucene-core/$LUCENE_VERSION/lucene-core-$LUCENE_VERSION.jar"
fi

case "$1" in
    golden)
        mkdir -p "$HERE/build"
        javac -cp "$CORE_JAR" -d "$HERE/build" "$HERE/src/GenerateGolden.java"
        java -cp "$CORE_JAR:$HERE/build" GenerateGolden "$2"
        ;;
    check)
        # Exit code 0 = clean index; anything else fails the gate (SPEC §10.5).
        java --add-modules jdk.incubator.vector -cp "$CORE_JAR" \
            org.apache.lucene.index.CheckIndex "$2" -level 2
        ;;
    *) usage ;;
esac
