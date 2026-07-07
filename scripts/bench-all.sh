#!/usr/bin/env bash
#
# bench-all.sh — run the full lucene-arrow performance suite and collect
# results into one report. Degrades gracefully: CPU rows always run;
# GPU / cuVS / JVM rows run only when those are available.
#
# Scale knobs (env, defaults match the README numbers):
#   NUM_DOCS     numeric index for the read/write baseline   (16000000)
#   TEXT_DOCS    postings index for the scan baseline        (4000000)
#   CORPUS_DOCS  synthetic BM25 corpus size                  (300000)
#   CORPUS       use an existing line-per-doc corpus instead of generating
#   QUICK=1      shrink every scale for a fast smoke run
#
# A full default run generates multi-GB indexes and takes many minutes.
# Results are logged under target/bench-results/ and summarised at the end.

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ "${QUICK:-0}" = 1 ]; then
    NUM_DOCS="${NUM_DOCS:-1000000}"; TEXT_DOCS="${TEXT_DOCS:-400000}"; CORPUS_DOCS="${CORPUS_DOCS:-50000}"
fi
NUM_DOCS="${NUM_DOCS:-16000000}"
TEXT_DOCS="${TEXT_DOCS:-4000000}"
CORPUS_DOCS="${CORPUS_DOCS:-300000}"

OUT="$ROOT/target/bench-results"
mkdir -p "$OUT"

# --- capability detection ---------------------------------------------------
HAVE_GPU=0
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1 && HAVE_GPU=1

HAVE_CUVS=0
[ -d "$ROOT/.pixi/envs/default" ] && HAVE_CUVS=1
if [ "$HAVE_CUVS" = 1 ]; then
    export CONDA_PREFIX="$ROOT/.pixi/envs/default"
    export PATH="$CONDA_PREFIX/bin:$PATH"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$CONDA_PREFIX/lib"
fi

JAVA=""; JAVAC=""
if command -v java >/dev/null 2>&1 && java -version 2>&1 | grep -qE '"(21|2[2-9]|[3-9][0-9])'; then
    JAVA=java; JAVAC=javac
elif [ -x "$HOME/.jbang/cache/jdks/21/bin/java" ]; then
    JAVA="$HOME/.jbang/cache/jdks/21/bin/java"; JAVAC="$HOME/.jbang/cache/jdks/21/bin/javac"
fi
JAR="$ROOT/harness/lib/lucene-core-10.3.2.jar"
HAVE_JVM=0
[ -n "$JAVA" ] && [ -f "$JAR" ] && HAVE_JVM=1

echo "==================================================================="
echo " lucene-arrow bench-all"
echo "   GPU=$HAVE_GPU  cuVS=$HAVE_CUVS  JVM=$HAVE_JVM"
echo "   NUM_DOCS=$NUM_DOCS  TEXT_DOCS=$TEXT_DOCS  CORPUS_DOCS=$CORPUS_DOCS"
echo "   logs: $OUT"
echo "==================================================================="

RESULTS=()   # "label|logfile" in report order

run() { # run <label> <logfile> <cmd...>
    local label="$1" log="$OUT/$2"; shift 2
    printf '  %-34s ' "$label"
    if "$@" >"$log" 2>&1; then echo "done"; else echo "skip/fail"; fi
    RESULTS+=("$label|$log")
}
jvm() { "$JAVA" --add-modules jdk.incubator.vector -cp "$JAR:$OUT/classes" "$@"; }

# --- data prep (idempotent) -------------------------------------------------
echo; echo "[1/4] data prep"

CORPUS="${CORPUS:-$OUT/corpus.txt}"
if [ ! -f "$CORPUS" ]; then
    printf '  %-34s ' "gen corpus ($CORPUS_DOCS docs)"
    cargo run -q --release -p lucene-arrow-postings --example gen_corpus -- "$CORPUS" "$CORPUS_DOCS" \
        >/dev/null 2>&1 && echo "done" || echo "FAIL"
else
    echo "  corpus present: $CORPUS"
fi

if [ "$HAVE_JVM" = 1 ]; then
    mkdir -p "$OUT/classes"
    # Only the Bench* baselines — other harness files (VerifyJVector) need
    # jars we don't classpath here, and one failing file aborts javac.
    "$JAVAC" -cp "$JAR" -d "$OUT/classes" harness/src/Bench*.java 2>"$OUT/javac.log" \
        || echo "  (javac failed — see $OUT/javac.log; jvm rows will skip)"

    # BenchIngest both *generates* the numeric index and *is* the write baseline.
    if [ ! -d "$ROOT/harness/bench-index" ]; then
        printf '  %-34s ' "JVM ingest -> bench-index"
        jvm BenchIngest "$ROOT/harness/bench-index" "$NUM_DOCS" >"$OUT/jvm_write.log" 2>&1 && echo "done" || echo "FAIL"
        RESULTS+=("JVM write (BenchIngest)|$OUT/jvm_write.log")
    fi
    if [ ! -d "$ROOT/harness/bench-text" ]; then
        printf '  %-34s ' "JVM ingest -> bench-text"
        jvm BenchText ingest "$ROOT/harness/bench-text" "$TEXT_DOCS" >/dev/null 2>&1 && echo "done" || echo "FAIL"
    fi
    if [ ! -d "$OUT/md-jvm" ]; then
        printf '  %-34s ' "JVM ingest -> md corpus index"
        jvm BenchMdIngest "$CORPUS" "$OUT/md-jvm" >"$OUT/jvm_bm25_ingest.log" 2>&1 && echo "done" || echo "FAIL"
        RESULTS+=("JVM BM25 ingest (BenchMdIngest)|$OUT/jvm_bm25_ingest.log")
    fi
fi

# --- ours: CPU + GPU benches ------------------------------------------------
echo; echo "[2/4] rust benches"

run "cpu: ordmap"        b_ordmap.log   cargo bench -q -p lucene-arrow-cpu      --bench ordmap_bench
run "cpu: write (DoPut)" b_write.log    cargo bench -q -p lucene-arrow-flight   --bench write_bench
run "cpu: postings scan" b_csr.log      cargo bench -q -p lucene-arrow-postings --bench csr_bench
run "cpu: BM25 ingest"   b_bm25ing.log  cargo bench -q -p lucene-arrow-postings --bench bm25_ingest -- "$CORPUS"

if [ "$HAVE_GPU" = 1 ]; then
    for b in gpu_decode e2e_decode gpu_encode knn_scale postings_gpu; do
        run "gpu: $b" "b_$b.log" cargo bench -q -p lucene-arrow-gpu --features gpu --bench "$b"
    done
    run "gpu: scan_dir"     b_scan_dir.log  cargo bench -q -p lucene-arrow-gpu --features gpu --bench scan_dir
    run "gpu: bm25_query"   b_bm25q.log     cargo bench -q -p lucene-arrow-gpu --features gpu --bench bm25_query -- "$CORPUS"
    run "gpu: text ingest"  b_gpuingest.log cargo bench -q -p lucene-arrow-gpu --features gpu --bench gpu_ingest -- "$CORPUS"
else
    echo "  (no GPU — skipping gpu benches)"
fi

if [ "$HAVE_CUVS" = 1 ]; then
    run "cuvs: knn three-way" b_knn3.log  cargo bench -q -p lucene-arrow-gpu --features cuvs --bench knn_threeway
    run "cuvs: hnsw build"    b_hnsw.log  cargo bench -q -p lucene-arrow-gpu --features cuvs --bench hnsw_build
else
    echo "  (no pixi/libcuvs env — skipping cuvs benches)"
fi

# --- JVM baselines ----------------------------------------------------------
echo; echo "[3/4] jvm baselines"
if [ "$HAVE_JVM" = 1 ]; then
    [ -d "$ROOT/harness/bench-index" ] && \
        run "jvm: read scan"      j_scan.log    jvm BenchScan "$ROOT/harness/bench-index" f0 f1 f2 f3
    [ -d "$ROOT/harness/bench-text" ] && \
        run "jvm: postings scan"  j_pscan.log   jvm BenchText scan "$ROOT/harness/bench-text"
    run "jvm: HNSW indexing"      j_knn.log     jvm BenchKnnIngest "$OUT/j-knn" 200000 128
    # BM25 scoring baseline needs the query files bm25_query writes to /tmp.
    if [ -d "$OUT/md-jvm" ] && [ -f /tmp/bm25_queries.txt ]; then
        run "jvm: BM25 scoring (selective)" j_bm25q_sel.log jvm BenchBM25Query "$OUT/md-jvm" /tmp/bm25_queries.txt 10
        [ -f /tmp/bm25_queries_heavy.txt ] && \
            run "jvm: BM25 scoring (heavy)"  j_bm25q_hvy.log jvm BenchBM25Query "$OUT/md-jvm" /tmp/bm25_queries_heavy.txt 10
    fi
else
    echo "  (no JDK 21 + lucene jar — skipping jvm baselines)"
fi

# --- summary ----------------------------------------------------------------
echo; echo "[4/4] results"
echo "==================================================================="
# Metric keywords, plus `\|` so table rows (engine | build | ... ) print too.
METRIC='Gval/s|Gdocs/s|GB/s|MB/s|Mvals/s|Mdocs/s|Mpostings/s|Mrows/s|kdocs/s|qps|recall|docs in|queries in|TOTAL|round|\|'
for entry in "${RESULTS[@]}"; do
    label="${entry%%|*}"; log="${entry##*|}"
    echo; echo "▸ $label"
    lines=""
    [ -f "$log" ] && lines="$(grep -hE "$METRIC" "$log" 2>/dev/null)"
    if [ -n "$lines" ]; then
        echo "$lines" | sed 's/^/    /'
    elif [ -f "$log" ]; then
        grep -vE '^[[:space:]]*$' "$log" | tail -n 5 | sed 's/^/    /'   # fallback: last lines
    else
        echo "    (no log — $log)"
    fi
done
echo; echo "==================================================================="
echo "full per-bench logs in $OUT"
