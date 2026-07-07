# SPDX-License-Identifier: Apache-2.0
#
# Generates 150 small test documents for impact verification.
# Uses a fixed seed for deterministic output. The generated files are checked
# into testdata/impact-docs/ — only re-run this if you need to regenerate.
#
# These docs use a small vocabulary so that common terms appear in 128+ documents,
# which is the threshold for competitive impact blocks in Lucene's postings format.

import os
import random

random.seed(42)

script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, "impact-docs")
if os.path.exists(out_dir):
    import shutil

    shutil.rmtree(out_dir)
os.makedirs(out_dir)

nouns = [
    "system", "network", "algorithm", "database", "framework", "protocol",
    "processor", "memory", "storage", "compiler", "module", "library",
    "function", "variable", "structure", "buffer", "cache", "pipeline",
    "thread", "process", "signal", "stream", "channel", "queue",
]

adjectives = [
    "efficient", "robust", "scalable", "distributed", "concurrent",
    "persistent", "dynamic", "abstract", "generic", "complex",
    "simple", "elegant", "powerful", "flexible", "reliable",
]

verbs = [
    "processes", "transforms", "analyzes", "generates", "optimizes",
    "transmits", "validates", "configures", "monitors", "implements",
]

total = 0
for i in range(150):
    sentences = []
    for _ in range(8):
        s = (
            f"The {random.choice(adjectives)} {random.choice(nouns)} "
            f"{random.choice(verbs)} the {random.choice(adjectives)} "
            f"{random.choice(nouns)}."
        )
        sentences.append(s)
    content = " ".join(sentences)
    path = os.path.join(out_dir, f"doc_{i + 1:03d}.txt")
    with open(path, "w") as f:
        f.write(content + "\n")
    total += len(content) + 1

print(f"Generated 150 files in {out_dir}")
print(f"Total size: {total:,} bytes ({total / 1024:.1f} KB)")
