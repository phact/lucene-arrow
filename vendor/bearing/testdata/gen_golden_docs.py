#!/usr/bin/env python3
"""Generate a small, deterministic corpus for golden index summary testing.

Produces 15 documents in testdata/golden-docs/ with varied content lengths
and recognizable text. The output is fully deterministic (fixed seed) so the
golden summary can be checked in and compared across runs.

Filenames use the pattern "doc_NNN_topic.txt" so that IndexAllFields can parse
the doc number for sparse doc values testing (even-numbered docs get sparse_count).

The generated files are checked into testdata/golden-docs/ — only re-run this
if you need to regenerate.
"""

import os
import random

random.seed(12345)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "golden-docs")

# Varied topics for recognizable content
TOPICS = [
    "algorithms",
    "databases",
    "networks",
    "security",
    "compilers",
    "graphics",
    "systems",
    "testing",
    "storage",
    "analysis",
    "robotics",
    "language",
    "quantum",
    "biology",
    "climate",
]

BASE_SENTENCES = [
    "Sorting algorithms provide efficient ways to organize data structures.",
    "Relational databases use SQL to query and manipulate structured data.",
    "Computer networks transmit packets across routers and switches.",
    "Encryption algorithms protect sensitive data during transmission.",
    "Compilers transform source code into executable machine instructions.",
    "Rendering engines process vertices and fragments for display output.",
    "Operating systems manage hardware resources and process scheduling.",
    "Automated testing validates software behavior against expected results.",
    "Storage systems manage persistent data across multiple disk drives.",
    "Data analysis extracts meaningful patterns from large datasets.",
    "Robotic systems integrate sensors and actuators for autonomous control.",
    "Natural language processing enables machines to understand human text.",
    "Quantum computing leverages superposition for parallel computation.",
    "Computational biology models protein folding and gene expression.",
    "Climate models simulate atmospheric and oceanic circulation patterns.",
]

EXTRA_SENTENCES = [
    "Performance optimization requires careful profiling and measurement.",
    "Distributed systems coordinate multiple processes across network boundaries.",
    "Machine learning models train on labeled data to make predictions.",
    "Version control systems track changes to source code over time.",
    "Cache hierarchies reduce memory access latency in modern processors.",
    "Functional programming emphasizes immutable data and pure functions.",
    "Container orchestration automates deployment and scaling of applications.",
    "Graph algorithms solve problems involving nodes and edges efficiently.",
    "Type systems prevent entire classes of programming errors at compile time.",
    "Parallel processing divides work across multiple computational units.",
]


def generate_docs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i in range(15):
        topic = TOPICS[i]
        # Use topic_NNN format so IndexAllFields.parseDocNum can extract the
        # number from the last underscore segment of the title
        filename = f"{topic}_{i + 1:03d}.txt"

        # Vary document length: 2-6 sentences
        num_extra = random.randint(1, 5)
        extras = random.sample(EXTRA_SENTENCES, num_extra)

        lines = [BASE_SENTENCES[i]] + extras
        content = " ".join(lines) + "\n"

        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "w") as f:
            f.write(content)

    print(f"Generated {15} documents in {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_docs()
