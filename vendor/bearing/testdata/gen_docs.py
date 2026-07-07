# SPDX-License-Identifier: Apache-2.0

import argparse, os, random

parser = argparse.ArgumentParser(description="Generate test documents for indexing")
parser.add_argument("-n", type=int, default=30, help="number of documents to generate (default: 30)")
args = parser.parse_args()
num_docs = args.n

out_dir = "/tmp/perf-docs"
if os.path.exists(out_dir):
    import shutil
    shutil.rmtree(out_dir)
os.makedirs(out_dir)

nouns = ["system", "network", "algorithm", "database", "framework", "protocol", "interface", "architecture",
         "processor", "memory", "storage", "compiler", "runtime", "kernel", "module", "library",
         "function", "variable", "structure", "buffer", "cache", "pipeline", "thread", "process",
         "signal", "socket", "stream", "channel", "queue", "stack", "tree", "graph",
         "table", "index", "record", "field", "query", "transaction", "session", "connection",
         "server", "client", "request", "response", "packet", "header", "payload", "message",
         "forest", "mountain", "river", "ocean", "valley", "desert", "island", "continent",
         "species", "population", "ecosystem", "habitat", "migration", "evolution", "adaptation", "biodiversity",
         "civilization", "empire", "dynasty", "revolution", "conquest", "treaty", "alliance", "colony",
         "philosophy", "mathematics", "physics", "chemistry", "biology", "astronomy", "geology", "engineering",
         "democracy", "republic", "monarchy", "federation", "constitution", "legislature", "judiciary", "sovereignty",
         "economy", "currency", "commerce", "industry", "agriculture", "manufacturing", "innovation", "enterprise"]

verbs = ["processes", "transforms", "analyzes", "computes", "generates", "allocates", "optimizes", "executes",
         "transmits", "receives", "validates", "encrypts", "decrypts", "compresses", "decompresses", "serializes",
         "implements", "extends", "overrides", "initializes", "configures", "deploys", "monitors", "scales",
         "discovers", "explores", "investigates", "examines", "demonstrates", "illustrates", "represents", "describes",
         "establishes", "maintains", "develops", "produces", "distributes", "manages", "coordinates", "facilitates"]

adjectives = ["efficient", "robust", "scalable", "distributed", "concurrent", "parallel", "asynchronous", "synchronous",
              "persistent", "volatile", "immutable", "dynamic", "static", "abstract", "concrete", "generic",
              "ancient", "modern", "traditional", "contemporary", "revolutionary", "fundamental", "advanced", "primitive",
              "complex", "simple", "elegant", "powerful", "flexible", "reliable", "secure", "transparent",
              "massive", "compact", "rapid", "gradual", "systematic", "comprehensive", "innovative", "conventional"]

adverbs = ["efficiently", "rapidly", "seamlessly", "automatically", "dynamically", "recursively", "iteratively",
           "concurrently", "precisely", "significantly", "substantially", "fundamentally", "systematically", "comprehensively"]

connectors = ["however", "furthermore", "moreover", "additionally", "consequently", "therefore", "nevertheless",
              "meanwhile", "subsequently", "specifically", "particularly", "essentially", "ultimately", "accordingly"]

topics = [
    "computer science", "distributed systems", "machine learning", "operating systems",
    "network protocols", "database design", "compiler theory", "cryptography",
    "ancient history", "medieval warfare", "renaissance art", "industrial revolution",
    "marine biology", "quantum physics", "organic chemistry", "evolutionary biology",
    "economic theory", "political philosophy", "urban planning", "environmental science",
    "space exploration", "artificial intelligence", "robotics", "nanotechnology",
    "world geography", "cultural anthropology", "linguistic theory", "cognitive science",
    "musical composition", "architectural design", "literary criticism", "film studies"
]

random.seed(42)

def generate_sentence():
    patterns = [
        f"The {random.choice(adjectives)} {random.choice(nouns)} {random.choice(verbs)} the {random.choice(adjectives)} {random.choice(nouns)}",
        f"{random.choice(connectors).capitalize()}, the {random.choice(nouns)} {random.choice(verbs)} {random.choice(adverbs)}",
        f"A {random.choice(adjectives)} {random.choice(nouns)} {random.choice(verbs)} each {random.choice(nouns)} in the {random.choice(nouns)}",
        f"The {random.choice(nouns)} and the {random.choice(nouns)} {random.choice(verbs)} the {random.choice(adjectives)} {random.choice(nouns)} {random.choice(adverbs)}",
        f"When the {random.choice(nouns)} {random.choice(verbs)} the {random.choice(nouns)}, the {random.choice(adjectives)} {random.choice(nouns)} {random.choice(verbs)} {random.choice(adverbs)}",
        f"Each {random.choice(adjectives)} {random.choice(nouns)} {random.choice(adverbs)} {random.choice(verbs)} multiple {random.choice(nouns)}s within the {random.choice(nouns)}",
        f"By examining the {random.choice(nouns)}, researchers can understand how the {random.choice(adjectives)} {random.choice(nouns)} {random.choice(verbs)} the {random.choice(nouns)}",
    ]
    return random.choice(patterns) + ". "

def generate_paragraph():
    return " ".join(generate_sentence() for _ in range(random.randint(6, 12)))

def generate_document(topic, doc_id):
    paragraphs = []
    paragraphs.append(f"Chapter {doc_id}: {topic.title()}\n\n")
    for section in range(random.randint(8, 14)):
        paragraphs.append(f"Section {section + 1}\n\n")
        for _ in range(random.randint(3, 6)):
            paragraphs.append(generate_paragraph() + "\n\n")
    return "".join(paragraphs)

width = len(str(num_docs))
total_bytes = 0
for i in range(num_docs):
    topic = topics[i % len(topics)]
    content = generate_document(topic, i + 1)
    while len(content) < 75000:
        content += generate_paragraph() + "\n\n"
    slug = topic.replace(" ", "_")
    fname = f"doc_{i+1:0{width}d}_{slug}.txt"
    path = os.path.join(out_dir, fname)
    with open(path, "w") as f:
        f.write(content)
    total_bytes += len(content)

print(f"Generated {num_docs} files in {out_dir}")
print(f"Total size: {total_bytes:,} bytes ({total_bytes/1024/1024:.1f} MB)")
