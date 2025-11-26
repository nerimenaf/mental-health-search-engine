import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from indexer import preprocess  # reuse same preprocessing

INDEX_PATH = Path("index/index.json")
METADATA_PATH = Path("data/documents/metadata.json")

# load index
with open(INDEX_PATH, encoding="utf-8") as f:
    index_data = json.load(f)

INVERTED_INDEX = index_data["inverted_index"]
IDF = index_data["idf"]
DOC_LENGTHS = index_data["doc_lengths"]
N = index_data["N"]

# load metadata
with open(METADATA_PATH, encoding="utf-8") as f:
    metadata_list = json.load(f)
METADATA = {m["doc_id"]: m for m in metadata_list}

def search(query: str, k: int = 5):
    start_time = time.time()

    tokens = preprocess(query)
    if not tokens:
        return [], 0.0

    q_tf = Counter(tokens)
    scores = defaultdict(float)  # doc_id -> score

    for term, q_freq in q_tf.items():
        if term not in INVERTED_INDEX:
            continue
        idf = IDF.get(term, 0.0)
        w_q = q_freq * idf  # query weight

        posting = INVERTED_INDEX[term]  # dict: doc_id -> tf
        for doc_id, d_freq in posting.items():
            w_d = d_freq * idf  # document weight
            scores[doc_id] += w_q * w_d  # dot product

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    elapsed = time.time() - start_time
    return ranked, elapsed