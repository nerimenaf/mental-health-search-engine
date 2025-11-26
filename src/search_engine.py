# src/search_engine.py
import json
import time
import math
from collections import Counter, defaultdict
from pathlib import Path

from indexer import preprocess  # reuse same preprocessing

INDEX_PATH = Path("index/index.json")
METADATA_PATH = Path("data/documents/metadata.json")

# load index
with open(INDEX_PATH, encoding="utf-8") as f:
    index_data = json.load(f)

INVERTED_INDEX = index_data["inverted_index"]          # term -> {doc_id: tf}
IDF = index_data["idf"]                                # term -> idf
DOC_LENGTHS = {k: int(v) for k, v in index_data["doc_lengths"].items()}  # doc_id -> length
N = index_data["N"]

# stats for web/statistics
VOCAB_SIZE = len(INVERTED_INDEX)
AVG_DOC_LENGTH = sum(DOC_LENGTHS.values()) / N

# load metadata
with open(METADATA_PATH, encoding="utf-8") as f:
    metadata_list = json.load(f)
METADATA = {m["doc_id"]: m for m in metadata_list}


# ---------- TF-IDF MODEL ----------

def score_tfidf(tokens):
    """Compute TF-IDF dot product scores."""
    q_tf = Counter(tokens)
    scores = defaultdict(float)

    for term, q_freq in q_tf.items():
        if term not in INVERTED_INDEX:
            continue
        idf = IDF.get(term, 0.0)
        w_q = q_freq * idf  # query weight

        posting = INVERTED_INDEX[term]  # {doc_id: tf}
        for doc_id, d_freq in posting.items():
            w_d = d_freq * idf  # doc weight
            scores[doc_id] += w_q * w_d

    return scores


# ---------- BM25 MODEL ----------

def score_bm25(tokens, k1=1.5, b=0.75):
    """
    Compute BM25 scores.
    BM25 formula:
      score(d, q) = sum over terms t in q of
        idf(t) * ( tf(t,d) * (k1 + 1) ) / ( tf(t,d) + k1 * (1 - b + b * |d|/avgdl) )
    """
    scores = defaultdict(float)
    unique_terms = set(tokens)

    for term in unique_terms:
        if term not in INVERTED_INDEX:
            continue
        idf = IDF.get(term, 0.0)
        posting = INVERTED_INDEX[term]  # {doc_id: tf}

        for doc_id, tf_td in posting.items():
            dl = DOC_LENGTHS[doc_id]
            denom = tf_td + k1 * (1 - b + b * dl / AVG_DOC_LENGTH)
            score = idf * (tf_td * (k1 + 1)) / denom
            scores[doc_id] += score

    return scores


# ---------- GENERIC SEARCH FUNCTION ----------

def search(query: str, k: int = 5, model: str = "tfidf"):
    """
    model: "tfidf" or "bm25"
    """
    start_time = time.time()

    tokens = preprocess(query)
    if not tokens:
        return [], 0.0

    if model.lower() == "bm25":
        scores = score_bm25(tokens)
    else:
        scores = score_tfidf(tokens)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    elapsed = time.time() - start_time
    return ranked, elapsed