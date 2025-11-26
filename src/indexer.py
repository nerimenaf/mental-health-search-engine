import re
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# download stopwords if not already done
nltk.download("stopwords", quiet=True)

DATA_DIR = Path("data")
DOC_DIR = DATA_DIR / "documents"
METADATA_PATH = DOC_DIR / "metadata.json"
INDEX_DIR = Path("index")
INDEX_PATH = INDEX_DIR / "index.json"

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess(text: str):
    # lowercase
    text = text.lower()
    # keep only letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    # remove stopwords
    tokens = [t for t in tokens if t not in stop_words]
    # stemming
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens

def build_index():
    with open(METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)

    inverted_index = defaultdict(dict)  # term -> {doc_id: tf}
    doc_lengths = {}                    # doc_id -> number of tokens
    N = len(metadata)                   # number of documents

    for meta in metadata:
        doc_id = meta["doc_id"]
        path = meta["path"]

        with open(path, encoding="utf-8") as f:
            text = f.read()

        tokens = preprocess(text)
        doc_lengths[doc_id] = len(tokens)
        tf = Counter(tokens)  # term frequency in this document

        for term, freq in tf.items():
            inverted_index[term][doc_id] = freq

    # compute IDF
    idf = {}
    for term, posting in inverted_index.items():
        df = len(posting)  # document frequency
        idf[term] = math.log((N + 1) / (df + 1)) + 1  # smoothed IDF

    INDEX_DIR.mkdir(exist_ok=True)

    # convert defaultdict to normal dict for JSON
    index_data = {
        "inverted_index": {term: posting for term, posting in inverted_index.items()},
        "idf": idf,
        "doc_lengths": doc_lengths,
        "N": N,
    }

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index_data, f)

    print(f"Index built for {N} documents and saved to {INDEX_PATH}")

if __name__ == "__main__":
    build_index()