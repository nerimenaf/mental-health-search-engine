# src/evaluator.py
import csv
import sys
from pathlib import Path

from search_engine import search

GROUND_TRUTH_PATH = Path("eval/ground_truth.csv")

def load_ground_truth():
    gt = {}
    with open(GROUND_TRUTH_PATH, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row["query_id"].strip()
            query_text = row["query_text"].strip()
            doc_id = row["doc_id"].strip()
            rel = int(row["relevant"])

            if qid not in gt:
                gt[qid] = {"query": query_text, "relevant_docs": set()}
            if rel == 1:
                gt[qid]["relevant_docs"].add(doc_id)
    return gt

def evaluate(k: int = 5, model: str = "tfidf"):
    gt = load_ground_truth()
    results = []

    for qid, info in gt.items():
        query = info["query"]
        relevant_docs = info["relevant_docs"]

        retrieved, _ = search(query, k=k, model=model)
        retrieved_docs = [doc_id for doc_id, _ in retrieved]

        tp = len(set(retrieved_docs) & relevant_docs)
        fp = len(retrieved_docs) - tp
        fn = len(relevant_docs) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        results.append({
            "query_id": qid,
            "query": query,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    print(f"Evaluation - Model: {model}")
    print("Query_ID | Precision | Recall | F1 | Query")
    print("-" * 90)
    avg_p = avg_r = avg_f1 = 0.0

    for r in results:
        print(f"{r['query_id']:7} | {r['precision']:.3f}     | {r['recall']:.3f}  | {r['f1']:.3f} | {r['query']}")
        avg_p  += r["precision"]
        avg_r  += r["recall"]
        avg_f1 += r["f1"]

    n = len(results)
    if n > 0:
        avg_p  /= n
        avg_r  /= n
        avg_f1 /= n
        print("-" * 90)
        print(f"AVERAGE | {avg_p:.3f}     | {avg_r:.3f}  | {avg_f1:.3f} |")

if __name__ == "__main__":
    model = "tfidf"
    if len(sys.argv) > 1 and sys.argv[1].lower() in ("tfidf", "bm25"):
        model = sys.argv[1].lower()
    evaluate(k=5, model=model)