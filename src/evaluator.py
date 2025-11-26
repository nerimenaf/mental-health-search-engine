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

def compute_map_mrr(model: str = "tfidf"):
    """
    Calcule MAP (Mean Average Precision) et MRR (Mean Reciprocal Rank)
    pour le modèle donné sur toutes les requêtes d'évaluation.
    """
    gt = load_ground_truth()
    ap_list = []
    rr_list = []

    for qid, info in gt.items():
        query = info["query"]
        relevant_docs = info["relevant_docs"]
        if not relevant_docs:
            continue

        # Récupérer le ranking complet (tous les documents)
        ranking, _ = search(query, k=None, model=model)
        ranked_docs = [doc_id for doc_id, _ in ranking]

        num_relevant_found = 0
        precisions_at_relevant = []
        first_relevant_rank = None
        total_relevant = len(relevant_docs)

        for i, doc_id in enumerate(ranked_docs, start=1):
            if doc_id in relevant_docs:
                num_relevant_found += 1
                precision_at_i = num_relevant_found / i
                precisions_at_relevant.append(precision_at_i)
                if first_relevant_rank is None:
                    first_relevant_rank = i

        # Average Precision (AP)
        if precisions_at_relevant:
            ap = sum(precisions_at_relevant) / total_relevant
            ap_list.append(ap)

        # Reciprocal Rank (RR)
        if first_relevant_rank is not None:
            rr_list.append(1.0 / first_relevant_rank)

    MAP = sum(ap_list) / len(ap_list) if ap_list else 0.0
    MRR = sum(rr_list) / len(rr_list) if rr_list else 0.0

    print(f"MAP ({model.upper()}): {MAP:.3f}")
    print(f"MRR ({model.upper()}): {MRR:.3f}")

if __name__ == "__main__":
    model = "tfidf"
    if len(sys.argv) > 1 and sys.argv[1].lower() in ("tfidf", "bm25"):
        model = sys.argv[1].lower()

    # P, R, F1 @k
    evaluate(k=5, model=model)

    # MAP & MRR (bonus)
    compute_map_mrr(model=model)

