# src/app.py
import math

import streamlit as st
import matplotlib.pyplot as plt

from search_engine import search, METADATA, N, VOCAB_SIZE, AVG_DOC_LENGTH
from evaluator import load_ground_truth

# --- Page config (must be first Streamlit call) ---
st.set_page_config(
    page_title="Health Care - Mental Health FAQ Search",
    layout="wide",
)

# --- Custom CSS for styling ---
CUSTOM_CSS = """
<style>
/* Global */
.main {
    padding-top: 1.5rem;
}
h1, h2, h3 {
    color: #0f172a;
}

/* Result cards */
.result-card {
    background-color: #ffffff;
    border-radius: 0.8rem;
    padding: 0.9rem 1rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.08);
    border: 1px solid #e2e8f0;
}
.result-title {
    font-weight: 600;
    font-size: 1.0rem;
    margin-bottom: 0.2rem;
}
.result-meta {
    font-size: 0.80rem;
    color: #64748b;
    margin-bottom: 0.35rem;
}
.result-score-pill {
    display: inline-block;
    padding: 0.1rem 0.6rem;
    border-radius: 999px;
    background-color: #e0f2f1;
    color: #00695c;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 0.4rem;
}
.result-snippet {
    font-size: 0.90rem;
    color: #334155;
}

/* Buttons in pagination */
.stButton>button {
    border-radius: 999px;
}

/* Sidebar title */
.css-1d391kg, .css-1q8dd3e {
    /* older/newer streamlit class names; ignore if not found */
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- Sidebar: PR curve settings ---
st.sidebar.header("Évaluation – Courbe Précision/Rappel")
show_pr_curve = st.sidebar.checkbox("Afficher une courbe P-R (requêtes de test)")

gt = load_ground_truth()
query_ids = sorted(gt.keys())

if show_pr_curve and query_ids:
    selected_qid = st.sidebar.selectbox("Requête d'évaluation (query_id)", query_ids)
    max_k = st.sidebar.number_input("K maximal pour la courbe", min_value=1, max_value=50, value=10, step=1)
else:
    selected_qid = None
    max_k = 10

# --- Hero / header section ---
st.markdown(
    """
    <h1>Health Care – Moteur de recherche FAQ santé mentale</h1>
    <p style="color:#475569; font-size:0.95rem; max-width:720px;">
        Recherchez dans une base de questions–réponses sur la santé mentale.
        Comparez deux modèles de recherche d'information (TF‑IDF et BM25) et visualisez
        les performances de votre système.
    </p>
    """,
    unsafe_allow_html=True,
)

# --- Top layout: query + options + stats ---
col_query, col_opts, col_stats = st.columns([2.5, 1.2, 1.3])

with col_query:
    query = st.text_input("Requête en langage naturel", "")

with col_opts:
    st.markdown("**Modèle de RI**")
    model = st.radio(
        "Modèle de recherche d'information",
        options=["tfidf", "bm25"],
        format_func=lambda m: "TF‑IDF (modèle vectoriel)" if m == "tfidf" else "BM25 (modèle probabiliste)",
        index=0,
        label_visibility="collapsed",
    )

    k = st.number_input("Nombre maximal de résultats (K)", min_value=1, max_value=50, value=20, step=1)
    page_size = st.number_input("Résultats par page", min_value=1, max_value=20, value=5, step=1)

with col_stats:
    st.markdown("**Statistiques du corpus**")
    s1, s2, s3 = st.columns(3)
    s1.metric("Documents", N)
    s2.metric("Vocabulaire", VOCAB_SIZE)
    s3.metric("Longueur moyenne", f"{AVG_DOC_LENGTH:.0f} tokens")

# --- Session state for pagination ---
if "page" not in st.session_state:
    st.session_state.page = 0
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "last_model" not in st.session_state:
    st.session_state.last_model = ""

# --- Search & results ---
if query:
    # Reset page if query or model changed
    if query != st.session_state.last_query or model != st.session_state.last_model:
        st.session_state.page = 0
        st.session_state.last_query = query
        st.session_state.last_model = model

    # Run search
    results, elapsed = search(query, k=int(k), model=model)
    total_results = len(results)

    st.markdown(
        f"**Modèle :** `{model}` – **{total_results}** résultat(s) en **{elapsed:.4f} s**"
    )

    if results:
        total_pages = math.ceil(total_results / page_size)
        page = st.session_state.page

        start = page * page_size
        end = start + page_size
        page_results = results[start:end]

        st.write(
            f"Affichage des résultats **{start+1}–{min(end, total_results)}** sur **{total_results}**"
        )

        # Display results as cards
        for idx_global, (doc_id, score) in enumerate(page_results, start=start + 1):
            meta = METADATA[doc_id]
            title = meta["title"]

            with open(meta["path"], encoding="utf-8") as f:
                content = f.read()
            snippet = content[:450].replace("\n", " ")

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-title">{idx_global}. {title}</div>
                    <div class="result-meta">
                        <span class="result-score-pill">Score: {score:.4f}</span>
                        <span>doc_id: {doc_id}</span>
                    </div>
                    <div class="result-snippet">{snippet}...</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Pagination controls
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("⟵ Page précédente") and page > 0:
                st.session_state.page -= 1
        with c2:
            st.write(f"Page **{page+1} / {total_pages}**")
        with c3:
            if st.button("Page suivante ⟶") and page < total_pages - 1:
                st.session_state.page += 1

# --- Precision–Recall helper ---

def compute_pr_curve_for_query(qid: str, model: str = "tfidf", max_k: int = 10):
    info = gt[qid]
    query_text = info["query"]
    relevant_docs = info["relevant_docs"]

    if not relevant_docs:
        return [], [], query_text

    retrieved, _ = search(query_text, k=int(max_k), model=model)
    retrieved_docs = [doc_id for doc_id, _ in retrieved]

    recalls = []
    precisions = []
    tp = 0
    total_relevant = len(relevant_docs)

    for i, doc_id in enumerate(retrieved_docs, start=1):
        if doc_id in relevant_docs:
            tp += 1
        precision = tp / i
        recall = tp / total_relevant
        precisions.append(precision)
        recalls.append(recall)

    return recalls, precisions, query_text

# --- Precision–Recall curve section ---
if show_pr_curve and selected_qid:
    st.markdown("---")
    st.subheader("Courbe Précision–Rappel (requêtes d'évaluation)")

    recalls, precisions, q_text = compute_pr_curve_for_query(
        selected_qid, model=model, max_k=int(max_k)
    )

    if recalls and precisions:
        fig, ax = plt.subplots()
        ax.plot(recalls, precisions, marker="o", color="#0f766e")
        ax.set_xlabel("Rappel")
        ax.set_ylabel("Précision")
        ax.set_title(f"Courbe P–R – {selected_qid} ({model})")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        st.write(f"Requête **{selected_qid}** : *{q_text}*")
    else:
        st.info("Pas assez de documents pertinents pour calculer une courbe P‑R.")