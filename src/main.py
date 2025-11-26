from search_engine import search, METADATA

def show_results(results, elapsed):
    print(f"\nFound {len(results)} result(s) in {elapsed:.4f} seconds\n")
    for rank, (doc_id, score) in enumerate(results, start=1):
        meta = METADATA[doc_id]
        title = meta["title"]

        # read part of the document as snippet
        with open(meta["path"], encoding="utf-8") as f:
            content = f.read()
        snippet = content[:200].replace("\n", " ")

        print(f"{rank}. [{score:.4f}] {title}")
        print(f"   doc_id : {doc_id}")
        print(f"   Excerpt: {snippet}...")
        print()

def main():
    print("Health Care - Mental Health FAQ Search Engine")
    print("Type 'quit' to exit.\n")

    while True:
        query = input("Query > ").strip()
        if query.lower() in ("quit", "exit", ""):
            break
        results, elapsed = search(query, k=5)
        show_results(results, elapsed)

if __name__ == "__main__":
    main()

#what does it mean to have a mental illness
#who does mental illness affect
#causes of mental illness
#warning signs mental illness*/