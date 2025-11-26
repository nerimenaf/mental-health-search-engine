import csv
import json
from pathlib import Path

DATA_DIR = Path("data")
RAW_CSV = DATA_DIR / "raw" / "Mental_Health_FAQ.csv"
DOC_DIR = DATA_DIR / "documents"
METADATA_PATH = DOC_DIR / "metadata.json"

def fix_encoding(text: str) -> str:
    # fixes weird characters like â€™ to normal apostrophe
    return text.replace("â€™", "'")

def collect():
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    metadata = []

    with open(RAW_CSV, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = row["Question_ID"].strip()
            question = fix_encoding(row["Questions"].strip())
            answer = fix_encoding(row["Answers"].strip())

            content = question + "\n\n" + answer
            doc_path = DOC_DIR / f"{doc_id}.txt"

            with open(doc_path, "w", encoding="utf-8") as out:
                out.write(content)

            metadata.append({
                "doc_id": doc_id,
                "title": question,
                "source": "Health Care Mental Health FAQ",
                "path": str(doc_path),
            })

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    collect()