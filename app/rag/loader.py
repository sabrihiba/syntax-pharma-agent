
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parents[2] / "corpus"


def load_documents():
    documents = []

    for file_path in sorted(CORPUS_DIR.glob("*.md")):
        text = file_path.read_text(encoding="utf-8")

        document = {
            "source": file_path.name,
            "content": text
        }

        documents.append(document)

    return documents
