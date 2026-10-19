"""Load Open RAG Bench (arXiv split) and turn sections into retrievable chunks."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

DATASET_REPO = "vectara/open_ragbench"
DEFAULT_DATA_DIR = Path.home() / ".cache" / "open_ragbench"

_PLACEHOLDER = re.compile(r"!\[([^\]]+)\]\(\1\)")


@dataclass(frozen=True)
class Query:
    id: str
    text: str
    type: str  # "abstractive" | "extractive"
    source: str  # "text" | "text-image" | "text-table" | "text-table-image"
    doc_id: str
    section_id: int

    @property
    def section_key(self) -> str:
        return section_key(self.doc_id, self.section_id)


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    section_id: int
    text: str

    @property
    def section_key(self) -> str:
        return section_key(self.doc_id, self.section_id)


def section_key(doc_id: str, section_id: int) -> str:
    return f"{doc_id}#{section_id}"


def data_dir() -> Path:
    return Path(os.environ.get("ORB_DATA_DIR", DEFAULT_DATA_DIR))


def download(target: Path | None = None) -> Path:
    """Fetch the JSON files of the arXiv split (about 700 MB) from Hugging Face."""
    from huggingface_hub import snapshot_download

    target = target or data_dir()
    snapshot_download(
        DATASET_REPO,
        repo_type="dataset",
        local_dir=target,
        allow_patterns=["pdf/arxiv/*.json", "pdf/arxiv/corpus/*.json"],
    )
    return target


def _arxiv_dir(root: Path | None) -> Path:
    path = (root or data_dir()) / "pdf" / "arxiv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `python -m rag_retrieval_eval download` first")
    return path


def load_queries(root: Path | None = None) -> list[Query]:
    path = _arxiv_dir(root)
    queries = json.loads((path / "queries.json").read_text(encoding="utf-8"))
    qrels = json.loads((path / "qrels.json").read_text(encoding="utf-8"))
    return [
        Query(
            id=qid,
            text=q["query"],
            type=q["type"],
            source=q["source"],
            doc_id=qrels[qid]["doc_id"],
            section_id=int(qrels[qid]["section_id"]),
        )
        for qid, q in queries.items()
    ]


def render_section(section: dict, include_tables: bool) -> str:
    """Replace table/image placeholders: tables become inline markdown or vanish, images always vanish."""
    tables: dict[str, str] = section.get("tables") or {}

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if include_tables and key in tables:
            return "\n" + tables[key] + "\n"
        return ""

    return _PLACEHOLDER.sub(replace, section["text"])


def split_words(text: str, size: int, overlap: int) -> list[str]:
    """Split into windows of `size` words that overlap by `overlap` words. size <= 0 keeps the text whole."""
    words = text.split()
    if size <= 0 or len(words) <= size:
        return [" ".join(words)] if words else []
    step = size - overlap
    return [" ".join(words[i : i + size]) for i in range(0, len(words) - overlap, step)]


def load_chunks(
    root: Path | None = None,
    include_tables: bool = True,
    chunk_words: int = 200,
    overlap_words: int = 40,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted((_arxiv_dir(root) / "corpus").glob("*.json")):
        paper = json.loads(path.read_text(encoding="utf-8"))
        title = paper["title"].strip()
        for section in paper["sections"]:
            body = render_section(section, include_tables)
            for piece in split_words(body, chunk_words, overlap_words):
                chunks.append(Chunk(paper["id"], int(section["section_id"]), f"{title}\n{piece}"))
    return chunks
