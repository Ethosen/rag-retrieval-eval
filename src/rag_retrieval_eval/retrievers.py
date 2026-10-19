"""Retrievers that rank sections for a batch of queries.

Every retriever scores chunks, then keeps the best chunk score per section (max-pooling),
so results are always section rankings regardless of how the corpus was chunked.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import scipy.sparse as sp

from .data import Chunk

Ranking = list[tuple[str, float]]  # (section_key, score), best first

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "which",
        "with",
    ]
)


class Retriever(Protocol):
    name: str

    def search(self, queries: Sequence[str], k: int) -> list[Ranking]: ...


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


def _top_sections(scores: np.ndarray, chunks: Sequence[Chunk], k: int, pool: int) -> Ranking:
    """Max-pool chunk scores into sections and return the k best sections."""
    pool = min(pool, len(scores))
    idx = np.argpartition(-scores, pool - 1)[:pool]
    idx = idx[np.argsort(-scores[idx])]
    best: dict[str, float] = {}
    for i in idx:
        key = chunks[i].section_key
        if key not in best:
            best[key] = float(scores[i])
            if len(best) == k:
                break
    return list(best.items())


class BM25:
    """Okapi BM25 over a sparse term matrix: one sparse product scores every chunk for a query."""

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.2, b: float = 0.75, name: str = "bm25") -> None:
        self.name = name
        self.chunks = chunks
        vocab: dict[str, int] = {}
        rows: list[int] = []
        cols: list[int] = []
        vals: list[int] = []
        lengths = np.zeros(len(chunks))
        for row, chunk in enumerate(chunks):
            tokens = tokenize(chunk.text)
            lengths[row] = len(tokens)
            counts: dict[int, int] = {}
            for tok in tokens:
                col = vocab.setdefault(tok, len(vocab))
                counts[col] = counts.get(col, 0) + 1
            rows.extend([row] * len(counts))
            cols.extend(counts.keys())
            vals.extend(counts.values())
        self.vocab = vocab
        tf = sp.csr_matrix((vals, (rows, cols)), shape=(len(chunks), len(vocab)), dtype=np.float32)

        df = np.bincount(tf.indices, minlength=len(vocab))
        self.idf = np.log1p((len(chunks) - df + 0.5) / (df + 0.5)).astype(np.float32)
        norm = k1 * (1 - b + b * lengths / lengths.mean())
        tf = tf.tocoo()
        weighted = tf.data * (k1 + 1) / (tf.data + norm[tf.row])
        self.weights = sp.csc_matrix((weighted.astype(np.float32), (tf.row, tf.col)), shape=tf.shape)

    def scores(self, query: str) -> np.ndarray:
        cols = [self.vocab[t] for t in set(tokenize(query)) if t in self.vocab]
        if not cols:
            return np.zeros(len(self.chunks), dtype=np.float32)
        return np.asarray(self.weights[:, cols] @ self.idf[cols]).ravel()

    def search(self, queries: Sequence[str], k: int) -> list[Ranking]:
        return [_top_sections(self.scores(q), self.chunks, k, pool=k * 20) for q in queries]


@dataclass
class DenseModel:
    """A sentence-transformers bi-encoder plus the instruction prefixes it was trained with."""

    model_id: str
    query_prefix: str = ""
    passage_prefix: str = ""
    max_seq_length: int = 512


DENSE_MODELS = {
    "minilm": DenseModel("sentence-transformers/all-MiniLM-L6-v2", max_seq_length=256),
    "bge-small": DenseModel(
        "BAAI/bge-small-en-v1.5", query_prefix="Represent this sentence for searching relevant passages: "
    ),
    "e5-small": DenseModel("intfloat/e5-small-v2", query_prefix="query: ", passage_prefix="passage: "),
}


@dataclass
class Dense:
    chunks: Sequence[Chunk]
    model: DenseModel
    name: str
    cache_dir: Path
    batch_size: int = 64
    _encoder: object = field(default=None, init=False, repr=False)
    _matrix: np.ndarray | None = field(default=None, init=False, repr=False)

    def _load(self):  # noqa: ANN202 - SentenceTransformer import is deferred on purpose
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model.model_id)
            self._encoder.max_seq_length = self.model.max_seq_length
        return self._encoder

    def _encode(self, texts: Sequence[str], prefix: str, progress: bool) -> np.ndarray:
        return (
            self._load()
            .encode(
                [prefix + t for t in texts],
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=progress,
            )
            .astype(np.float32)
        )

    def matrix(self) -> np.ndarray:
        """Chunk embeddings, cached on disk keyed by model and corpus content."""
        if self._matrix is None:
            digest = hashlib.sha1()
            digest.update(self.model.model_id.encode())
            for c in self.chunks:
                digest.update(c.text.encode())
            path = self.cache_dir / f"{self.name}-{digest.hexdigest()[:12]}.npy"
            if path.exists():
                self._matrix = np.load(path)
            else:
                self._matrix = self._encode([c.text for c in self.chunks], self.model.passage_prefix, progress=True)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, self._matrix)
        return self._matrix

    def scores_batch(self, queries: Sequence[str]) -> np.ndarray:
        q = self._encode(queries, self.model.query_prefix, progress=False)
        return q @ self.matrix().T

    def search(self, queries: Sequence[str], k: int) -> list[Ranking]:
        out: list[Ranking] = []
        for start in range(0, len(queries), 256):
            for row in self.scores_batch(queries[start : start + 256]):
                out.append(_top_sections(row, self.chunks, k, pool=k * 20))
        return out


@dataclass
class HybridRRF:
    """Reciprocal Rank Fusion of several retrievers (Cormack et al., 2009)."""

    parts: Sequence[Retriever]
    name: str
    rrf_k: int = 60
    depth: int = 100

    def search(self, queries: Sequence[str], k: int) -> list[Ranking]:
        runs = [r.search(queries, self.depth) for r in self.parts]
        fused: list[Ranking] = []
        for per_query in zip(*runs, strict=True):
            scores: dict[str, float] = {}
            for ranking in per_query:
                for rank, (key, _) in enumerate(ranking):
                    scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            fused.append(sorted(scores.items(), key=lambda kv: -kv[1])[:k])
        return fused


@dataclass
class Reranked:
    """Re-score the first stage's top sections with a cross-encoder, using each section's best chunk."""

    first_stage: Retriever
    chunks: Sequence[Chunk]
    name: str
    model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    depth: int = 20
    batch_size: int = 64

    def search(self, queries: Sequence[str], k: int) -> list[Ranking]:
        from sentence_transformers import CrossEncoder
        from tqdm import tqdm

        encoder = CrossEncoder(self.model_id, max_length=512)
        by_section: dict[str, list[str]] = {}
        for c in self.chunks:
            by_section.setdefault(c.section_key, []).append(c.text)

        candidates = self.first_stage.search(queries, self.depth)
        out: list[Ranking] = []
        for query, ranking in tqdm(list(zip(queries, candidates, strict=True)), desc=f"rerank {self.name}"):
            pairs, owners = [], []
            for key, _ in ranking:
                for text in by_section[key]:
                    pairs.append((query, text))
                    owners.append(key)
            scores = encoder.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
            best: dict[str, float] = {}
            for key, s in zip(owners, scores, strict=True):
                best[key] = max(best.get(key, -np.inf), float(s))
            out.append(sorted(best.items(), key=lambda kv: -kv[1])[:k])
        return out
