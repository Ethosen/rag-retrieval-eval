"""How much does each retriever depend on the question reusing words from its gold section?

Writes results/OVERLAP.md. Run after `python -m rag_retrieval_eval run`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rag_retrieval_eval import data
from rag_retrieval_eval.retrievers import tokenize

SYSTEMS = ["bm25", "bge-small", "e5-small", "hybrid", "hybrid+rerank"]


def main(out_dir: Path = Path("results")) -> None:
    queries = {q.id: q for q in data.load_queries()}
    sections = {c.section_key: set(tokenize(c.text)) for c in data.load_chunks(chunk_words=0)}

    def coverage(q: data.Query) -> float:
        terms = tokenize(q.text)
        return float(np.mean([t in sections[q.section_key] for t in terms])) if terms else 0.0

    df = pd.read_csv(out_dir / "per_query.csv")
    wide = df.pivot(index="query_id", columns="system", values="ndcg@10")
    systems = [s for s in SYSTEMS if s in wide.columns]
    wide["coverage"] = pd.Series({qid: coverage(q) for qid, q in queries.items()})
    wide["quartile"] = pd.qcut(wide["coverage"], 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])

    by_quartile = wide.groupby("quartile", observed=True).agg(
        coverage=("coverage", "mean"), **{s: (s, "mean") for s in systems}
    )
    lines = [
        "# Lexical overlap analysis",
        "",
        "Coverage = share of a question's terms (lowercased, stopwords removed) that appear in its gold section. "
        f"Mean coverage over all {len(wide)} questions: {wide['coverage'].mean():.3f}.",
        "",
        "nDCG@10 by coverage quartile:",
        "",
        by_quartile.to_markdown(floatfmt=".3f"),
        "",
        f"BM25 beats bge-small on {(wide['bm25'] > wide['bge-small']).mean():.1%} of questions, "
        f"bge-small beats BM25 on {(wide['bge-small'] > wide['bm25']).mean():.1%}, the rest are ties.",
        "",
    ]
    path = out_dir / "OVERLAP.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
