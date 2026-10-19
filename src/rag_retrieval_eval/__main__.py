"""Command line: `python -m rag_retrieval_eval download|run|report`."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import pandas as pd

from . import data
from .metrics import evaluate
from .report import write_report
from .retrievers import BM25, DENSE_MODELS, Dense, HybridRRF, Reranked, Retriever

K = 20


def build_systems(cache_dir: Path) -> dict[str, Callable[[], Retriever]]:
    """Each system is built lazily so a run can select a subset without loading every model."""
    corpora: dict[tuple[bool, int], list[data.Chunk]] = {}

    def chunks(tables: bool = True, words: int = 200) -> list[data.Chunk]:
        if (tables, words) not in corpora:
            corpora[tables, words] = data.load_chunks(include_tables=tables, chunk_words=words)
        return corpora[tables, words]

    built: dict[str, Retriever] = {}

    def get(name: str) -> Retriever:
        if name not in built:
            built[name] = factories[name]()
        return built[name]

    def dense(model: str, tables: bool = True, words: int = 200) -> Callable[[], Retriever]:
        def make() -> Retriever:
            tag = ("" if tables else "-tablesFalse") + ("" if words == 200 else f"-words{words}")
            return Dense(chunks(tables, words), DENSE_MODELS[model], name=f"{model}{tag}", cache_dir=cache_dir)

        return make

    factories: dict[str, Callable[[], Retriever]] = {
        "bm25": lambda: BM25(chunks(), name="bm25"),
        "minilm": dense("minilm"),
        "bge-small": dense("bge-small"),
        "e5-small": dense("e5-small"),
        "hybrid": lambda: HybridRRF([get("bm25"), get("bge-small")], name="hybrid"),
        "hybrid+rerank": lambda: Reranked(get("hybrid"), chunks(), name="hybrid+rerank"),
        # Ablations
        "bm25/no-tables": lambda: BM25(chunks(tables=False), name="bm25/no-tables"),
        "bge-small/no-tables": dense("bge-small", tables=False),
        "bge-small/whole-section": dense("bge-small", words=0),
    }
    return {name: partial(get, name) for name in factories}


def run(systems: list[str], limit: int | None, out_dir: Path) -> None:
    queries = data.load_queries()
    if limit:
        queries = queries[:limit]
    texts = [q.text for q in queries]
    available = build_systems(out_dir / "cache")
    unknown = set(systems) - set(available)
    if unknown:
        raise SystemExit(f"unknown systems: {sorted(unknown)}; choose from {list(available)}")

    per_query_path = out_dir / "per_query.csv"
    previous = pd.read_csv(per_query_path) if per_query_path.exists() and not limit else None
    frames, timings = [], {}
    for name in systems:
        start = time.perf_counter()
        retriever = available[name]()
        rankings = retriever.search(texts, K)
        timings[name] = round(time.perf_counter() - start, 1)
        print(f"{name}: {timings[name]}s")
        rows = evaluate(queries, rankings)
        frame = pd.DataFrame(rows)
        frame.insert(0, "system", name)
        frame.insert(1, "query_id", [q.id for q in queries])
        frame.insert(2, "type", [q.type for q in queries])
        frame.insert(3, "source", [q.source for q in queries])
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True)
    if limit:
        print(result.drop(columns=["query_id", "type", "source"]).groupby("system").mean().round(3))
        return
    if previous is not None:
        result = pd.concat([previous[~previous["system"].isin(systems)], result], ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(per_query_path, index=False)
    timings_path = out_dir / "timings.json"
    old = json.loads(timings_path.read_text()) if timings_path.exists() else {}
    timings_path.write_text(json.dumps({**old, **timings}, indent=2))
    write_report(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag_retrieval_eval")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="fetch the Open RAG Bench arXiv split from Hugging Face")
    p_run = sub.add_parser("run", help="evaluate retrievers and update results/")
    p_run.add_argument("systems", nargs="*", default=["bm25", "minilm", "bge-small", "e5-small", "hybrid"])
    p_run.add_argument("--limit", type=int, help="only the first N queries, printed and not saved (smoke test)")
    p_run.add_argument("--out", type=Path, default=Path("results"))
    p_report = sub.add_parser("report", help="rebuild results/RESULTS.md from per_query.csv")
    p_report.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    if args.command == "download":
        print(data.download())
    elif args.command == "run":
        run(args.systems, args.limit, args.out)
    else:
        write_report(args.out)


if __name__ == "__main__":
    main()
