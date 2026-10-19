"""Turn results/per_query.csv into results/RESULTS.md."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .metrics import bootstrap_ci, paired_bootstrap_p

MAIN_METRICS = ["recall@1", "recall@5", "recall@10", "recall@20", "mrr@10", "ndcg@10", "doc_recall@5"]
ABLATIONS = [
    ("hybrid", "bm25", "Hybrid (BM25 + bge-small, RRF) vs. BM25 alone"),
    ("hybrid+rerank", "hybrid", "Cross-encoder rerank of the hybrid top 20 vs. hybrid"),
    ("bm25", "bm25/no-tables", "Tables inlined as markdown vs. stripped (BM25)"),
    ("bge-small", "bge-small/no-tables", "Tables inlined as markdown vs. stripped (bge-small)"),
    ("bge-small", "bge-small/whole-section", "200-word chunks vs. whole section truncated to 512 tokens (bge-small)"),
]


def _fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _table(frame: pd.DataFrame, floatfmt: str = ".3f") -> str:
    return frame.to_markdown(floatfmt=floatfmt)


def _ordered(frame: pd.DataFrame, systems: list[str]) -> pd.DataFrame:
    return frame.reindex([s for s in systems if s in frame.index])


def write_report(out_dir: Path) -> Path:
    df = pd.read_csv(out_dir / "per_query.csv")
    timings = json.loads((out_dir / "timings.json").read_text()) if (out_dir / "timings.json").exists() else {}
    systems = list(dict.fromkeys(df["system"]))
    main = [s for s in systems if "/" not in s]
    lines = [
        "# Results",
        "",
        f"{df['query_id'].nunique()} queries, one gold section each. Higher is better. "
        "95% bootstrap intervals over queries are given for nDCG@10. "
        "Times are wall-clock on one RTX 2060 and exclude parts already built for an earlier system "
        "(the hybrid reuses the BM25 index and bge-small embeddings).",
        "",
        "## Overall",
        "",
    ]

    overall = df.groupby("system")[MAIN_METRICS].mean()
    ci = df.groupby("system")["ndcg@10"].apply(lambda s: "[{:.3f}, {:.3f}]".format(*bootstrap_ci(s)))
    overall["ndcg@10 95% CI"] = ci
    overall["time (s)"] = pd.Series(timings)
    lines += [_table(_ordered(overall, main)), ""]

    for column, title in (("source", "By question source (nDCG@10)"), ("type", "By question type (nDCG@10)")):
        pivot = df.pivot_table(index="system", columns=column, values="ndcg@10", aggfunc="mean")
        counts = df[df["system"] == systems[0]][column].value_counts()
        pivot.columns = [f"{c} (n={counts[c]})" for c in pivot.columns]
        lines += [f"## {title}", "", _table(_ordered(pivot, main)), ""]

    rows = []
    for base, variant, label in ABLATIONS:
        if base not in systems or variant not in systems:
            continue
        for subset, mask in (("all", None), ("table questions", "table")):
            a = df[df["system"] == base].set_index("query_id")
            b = df[df["system"] == variant].set_index("query_id").loc[a.index]
            if mask:
                keep = a["source"].str.contains(mask)
                a, b = a[keep], b[keep]
            rows.append(
                {
                    "comparison": label,
                    "queries": f"{subset} (n={len(a)})",
                    "nDCG@10 base": a["ndcg@10"].mean(),
                    "nDCG@10 variant": b["ndcg@10"].mean(),
                    "delta": a["ndcg@10"].mean() - b["ndcg@10"].mean(),
                    "p (paired bootstrap)": _fmt_p(paired_bootstrap_p(a["ndcg@10"], b["ndcg@10"])),
                }
            )
    if rows:
        lines += [
            "## Paired comparisons",
            "",
            "delta = base minus variant; positive means the base setting helps. "
            "p comes from a paired bootstrap over queries (2000 resamples).",
            "",
            pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f"),
            "",
        ]

    path = out_dir / "RESULTS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
