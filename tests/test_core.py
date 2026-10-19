import math

from rag_retrieval_eval.data import Chunk, Query, render_section, split_words
from rag_retrieval_eval.metrics import paired_bootstrap_p, per_query
from rag_retrieval_eval.retrievers import BM25, HybridRRF


def make_query(doc: str = "d1", section: int = 0) -> Query:
    return Query("q", "text", "extractive", "text", doc, section)


def test_split_words_covers_all_words_with_overlap() -> None:
    words = [str(i) for i in range(250)]
    pieces = split_words(" ".join(words), size=100, overlap=20)
    assert pieces[0].split()[0] == "0"
    assert pieces[-1].split()[-1] == "249"
    assert pieces[1].split()[0] == "80"
    assert all(len(p.split()) <= 100 for p in pieces)


def test_split_words_keeps_short_or_unchunked_text_whole() -> None:
    assert split_words("a b c", size=10, overlap=2) == ["a b c"]
    assert split_words("a b c", size=0, overlap=0) == ["a b c"]
    assert split_words("   ", size=10, overlap=2) == []


def test_render_section_inlines_tables_and_drops_images() -> None:
    section = {"text": "before ![table_0](table_0) mid ![img-0.jpeg](img-0.jpeg) after", "tables": {"table_0": "| a |"}}
    assert "| a |" in render_section(section, include_tables=True)
    stripped = render_section(section, include_tables=False)
    assert "table_0" not in stripped and "img-0" not in stripped and "| a |" not in stripped


def test_metrics_for_hit_at_rank_two() -> None:
    row = per_query(make_query(), [("d2#0", 1.0), ("d1#0", 0.5)])
    assert row["recall@1"] == 0.0 and row["recall@5"] == 1.0
    assert row["mrr@10"] == 0.5
    assert math.isclose(row["ndcg@10"], 1 / math.log2(3))
    assert row["doc_recall@5"] == 1.0


def test_metrics_for_miss() -> None:
    row = per_query(make_query(), [("d1#3", 1.0)])
    assert row["recall@20"] == 0.0 and row["mrr@10"] == 0.0
    assert row["doc_recall@5"] == 1.0  # right paper, wrong section


def test_bm25_prefers_matching_chunk_and_pools_by_section() -> None:
    chunks = [
        Chunk("d1", 0, "graph neural networks for molecules"),
        Chunk("d1", 0, "graph neural networks again"),
        Chunk("d2", 0, "convex optimisation of portfolios"),
    ]
    ranking = BM25(chunks).search(["portfolio optimisation"], k=5)[0]
    assert ranking[0][0] == "d2#0"
    assert [key for key, _ in ranking].count("d1#0") <= 1


def test_rrf_rewards_agreement() -> None:
    class Fixed:
        def __init__(self, keys: list[str]) -> None:
            self.name, self.keys = "fixed", keys

        def search(self, queries, k):  # noqa: ANN001, ANN201
            return [[(key, 1.0) for key in self.keys[:k]] for _ in queries]

    fused = HybridRRF([Fixed(["a", "b", "c"]), Fixed(["b", "c", "a"])], name="h").search(["q"], k=3)[0]
    assert fused[0][0] == "b"


def test_paired_bootstrap_detects_clear_difference() -> None:
    assert paired_bootstrap_p([1.0] * 50, [0.0] * 50) < 0.01
    assert paired_bootstrap_p([0.5] * 50, [0.5] * 50) == 1.0
