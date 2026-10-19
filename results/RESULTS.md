# Results

3045 queries, one gold section each. Higher is better. 95% bootstrap intervals over queries are given for nDCG@10. Times are wall-clock on one RTX 2060 and exclude parts already built for an earlier system (the hybrid reuses the BM25 index and bge-small embeddings).

## Overall

| system        |   recall@1 |   recall@5 |   recall@10 |   recall@20 |   mrr@10 |   ndcg@10 |   doc_recall@5 | ndcg@10 95% CI   |   time (s) |
|:--------------|-----------:|-----------:|------------:|------------:|---------:|----------:|---------------:|:-----------------|-----------:|
| bm25          |      0.633 |      0.892 |       0.945 |       0.964 |    0.744 |     0.793 |          0.981 | [0.782, 0.804]   |     16.300 |
| minilm        |      0.459 |      0.766 |       0.865 |       0.923 |    0.589 |     0.655 |          0.948 | [0.642, 0.669]   |    171.100 |
| bge-small     |      0.555 |      0.856 |       0.924 |       0.955 |    0.684 |     0.743 |          0.967 | [0.732, 0.754]   |    642.700 |
| e5-small      |      0.539 |      0.851 |       0.914 |       0.949 |    0.670 |     0.729 |          0.963 | [0.718, 0.740]   |    598.000 |
| hybrid        |      0.641 |      0.924 |       0.964 |       0.985 |    0.758 |     0.809 |          0.990 | [0.799, 0.819]   |      7.900 |
| hybrid+rerank |      0.690 |      0.937 |       0.969 |       0.985 |    0.796 |     0.839 |          0.994 | [0.830, 0.849]   |   1567.000 |

## By question source (nDCG@10)

| system        |   text (n=1914) |   text-image (n=763) |   text-table (n=148) |   text-table-image (n=220) |
|:--------------|----------------:|---------------------:|---------------------:|---------------------------:|
| bm25          |           0.848 |                0.676 |                0.743 |                      0.755 |
| minilm        |           0.655 |                0.626 |                0.743 |                      0.706 |
| bge-small     |           0.758 |                0.696 |                0.780 |                      0.750 |
| e5-small      |           0.762 |                0.643 |                0.766 |                      0.719 |
| hybrid        |           0.838 |                0.740 |                0.815 |                      0.796 |
| hybrid+rerank |           0.869 |                0.773 |                0.848 |                      0.799 |

## By question type (nDCG@10)

| system        |   abstractive (n=1793) |   extractive (n=1252) |
|:--------------|-----------------------:|----------------------:|
| bm25          |                  0.747 |                 0.858 |
| minilm        |                  0.637 |                 0.682 |
| bge-small     |                  0.718 |                 0.778 |
| e5-small      |                  0.702 |                 0.769 |
| hybrid        |                  0.777 |                 0.855 |
| hybrid+rerank |                  0.807 |                 0.885 |

## Paired comparisons

delta = base minus variant; positive means the base setting helps. p comes from a paired bootstrap over queries (2000 resamples).

| comparison                                                            | queries                 |   nDCG@10 base |   nDCG@10 variant |   delta | p (paired bootstrap)   |
|:----------------------------------------------------------------------|:------------------------|---------------:|------------------:|--------:|:-----------------------|
| Hybrid (BM25 + bge-small, RRF) vs. BM25 alone                         | all (n=3045)            |          0.809 |             0.793 |   0.016 | <0.001                 |
| Hybrid (BM25 + bge-small, RRF) vs. BM25 alone                         | table questions (n=368) |          0.804 |             0.751 |   0.053 | <0.001                 |
| Cross-encoder rerank of the hybrid top 20 vs. hybrid                  | all (n=3045)            |          0.839 |             0.809 |   0.030 | <0.001                 |
| Cross-encoder rerank of the hybrid top 20 vs. hybrid                  | table questions (n=368) |          0.819 |             0.804 |   0.015 | 0.232                  |
| Tables inlined as markdown vs. stripped (BM25)                        | all (n=3045)            |          0.793 |             0.790 |   0.003 | 0.061                  |
| Tables inlined as markdown vs. stripped (BM25)                        | table questions (n=368) |          0.751 |             0.727 |   0.024 | 0.014                  |
| Tables inlined as markdown vs. stripped (bge-small)                   | all (n=3045)            |          0.743 |             0.739 |   0.004 | 0.017                  |
| Tables inlined as markdown vs. stripped (bge-small)                   | table questions (n=368) |          0.762 |             0.727 |   0.036 | 0.001                  |
| 200-word chunks vs. whole section truncated to 512 tokens (bge-small) | all (n=3045)            |          0.743 |             0.720 |   0.023 | <0.001                 |
| 200-word chunks vs. whole section truncated to 512 tokens (bge-small) | table questions (n=368) |          0.762 |             0.679 |   0.083 | <0.001                 |
