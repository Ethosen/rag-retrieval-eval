# Lexical overlap analysis

Coverage = share of a question's terms (lowercased, stopwords removed) that appear in its gold section. Mean coverage over all 3045 questions: 0.716.

nDCG@10 by coverage quartile:

| quartile     |   coverage |   bm25 |   bge-small |   e5-small |   hybrid |   hybrid+rerank |
|:-------------|-----------:|-------:|------------:|-----------:|---------:|----------------:|
| Q1 (lowest)  |      0.472 |  0.590 |       0.640 |      0.614 |    0.676 |           0.735 |
| Q2           |      0.691 |  0.835 |       0.747 |      0.751 |    0.830 |           0.850 |
| Q3           |      0.814 |  0.878 |       0.797 |      0.775 |    0.871 |           0.892 |
| Q4 (highest) |      0.946 |  0.904 |       0.810 |      0.798 |    0.885 |           0.901 |

BM25 beats bge-small on 29.5% of questions, bge-small beats BM25 on 20.6%, the rest are ties.
