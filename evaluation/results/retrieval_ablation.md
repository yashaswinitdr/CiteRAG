| Embedding model | Chunk size | Mode | Chunks | Hit@1 | Hit@3 | Hit@5 | MRR | Top-1 source acc. |
|---|---|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 400 | dense | 1087 | 0.29 | 0.38 | 0.48 | 0.35 | 0.91 |
| all-MiniLM-L6-v2 | 400 | dense_rerank | 1087 | 0.52 | 0.57 | 0.67 | 0.56 | 0.91 |
| all-MiniLM-L6-v2 | 400 | hybrid | 1087 | 0.29 | 0.57 | 0.67 | 0.42 | 0.95 |
| all-MiniLM-L6-v2 | 400 | hybrid_rerank | 1087 | 0.71 | 0.81 | 0.91 | 0.77 | 0.95 |
| all-MiniLM-L6-v2 | 800 | dense | 560 | 0.29 | 0.38 | 0.43 | 0.34 | 0.91 |
| all-MiniLM-L6-v2 | 800 | dense_rerank | 560 | 0.43 | 0.62 | 0.62 | 0.51 | 0.91 |
| all-MiniLM-L6-v2 | 800 | hybrid | 560 | 0.38 | 0.57 | 0.67 | 0.49 | 0.86 |
| all-MiniLM-L6-v2 | 800 | hybrid_rerank | 560 | 0.48 | 0.76 | 0.81 | 0.61 | 0.95 |
| all-MiniLM-L6-v2 | 1200 | dense | 374 | 0.10 | 0.48 | 0.52 | 0.29 | 0.86 |
| all-MiniLM-L6-v2 | 1200 | dense_rerank | 374 | 0.33 | 0.48 | 0.57 | 0.42 | 1.00 |
| all-MiniLM-L6-v2 | 1200 | hybrid | 374 | 0.33 | 0.52 | 0.62 | 0.44 | 0.95 |
| all-MiniLM-L6-v2 | 1200 | hybrid_rerank | 374 | 0.48 | 0.62 | 0.71 | 0.56 | 1.00 |
| bge-small-en-v1.5 | 400 | dense | 1087 | 0.43 | 0.57 | 0.57 | 0.48 | 0.95 |
| bge-small-en-v1.5 | 400 | dense_rerank | 1087 | 0.62 | 0.76 | 0.81 | 0.68 | 1.00 |
| bge-small-en-v1.5 | 400 | hybrid | 1087 | 0.38 | 0.71 | 0.81 | 0.54 | 0.95 |
| bge-small-en-v1.5 | 400 | hybrid_rerank | 1087 | 0.67 | 0.76 | 0.91 | 0.74 | 1.00 |
| bge-small-en-v1.5 | 800 | dense | 560 | 0.33 | 0.52 | 0.62 | 0.44 | 0.86 |
| bge-small-en-v1.5 | 800 | dense_rerank | 560 | 0.48 | 0.71 | 0.76 | 0.59 | 0.95 |
| bge-small-en-v1.5 | 800 | hybrid | 560 | 0.43 | 0.67 | 0.76 | 0.56 | 0.91 |
| bge-small-en-v1.5 | 800 | hybrid_rerank | 560 | 0.48 | 0.71 | 0.86 | 0.62 | 0.95 |
| bge-small-en-v1.5 | 1200 | dense | 374 | 0.33 | 0.52 | 0.62 | 0.44 | 0.95 |
| bge-small-en-v1.5 | 1200 | dense_rerank | 374 | 0.48 | 0.62 | 0.71 | 0.56 | 1.00 |
| bge-small-en-v1.5 | 1200 | hybrid | 374 | 0.48 | 0.67 | 0.76 | 0.59 | 1.00 |
| bge-small-en-v1.5 | 1200 | hybrid_rerank | 374 | 0.48 | 0.62 | 0.71 | 0.56 | 1.00 |
