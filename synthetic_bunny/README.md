# synthetic_bunny

`synthetic_bunny` is a synthetic-only, self-contained mini-pipeline that mimics the core selection behavior of:
- GraphRAG
- BunnyRAG

It is designed for experiments where graph nodes are represented by numeric vectors (not text).  
Because there is no text embedding step, this folder does **not** call Hugging Face or `SentenceTransformer`, so runs are faster and cleaner for synthetic experiments.

## What "self-contained" means

Within this folder, all retrieval/sweep/visualization logic is implemented locally:
- no dependency on the notebook chains
- no dependency on LLM APIs
- no dependency on Hugging Face model downloads

It still needs two input files (usually generated elsewhere in this repo):
- a Bunny-compatible graph JSON
- a vectors JSON for the same node IDs

## Inputs expected

1. Graph JSON (Bunny-compatible)
- Path example: `Bunny Rags/random_spherical_bunny_graph_75.json`
- Required shape:
  - `nodes`: object keyed by node ID strings
  - `edges`: list of `[src, dst, {"weight": <float>}]`

2. Vectors JSON
- Path example: `Graph Algorithm/random_spherical_vectors_75.json`
- Required shape:
  - `vectors`: either a list (index -> node id) or object keyed by node ID strings

## Scripts in this folder

- `common.py`
  - Shared loading, similarity, weighted-distance, and effective-resistance utilities.

- `synthetic_graphrag.py`
  - Synthetic GraphRAG selection.
  - Seed nodes are chosen by query-vector cosine similarity.
  - Expansion ranking uses **average weighted shortest-path distance** to seed nodes.
  - Edge cost is `1 / weight` (higher weight = closer).

- `synthetic_bunnyrag.py`
  - Synthetic BunnyRAG selection.
  - Uses effective resistance on the weighted graph.
  - Utility:
    - reward = average normalized conductance over seeds
    - penalty = `lambda * average seed cosine`

- `synthetic_lambda_sweep.py`
  - Runs BunnyRAG across multiple lambda values.
  - Also computes a GraphRAG baseline and overlap reports.

- `visualize_lambda_sweep.py`
  - Interactive HTML visualization with slider:
    - `GraphRAG`
    - `λ = ...` for each lambda from sweep output
  - Uses a community-aware layout for clearer cluster structure.

## Query modes

All synthetic scripts support:

1. Vertex-subset query:
- `--query-vertices "0,4,12"`
- Query vector = normalized mean of listed node vectors

2. Random-query mode:
- `--query-random-points 1 --query-seed 17`
- Current behavior uses one random unit vector in the same dimension as node vectors
- `--query-seed` controls reproducibility

## Quick start

Run a lambda sweep:

```powershell
python synthetic_bunny\synthetic_lambda_sweep.py ^
  --graph-path "Bunny Rags/random_spherical_bunny_graph_75.json" ^
  --vectors-path "Graph Algorithm/random_spherical_vectors_75.json" ^
  --query-random-points 1 --query-seed 17 ^
  --seed-k 3 --top-k 10 ^
  --lambdas "0,0.1,0.2,0.3,0.4,0.5" ^
  --graphrag-max-distance 6.0 ^
  --output-dir "synthetic_bunny/output/example_run"
```

Generate visualization:

```powershell
python synthetic_bunny\visualize_lambda_sweep.py ^
  --graph-path "Bunny Rags/random_spherical_bunny_graph_75.json" ^
  --selected-nodes-path "synthetic_bunny/output/example_run/synthetic_bunny_lambda_selected_nodes.json" ^
  --graphrag-path "synthetic_bunny/output/example_run/synthetic_graphrag_topk_selected_nodes.json" ^
  --output-html "synthetic_bunny/output/example_run/synthetic_lambda_sweep_visualization.html"
```

## Outputs

Typical sweep outputs:
- `synthetic_bunny_lambda_selected_nodes.json`
- `synthetic_graphrag_topk_selected_nodes.json`
- `synthetic_bunny_lambda_sweep_summary.csv`
- `synthetic_bunny_vs_graphrag_overlap.csv`
- `synthetic_bunny_lambda_sweep_report.txt`

Visualization output:
- `synthetic_lambda_sweep_visualization.html`

## Limitations

- This folder is for **synthetic vector/graph data only**.
- It is not a drop-in replacement for the text-based BunnyRAG/GraphRAG pipelines.
- Query semantics are geometric (vector-space), not natural-language understanding.

## Reproducibility tips

- Fix all seeds (`graph` generation seed and `query-seed`) for stable comparisons.
- Keep graph and vector files paired from the same generation run.
- Record `seed_k`, `top_k`, lambda list, and max-distance settings with outputs.

