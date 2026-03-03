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

- `generate_synthetic_data.py`
  - Generates synthetic graph + vectors payloads.
  - Supports `--vector-space {orthant,sphere}`.

- `synthetic_graphrag.py`
  - Synthetic GraphRAG selection.
  - Seed nodes are chosen by query-vector cosine similarity.
  - Expansion ranking uses average weighted shortest-path distance to seed nodes.
  - Edge cost is `1 / weight` (higher weight = closer).
  - Supports `--query-vector-space {orthant,sphere}` in random-query mode.

- `synthetic_bunnyrag.py`
  - Synthetic BunnyRAG selection.
  - Uses effective resistance on the weighted graph.
  - Utility:
    - reward = average normalized conductance over seeds
    - penalty = `lambda * average seed cosine`
  - Supports `--query-vector-space {orthant,sphere}` in random-query mode.

- `synthetic_lambda_sweep.py`
  - Runs BunnyRAG across multiple lambda values.
  - Also computes a GraphRAG baseline and overlap reports.
  - Supports `--query-vector-space {orthant,sphere}` in random-query mode.

- `visualize_lambda_sweep.py`
  - Interactive HTML visualization with slider:
    - `GraphRAG`
    - `lambda = ...` for each lambda from sweep output
  - Uses a community-aware layout for clearer cluster structure.

- `behavior_test_runner.py`
  - Runs multi-dataset, multi-query behavior experiments.
  - Supports orthant/sphere graph generation and same/mixed seed-community policy.
  - Uses similarity deltas (`bunny - baseline`) for relevance metrics.

- `export_behavior_excel.py` / `export_behavior_excel_condensed.py`
  - Convert behavior runner CSV + metadata outputs into Excel reports.

## Query modes

All synthetic scripts support:

1. Vertex-subset query:
- `--query-vertices "0,4,12"`
- Query vector = normalized mean of listed node vectors

2. Random-query mode:
- `--query-random-points 1 --query-seed 17`
- One random unit vector is used in the same dimension as node vectors
- `--query-seed` controls reproducibility
- `--query-vector-space {orthant,sphere}` controls random-query sampling:
  - `orthant`: non-negative coordinates
  - `sphere`: full-sphere Gaussian sampling, then normalization

## Quick start

Generate synthetic graph + vectors:

```powershell
python synthetic_bunny\generate_synthetic_data.py ^
  --n 75 --dim 4 --scale-prob 0.1 ^
  --vector-space orthant ^
  --output-path "synthetic_bunny/output/example_run/random_spherical_bunny_graph.json" ^
  --vectors-output-path "synthetic_bunny/output/example_run/random_spherical_vectors.json"
```

Use `--vector-space sphere` to sample node vectors over the full unit sphere.

Run a lambda sweep:

```powershell
python synthetic_bunny\synthetic_lambda_sweep.py ^
  --graph-path "synthetic_bunny/output/example_run/random_spherical_bunny_graph.json" ^
  --vectors-path "synthetic_bunny/output/example_run/random_spherical_vectors.json" ^
  --query-random-points 1 --query-seed 17 --query-vector-space sphere ^
  --seed-k 3 --top-k 10 ^
  --lambdas "0,0.1,0.2,0.3,0.4,0.5" ^
  --graphrag-max-distance 6.0 ^
  --output-dir "synthetic_bunny/output/example_run"
```

Generate visualization:

```powershell
python synthetic_bunny\visualize_lambda_sweep.py ^
  --graph-path "synthetic_bunny/output/example_run/random_spherical_bunny_graph.json" ^
  --selected-nodes-path "synthetic_bunny/output/example_run/synthetic_bunny_lambda_selected_nodes.json" ^
  --graphrag-path "synthetic_bunny/output/example_run/synthetic_graphrag_topk_selected_nodes.json" ^
  --output-html "synthetic_bunny/output/example_run/synthetic_lambda_sweep_visualization.html"
```

## Behavior runner quick start

Run a behavior sweep:

```powershell
python synthetic_bunny\behavior_test_runner.py ^
  --num-datasets 5 --queries-per-dataset 10 ^
  --vector-space sphere ^
  --seed-community-policy same ^
  --output-dir "synthetic_bunny/output/behavior_runner/sphere_same"
```

Export condensed Excel report:

```powershell
python synthetic_bunny\export_behavior_excel_condensed.py ^
  --rows-csv "synthetic_bunny/output/behavior_runner/sphere_same/behavior_results_rows.csv" ^
  --metadata-json "synthetic_bunny/output/behavior_runner/sphere_same/behavior_metadata.json" ^
  --output-xlsx "presentation/testing/behavior_report_condensed_sphere_same.xlsx"
```

## Outputs

Typical sweep outputs:
- `synthetic_bunny_lambda_selected_nodes.json`
- `synthetic_graphrag_topk_selected_nodes.json`
- `synthetic_bunny_lambda_sweep_report.txt`

Visualization output:
- `synthetic_lambda_sweep_visualization.html`

Behavior runner outputs:
- `behavior_results_rows.json`
- `behavior_results_rows.csv`
- `behavior_metadata.json`

## Limitations

- This folder is for **synthetic vector/graph data only**.
- It is not a drop-in replacement for the text-based BunnyRAG/GraphRAG pipelines.
- Query semantics are geometric (vector-space), not natural-language understanding.

## Reproducibility tips

- Fix all seeds (`graph` generation seed and `query-seed`) for stable comparisons.
- Keep graph and vector files paired from the same generation run.
- Record `seed_k`, `top_k`, lambda list, and max-distance settings with outputs.
