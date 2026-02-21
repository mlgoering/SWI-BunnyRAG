# Seed 58 Same-Community Example

This folder stores one curated, reproducible synthetic run where the 3 query seed nodes are all in the same detected community.

## Included files

- `random_spherical_bunny_graph.json`
- `random_spherical_vectors.json`
- `lambda_sweep/synthetic_bunny_lambda_selected_nodes.json`
- `lambda_sweep/synthetic_graphrag_topk_selected_nodes.json`
- `lambda_sweep/synthetic_bunny_lambda_sweep_report.txt`
- `lambda_sweep/synthetic_lambda_sweep_visualization.html`

## How this example was produced

1. Generate data:

```powershell
python synthetic_bunny\generate_synthetic_data.py --n 75 --dim 4 --scale-prob 0.1 --output-path "synthetic_bunny/output/generated/run_20260221_113451/random_spherical_bunny_graph.json" --vectors-output-path "synthetic_bunny/output/generated/run_20260221_113451/random_spherical_vectors.json"
```

2. Search random query seeds until top-3 seed nodes are in one community.
   - Found `query_seed=58` (seed nodes `39,57,61` in same community).

3. Run lambda sweep:

```powershell
python synthetic_bunny\synthetic_lambda_sweep.py --graph-path "synthetic_bunny/output/generated/run_20260221_113451/random_spherical_bunny_graph.json" --vectors-path "synthetic_bunny/output/generated/run_20260221_113451/random_spherical_vectors.json" --query-random-points 1 --query-seed 58 --seed-k 3 --top-k 10 --lambdas "0,0.1,0.2,0.3,0.4,0.5" --graphrag-max-distance 6.0 --output-dir "synthetic_bunny/output/generated/run_20260221_113451/lambda_sweep_same_comm"
```

4. Build visualization:

```powershell
python synthetic_bunny\visualize_lambda_sweep.py --graph-path "synthetic_bunny/output/generated/run_20260221_113451/random_spherical_bunny_graph.json" --selected-nodes-path "synthetic_bunny/output/generated/run_20260221_113451/lambda_sweep_same_comm/synthetic_bunny_lambda_selected_nodes.json" --graphrag-path "synthetic_bunny/output/generated/run_20260221_113451/lambda_sweep_same_comm/synthetic_graphrag_topk_selected_nodes.json" --output-html "synthetic_bunny/output/generated/run_20260221_113451/lambda_sweep_same_comm/synthetic_lambda_sweep_visualization.html"
```
