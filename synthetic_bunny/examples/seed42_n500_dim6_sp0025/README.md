# Seed 42, n=500, dim=6, scale-prob=0.025 Example

This folder stores a curated synthetic run that matched the 500-node settings you asked to preserve for visualization quality.

## Included files

- `random_spherical_bunny_graph.json`
- `random_spherical_vectors.json`
- `lambda_sweep/synthetic_bunny_lambda_selected_nodes.json`
- `lambda_sweep/synthetic_graphrag_topk_selected_nodes.json`
- `lambda_sweep/synthetic_bunny_lambda_sweep_report.txt`
- `lambda_sweep/synthetic_lambda_sweep_visualization.html`

## Exact inputs used

Data generation:

```powershell
python synthetic_bunny\generate_synthetic_data.py --n 500 --dim 6 --scale-prob 0.025 --seed 42 --output-path "synthetic_bunny/output/example_500_nodes_dim6_sp0025_20260224_103611/random_spherical_bunny_graph.json" --vectors-output-path "synthetic_bunny/output/example_500_nodes_dim6_sp0025_20260224_103611/random_spherical_vectors.json"
```

Lambda sweep:

```powershell
python synthetic_bunny\synthetic_lambda_sweep.py --graph-path "synthetic_bunny/output/example_500_nodes_dim6_sp0025_20260224_103611/random_spherical_bunny_graph.json" --vectors-path "synthetic_bunny/output/example_500_nodes_dim6_sp0025_20260224_103611/random_spherical_vectors.json" --query-random-points 1 --query-seed 17 --seed-k 3 --top-k 10 --lambdas "0,0.1,0.2,0.3,0.4,0.5" --graphrag-max-distance 6.0 --output-dir "synthetic_bunny/output/example_500_nodes_dim6_sp0025_20260224_103611/lambda_sweep"
```

Visualization:

```powershell
python synthetic_bunny\visualize_lambda_sweep.py --graph-path "synthetic_bunny/output/example_500_nodes_dim6_sp0025_20260224_103611/random_spherical_bunny_graph.json" --selected-nodes-path "synthetic_bunny/output/example_500_nodes_dim6_sp0025_20260224_103611/lambda_sweep/synthetic_bunny_lambda_selected_nodes.json" --graphrag-path "synthetic_bunny/output/example_500_nodes_dim6_sp0025_20260224_103611/lambda_sweep/synthetic_graphrag_topk_selected_nodes.json" --output-html "synthetic_bunny/output/example_500_nodes_dim6_sp0025_20260224_103611/lambda_sweep/synthetic_lambda_sweep_visualization.html"
```
