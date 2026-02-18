# PR3 Notes: Dataset Parameterization + Methodology Limitation

Date: 2026-02-18

## What This PR Adds
- Parameterized chain modules and v2 notebooks:
  - `Bunny Rags/bunny_chain.py`
  - `Graph Algorithm/causal_chain.py`
  - `Bunny Rags/bunny rag chain v2.ipynb`
  - `Graph Algorithm/casual rag chain v2.ipynb`
- Parameterized lambda sweep:
  - `tests/bunny_lambda_sweep.py` now accepts configurable graph path, query, lambda list, and Causal baseline options.
  - Adds Bunny-vs-Causal top-k overlap outputs and consecutive-lambda Jaccard reporting.
- New smoke coverage:
  - `tests/test_smoke_v2.py` (exercises Bunny + Causal v2 modules)
- CauseNet tooling:
  - `Data generation/convert_causenet_sample_to_bunny.py`
  - `Data generation/convert_causenet_precision_to_bunny.py`
  - `docs/causenet-precision-agent-runbook.md`

## Major Methodology Limitation (Important)
The current Bunny resistance-based scoring assumes graph conditions that are not met by large parts of available causal datasets.

### Observed structure mismatch
- `Bunny Rags/causenet_sample_bunny_graph.json` is almost entirely tiny disconnected components:
  - 524 nodes, 264 edges, 264 components, mostly size-2.
- The current Wikipedia-derived graph also has many tiny components:
  - 219 nodes, 151 edges, 81 components, including many size-2 components.
- `causenet-precision` has a giant component but still many tiny components:
  - 80,223 nodes, 197,806 edges, 6,949 components, including 6,102 size-2 components.

### Why this breaks the intended interpretation
- `Bunny Rags/bunny_retriever.py` computes effective resistance via global Laplacian pseudoinverse (`np.linalg.pinv`).
- On disconnected graphs, this can yield finite cross-component values instead of the intended "no connection / infinite resistance" behavior.
- Result: conductance normalization can collapse to a few discrete values, which can dominate ranking behavior and violate expected theorem assumptions.

### Practical implication
- Current results can still be useful for engineering experiments and feature validation.
- But claims that rely on theorem assumptions about graph connectivity/behavior should be treated as unsupported until disconnected-component handling is corrected.

## Recommended Next Fix
- Compute effective resistance per connected component and force cross-component resistance to infinity.
- Exclude self-loop-only artifacts when building converted graphs.
- Re-run lambda sweeps after the fix before drawing theoretical conclusions.

## Lambda Sweep Outputs (Current)
- `tests/output/bunny_lambda_selected_nodes.json`
- `tests/output/causal_topk_selected_nodes.json`
- `tests/output/bunny_lambda_sweep_summary.csv`
- `tests/output/bunny_vs_causal_topk_overlap.csv`
- `tests/output/bunny_lambda_sweep_report.txt` (includes consecutive-lambda Jaccard only)
- `tests/output/bunny_lambda_top10_component_terms.csv`
