# PR2 Notes: BunnyRAG Retriever Update + Lambda Sweep

Date: 2026-02-17

## Goal
Improve and validate the BunnyRAG node-selection step used after seed retrieval.

## What Changed

### 1) Seed selection bug fix
File: `Bunny Rags/bunny_retriever.py`

Issue:
- Seed nodes were sorted in ascending cosine similarity, which selected less-related nodes.

Fix:
- Changed sorting to descending cosine similarity so seeds are the closest matches to the query.

---

### 2) MMR-like utility framework update
File: `Bunny Rags/bunny_retriever.py`

Current score for each candidate node `v`:

`score(v) = (1/|S|) * sum_s C_norm(v,s) - labda * (1/|S|) * sum_s cos(e_v, e_s)`

Where:
- `S` = seed/source nodes
- `R(v,s)` = effective resistance
- `C(v,s) = 1 / R(v,s)` (effective conductance)
- `C_norm(v,s)` = min-max normalized conductance in [0,1]
- `e_v`, `e_s` = embedding vectors

Rules implemented:
- If `R(v,s)` is non-finite (infinite), conductance reward is `0`.
- If `R(v,s)` is near zero (`<= 1e-12`), raise an error.
- Do not score seed nodes as candidates.
- Rank by highest score first.

## Experiment Added
File: `tests/bunny_lambda_sweep.py`

Purpose:
- Run one query across multiple lambda values and compare selected nodes + similarity behavior.

Query used:
- `What happens when the circumcenter is on the side of the triangle?`

Lambda values:
- `[-0.2, -0.1, 0.0, 0.1, 0.2]`

Outputs:
- `tests/output/bunny_lambda_selected_nodes.json`
- `tests/output/bunny_lambda_sweep_summary.csv`
- `tests/output/bunny_lambda_sweep_report.txt`
- `tests/output/bunny_lambda_top10_component_terms.csv`

## Key Observations From This Run
- Selected node sets differed across lambda groups.
- Average query similarity was non-increasing as lambda increased.
- For this run:
  - `lambda = -0.2, -0.1` produced one stable top-5 set
  - `lambda = 0.0, 0.1, 0.2` produced another stable top-5 set (with small order changes)

## Notes / Limitations
- The penalty term uses similarity to seed nodes, not direct query similarity.
- Query similarity was used as an external evaluation metric in the sweep.
- First model load may require network access to download Hugging Face model assets.
