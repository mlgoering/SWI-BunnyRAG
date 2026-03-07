# CauseNet-Precision Preparation Runbook (Agent)

## Goal
Prepare `causenet-precision` in the same output format used for `causenet-sample`:
- Input: CauseNet data
- Output: Bunny-compatible graph JSON with keys:
  - `nodes`
  - `variants`
  - `edges`

Target output file:
- `Bunny_Rags/causenet_precision_bunny_graph.json`

## Important Constraints
- Do **not** commit raw dataset files to Git (GitHub file-size limits, large diffs).
- Keep raw files under:
  - `data/external/causenet/`
- Recommended raw filename:
  - `data/external/causenet/causenet-precision.jsonl.bz2`

## Step 1: Download Dataset
PowerShell command:

```powershell
New-Item -ItemType Directory -Force "data\external\causenet" | Out-Null
Invoke-WebRequest `
  -Uri "https://zenodo.org/records/3876154/files/causenet-precision.jsonl.bz2?download=1" `
  -OutFile "data/external/causenet/causenet-precision.jsonl.bz2"
```

## Step 2: Convert to Bunny Format (Streaming)
Because this file is JSONL + BZ2, parse it line-by-line (do not fully load into memory).

Create/use a script (for example `Data_generation/convert_causenet_precision_to_bunny.py`) with this behavior:

1. Open `causenet-precision.jsonl.bz2` using `bz2.open(..., "rt", encoding="utf-8")`.
2. For each line:
   - Parse JSON object.
   - Read relation object:
     - Prefer `row["causal_relation"]` if present, else `row`.
   - Extract:
     - `cause` concept (`cause["concept"]` if dict, else string)
     - `effect` concept (`effect["concept"]` if dict, else string)
   - Normalize both concepts:
     - trim whitespace
     - collapse internal spaces
     - lowercase
   - Skip invalid/empty cause/effect.
3. Enforce graph constraints for methodology compatibility:
   - Remove self-loops (`cause == effect`).
   - Compute connected components on the undirected projection.
   - Keep only edges whose endpoints are in the **largest connected component**.
4. Build output structures from the filtered graph:
   - `nodes`: map each concept to itself (string ID -> display text)
   - `variants`: empty dict `{}` (same as sample conversion path)
   - `edges`: aggregate by `(cause, effect)`:
     - if `confidence/score/weight` exists: average it
     - else: use count as weight
5. Sort edges deterministically by `(cause, effect)`.
6. Write JSON:
   - `Bunny_Rags/causenet_precision_bunny_graph.json`
   - UTF-8, `indent=2`

Run command (default: largest-component filter ON, self-loop removal ON):

```powershell
python "Data_generation/convert_causenet_precision_to_bunny.py"
```

Optional flags:
- Disable largest-component filter: `--no-largest-component-filter`
- Keep self-loops: `--keep-self-loops`

Expected output schema:

```json
{
  "nodes": { "concept_a": "concept_a" },
  "variants": {},
  "edges": [
    ["concept_a", "concept_b", {"weight": 1.0}]
  ]
}
```

## Step 3: Validate Output
Run a minimal load check using Bunny builder:

```powershell
@'
import sys
from pathlib import Path
repo = Path(".").resolve()
sys.path.insert(0, str(repo / "Bunny_Rags"))
from builder import CausalGraphBuilder

graph_path = repo / "Bunny_Rags" / "causenet_precision_bunny_graph.json"
b = CausalGraphBuilder()
ok = b.load(str(graph_path))
print("loaded:", ok)
print("nodes:", b.get_graph().number_of_nodes())
print("edges:", b.get_graph().number_of_edges())
'@ | .\.venv\Scripts\python.exe -
```

## Step 4: Keep Raw Data Untracked
Ensure raw dataset remains untracked (recommended):
- Add to `.gitignore`:
  - `data/external/causenet/*.bz2`
  - optionally `data/external/causenet/*.jsonl`

## Notes for Experiment Planning
- Even after conversion, full precision graph is very large.
- Current Bunny lambda sweep uses dense effective-resistance pseudoinverse and is not feasible at full precision scale without further algorithmic changes or heavy compute.
- The default conversion path here intentionally enforces largest-connected-component filtering + self-loop removal to better match method assumptions.
