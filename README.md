# SWI-BunnyRAG

This repo is a research prototype that collects Wikipedia math articles, chunks them, builds a causal graph, and runs two RAG variants:
`CausalRAG` (in `Graph_Algorithm/`) and `BunnyRAG` (in `Bunny_Rags/`).

Below is a plain-English map of the repo, plus the dependencies each part needs.

## Portfolio Quickstart

Use this short path when demonstrating reproducible, industry-style workflow.

```powershell
# 1) Verify python
python -V

# 2) Install minimal dependencies in the project venv
.venv\Scripts\python -m pip install -r requirements.txt

# 3) Run tests
.venv\Scripts\python -m pytest -q

# 4) Rebuild website/demo assets
powershell -ExecutionPolicy Bypass -File scripts/build_portfolio_assets.ps1
```

Golden-path synthetic fixture demo (reproducible end-to-end):

```bash
python scripts/run_fixture_golden_path.py
```

What this command does:
- Runs synthetic smoke tests first (`tests/test_synthetic_*` + visualization/data-gen synthetic tests).
- Rebuilds fixture lambda-sweep artifacts for `seed42_n500_dim6_sp0025`.
- Rebuilds the interactive HTML visualization (with taller default plot height for web embedding).

Primary outputs:
- `synthetic_bunny/output/golden_path_seed42_n500_dim6_sp0025/lambda_sweep/synthetic_lambda_sweep_variants.json`
- `synthetic_bunny/output/golden_path_seed42_n500_dim6_sp0025/lambda_sweep/synthetic_bunny_lambda_selected_nodes.json`
- `synthetic_bunny/output/golden_path_seed42_n500_dim6_sp0025/lambda_sweep/synthetic_graphrag_topk_selected_nodes.json`
- `synthetic_bunny/output/golden_path_seed42_n500_dim6_sp0025/lambda_sweep/synthetic_bunny_lambda_sweep_report.txt`
- `synthetic_bunny/output/golden_path_seed42_n500_dim6_sp0025/lambda_sweep/synthetic_lambda_sweep_visualization.html`

Optional flags:

```bash
# Skip synthetic smoke tests (artifacts only)
python scripts/run_fixture_golden_path.py --skip-synth-tests

# Also copy HTML to docs/portfolio/golden_path/ for GitHub Pages
python scripts/run_fixture_golden_path.py --publish-to-docs
```

## Synthetic Analog

For synthetic-only experiments (no text, no Hugging Face model calls), use:
- `synthetic_bunny/`

This folder contains a self-contained synthetic analog of BunnyRAG/GraphRAG and lambda-sweep tooling over synthetic vector/graph data.

- Curated, versioned fixture datasets live in `synthetic_bunny/fixtures/`.
- Ephemeral/generated run outputs should go to `synthetic_bunny/output/` (gitignored).

Key synthetic additions on this branch:
- Data/query vector-space controls:
  - `generate_synthetic_data.py --vector-space {orthant,sphere}`
  - `synthetic_*rag.py` and `synthetic_lambda_sweep.py --query-vector-space {orthant,sphere}`
- Behavior benchmarking:
  - `synthetic_bunny/behavior_test_runner.py --seed-community-policy {same,mixed}`
  - Similarity comparison is tracked as **delta query similarity** (`bunny - baseline`) in behavior outputs/reports.

## Repo Map

### 1) Wikipedia article list + scraping
**What it is**
- A roughly curated list of Wikipedia math URLs.
- A notebook that downloads article text and chunks it into a JSON knowledge base.

**Files**
- `Data_generation/wiiki_scaper.ipynb`
  - Contains `URL_DISCIPLINE_MAP` (the URL list).
  - Scrapes pages using `wikipediaapi`.
  - Writes:
    - `Data_generation/wiki_math_knowledge_base_api.json` (chunked JSON)
    - `Data_generation/wiki_data_api_math/*.md` (one Markdown file per page)

**Dependencies**
- `wikipedia-api` (imported as `wikipediaapi`)
- Standard libs: `json`, `os`, `time`, `re`

Install:
```
pip install wikipedia-api
```

### 2) Chunk embeddings + cosine similarity retrieval
**What it is**
- Encodes query and node text with sentence-transformers.
- Computes cosine similarity to retrieve relevant nodes.

**Files**
- `Graph_Algorithm/retriever.py` (CausalRAG retriever)
- `Bunny_Rags/bunny_retriever.py` (BunnyRAG retriever)

**Dependencies**
- `sentence-transformers`
- `torch`
- `numpy`
- `scipy`
- `networkx`

Install (minimal path):
```
pip install -r requirements.txt
```

### 3) Causal graph construction
**What it is**
- Parses text into causal triples.
- Builds a directed causal graph and stores node embeddings.

**Files**
- `Graph_Algorithm/builder.py` (CausalRAG builder)
- `Bunny_Rags/builder.py` (BunnyRAG builder)

**Dependencies**
- Same as retrieval above.
- Optional extras used in some paths:
  - `pyvis` and `matplotlib` are for graph visualizations (interactive HTML or static plots).
  - `tqdm` is for progress bars during batch processing.

Install (full/dev):
```
pip install -r requirements-full.txt
```

### 4) CausalRAG pipeline (Graph_Algorithm)
**What it is**
- End-to-end CausalRAG chain and evaluation.

**Files**
- `Graph_Algorithm/casual_rag_chain_v1.ipynb`
  - Defines `CausalRAGChain`
  - Runs retrieval and evaluation (cosine similarity, ROUGE)

**Dependencies**
- Minimal requirements are usually sufficient.
- For evaluation metrics and visualization, use full/dev:
  - `ragas`, `matplotlib`, `pyvis`, `pandas`

Install:
```
pip install -r requirements-full.txt
```

### 5) BunnyRAG pipeline
**What it is**
- End-to-end BunnyRAG chain and evaluation.

**Files**
- `Bunny_Rags/bunny_rag_chain_v1.ipynb`
  - Defines `BunnyRAGChain`
  - Runs retrieval and evaluation (cosine similarity, ROUGE)

**Dependencies**
- Minimal requirements are usually sufficient.
- For evaluation + visualization, use full/dev:
  - `ragas`, `matplotlib`, `pyvis`, `pandas`

Install:
```
pip install -r requirements-full.txt
```

### 6) LLM-based causal graph generation (optional)
**What it is**
- Builds causal graphs using an LLM.

**Files**
- `Graph_Algorithm/LLM_Casual_Graph_Gen.ipynb`

**Dependencies**
- `openai` (used for OpenRouter client)
- `transformers`
- `torch`
- Plus the builder dependencies

Install:
```
pip install openai transformers
pip install -r requirements-full.txt
```

### 7) Evaluation outputs (examples)
**What it is**
- Example outputs from RAG runs.

**Files**
- `Graph_Algorithm/rag_output*.txt`
- `Bunny_Rags/rag_output_with_context.txt`
- `Graph_Algorithm/causal_*.(json|graphml)`
- `Bunny_Rags/causal_*.(json|graphml)`

### 8) Tests
**What it is**
- Smoke coverage for baseline graph loading and parameterized v2 chains.

**Files**
- `tests/test_smoke.py`
- `tests/test_smoke_v2.py` (Bunny + Causal v2 chain modules)
- `tests/bunny_lambda_sweep.py` (BunnyRAG lambda sweep experiment)

Run:
```
.venv\Scripts\python -m pytest -q tests/test_smoke.py
```

Run v2 smoke:
```
.venv\Scripts\python -m pytest -q tests/test_smoke_v2.py
```

Run synthetic-only test suite:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_synth_tests.ps1
```
The script deletes its per-run pytest temp folder automatically when tests pass.

Run lambda sweep experiment:
```
python tests/bunny_lambda_sweep.py
```

Outputs are written to:
- `tests/output/bunny_lambda_selected_nodes.json`
- `tests/output/causal_topk_selected_nodes.json`
- `tests/output/bunny_lambda_sweep_summary.csv`
- `tests/output/bunny_vs_causal_topk_overlap.csv`
- `tests/output/bunny_lambda_sweep_report.txt`
- `tests/output/bunny_lambda_top10_component_terms.csv`

### 9) Bunny Retriever Update (PR2)
Recent retrieval updates in `Bunny_Rags/bunny_retriever.py`:

- Seed/source selection now uses highest cosine similarity to the query.
- The MMR-like stage uses a utility score (higher is better):
  - Conductance reward: average normalized conductance to seed nodes
  - Semantic penalty: average seed cosine similarity scaled by `labda`

Implemented form:
```
score(v) = (1/|S|) * sum_s C_norm(v,s) - labda * (1/|S|) * sum_s cos(e_v, e_s)
```

With:
- `C(v,s) = 1 / R(v,s)` where `R` is effective resistance
- Infinite/non-finite resistance gives no reward (`C=0`)
- Near-zero resistance raises an error (guard against invalid scoring)
- Candidates are ranked by highest score first

## Install Guide (Quick)

Minimal BunnyRAG/CausalRAG runtime:
```
pip install -r requirements.txt
```

Full/dev (evaluation + visualization + extras):
```
pip install -r requirements-full.txt
```

Data scraping:
```
pip install wikipedia-api
```

## Notes
- The canonical end-to-end notebook for PR1 is `Bunny_Rags/bunny_rag_chain_v1.ipynb`.
- `Graph_Algorithm/requirements.txt` is for that submodule only (see `AGENTS.md`).
- Parameterized notebook variants:
  - `Bunny_Rags/bunny_rag_chain_v2.ipynb` (imports `bunny_chain.py`)
  - `Graph_Algorithm/casual_rag_chain_v2.ipynb` (imports `causal_chain.py`)
- PR3 technical note (includes methodology limitation and recommended fix path):
  - `docs/pr3-notes.md`

## Website Demo (GitHub Pages)

Build a Google Sites-friendly Plotly bundle (small HTML + sidecar JSON files) into `docs/portfolio/`:

```powershell
.venv\Scripts\python presentation/build_interactive_projection_google_sites.py --output-html docs/portfolio/interactive_projection.html
```

This writes:
- `docs/portfolio/interactive_projection.html`
- `docs/portfolio/interactive_projection.vector.json`
- `docs/portfolio/interactive_projection.graph.json`

`docs/index.html` embeds this interactive page and is intended as the GitHub Pages landing page.

## CauseNet Sample (Prebuilt Dataset)

Downloaded file:
- `data/external/causenet/causenet-sample.json`

Converter script:
- `Data_generation/convert_causenet_sample_to_bunny.py`

Generate Bunny-compatible graph JSON:
```bash
python "Data_generation/convert_causenet_sample_to_bunny.py"
```

Output:
- `Bunny_Rags/causenet_sample_bunny_graph.json`

## Parameterized Chain Modules

To make dataset swaps easier (for example, testing a different causal graph JSON), use these modules instead of hardcoding notebook paths:

- `Bunny_Rags/bunny_chain.py`
- `Graph_Algorithm/causal_chain.py`

Both support variable input paths for:
- Causal graph JSON (nodes/edges)
- Text chunk JSON (`raw_text`-style chunk files)

### BunnyRAG example
```python
import sys
sys.path.insert(0, r"Bunny_Rags")

from bunny_chain import BunnyRAGChain

chain = BunnyRAGChain(
    graph_path=r"Bunny_Rags/causal_math_graph_llm.json",
    knowledge_base_path=r"Data_generation/wiki_math_knowledge_base_api.json",
)
result = chain.explore_and_query(
    query="What happens when the circumcenter is on the side of the triangle?",
    top_k=5,
    labda=0.02,
)
print(result["results"][:2])
```

### Causal/GraphRAG example
```python
import sys
sys.path.insert(0, r"Graph_Algorithm")

from causal_chain import CausalRAGChain

chain = CausalRAGChain(
    graph_state_path=r"Graph_Algorithm/causal_math_graph_llm.json",
    knowledge_base_path=r"Data_generation/wiki_math_knowledge_base_api.json",
)
result = chain.run("Explain causal links around triangle centers.")
print(result["paths"][:2])
```
