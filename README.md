# SWI-BunnyRAG

This repo is a research prototype that collects Wikipedia math articles, chunks them, builds a causal graph, and runs two RAG variants:
`CausalRAG` (in `Graph Algorithm/`) and `BunnyRAG` (in `Bunny Rags/`).

Below is a plain-English map of the repo, plus the dependencies each part needs.

## Repo Map

### 1) Wikipedia article list + scraping
**What it is**
- A roughly curated list of Wikipedia math URLs.
- A notebook that downloads article text and chunks it into a JSON knowledge base.

**Files**
- `Data generation/wiiki scaper.ipynb`
  - Contains `URL_DISCIPLINE_MAP` (the URL list).
  - Scrapes pages using `wikipediaapi`.
  - Writes:
    - `Data generation/wiki_math_knowledge_base_api.json` (chunked JSON)
    - `Data generation/wiki_data_api_math/*.md` (one Markdown file per page)

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
- `Graph Algorithm/retriever.py` (CausalRAG retriever)
- `Bunny Rags/bunny_retriever.py` (BunnyRAG retriever)

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
- `Graph Algorithm/builder.py` (CausalRAG builder)
- `Bunny Rags/builder.py` (BunnyRAG builder)

**Dependencies**
- Same as retrieval above.
- Optional extras used in some paths:
  - `pyvis` and `matplotlib` are for graph visualizations (interactive HTML or static plots).
  - `tqdm` is for progress bars during batch processing.

Install (full/dev):
```
pip install -r requirements-full.txt
```

### 4) CausalRAG pipeline (Graph Algorithm)
**What it is**
- End-to-end CausalRAG chain and evaluation.

**Files**
- `Graph Algorithm/casual rag chain v1.ipynb`
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
- `Bunny Rags/bunny rag chain v1.ipynb`
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
- `Graph Algorithm/LLM Casual Graph Gen.ipynb`

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
- `Graph Algorithm/rag_output*.txt`
- `Bunny Rags/rag_output_with_context.txt`
- `Graph Algorithm/causal_*.(json|graphml)`
- `Bunny Rags/causal_*.(json|graphml)`

### 8) Tests
**What it is**
- A small smoke test.

**Files**
- `tests/test_smoke.py`
- `tests/bunny_lambda_sweep.py` (BunnyRAG lambda sweep experiment)

Run:
```
python -m pytest -q tests/test_smoke.py
```

Run lambda sweep experiment:
```
python tests/bunny_lambda_sweep.py
```

Outputs are written to:
- `tests/output/bunny_lambda_selected_nodes.json`
- `tests/output/bunny_lambda_sweep_summary.csv`
- `tests/output/bunny_lambda_sweep_report.txt`
- `tests/output/bunny_lambda_top10_component_terms.csv`

### 9) Bunny Retriever Update (PR2)
Recent retrieval updates in `Bunny Rags/bunny_retriever.py`:

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
- The canonical end-to-end notebook for PR1 is `Bunny Rags/bunny rag chain v1.ipynb`.
- `Graph Algorithm/requirements.txt` is for that submodule only (see `AGENTS.md`).
