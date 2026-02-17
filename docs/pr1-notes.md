# BunnyRAG (PR-1 Notes)

This PR makes a fresh clone reproducibly runnable with minimal dependencies.

## Setup
- Prefer Python 3.12. If installation fails due to dependency wheels, use Python 3.11.
- Avoid Python 3.14+ unless all dependencies install cleanly.

## Install (minimal)
```powershell
pip install -r requirements.txt
```

## Install (full/dev)
```powershell
pip install -r requirements-full.txt
```

## Main Path
- Canonical end-to-end notebook: `Bunny Rags/bunny rag chain v1.ipynb`

## Smoke Test
```powershell
python -m pytest -q tests/test_smoke.py
```

## Notes
- First run will download the sentence-transformers model weights (Hugging Face).
- This PR also cleaned up requirements: the old full list was moved to `requirements-full.txt`, and the minimal list is now the default `requirements.txt`.
- `requirements.txt` now includes the PyTorch CPU wheel index URL for more reliable installs.
- Added `pytest` to `requirements-full.txt` for running `tests/test_smoke.py`.
- On Windows you may see a Hugging Face symlink warning; it is harmless (model caching still works).
- `.gitignore` includes: `.venv/`, `venv/`, `.env`, `__pycache__/`, `*.py[cod]`, `.ipynb_checkpoints/`.
- Added `AGENTS.md` in repo root, which defines project intent, setup/install guidance, verification steps, and coding conventions.
- Canonical verification for PR-1 is `python -m pytest -q tests/test_smoke.py`.
- Updated `AGENTS.md` verification ladder: clarified V4 is not yet defined and added explicit guidance on when to add packages to `requirements.txt`.
- Refined `tests/test_smoke.py` to use `monkeypatch.syspath_prepend` and directory existence checks.
