# AGENTS.md

## Project intent
This repo is a research prototype that we are cleaning up into a reproducible, runnable codebase.
Prefer minimal diffs and small, reviewable changes.

## Setup (source of truth)
- Prefer Python 3.12. If installation fails due to dependency wheels, use Python 3.11.
- Avoid Python 3.14+ unless all dependencies install cleanly.

## Install
- Primary (main-path minimal): `pip install -r requirements.txt`
- Optional (dev/full): `pip install -r requirements-full.txt`

## Notes
- `Graph Algorithm/requirements.txt` is for that submodule only; do not use it for BunnyRAG unless explicitly working in that folder.

## Verification ladder (run in this order after changes)
Note: Windows commands shown; adapt for macOS/Linux.

### V0 — Repo sanity
- `git status` should be clean before starting, or changes should be understood.
- Confirm you are at repo root (the folder containing `AGENTS.md`).

### V1 — Fresh environment setup (preferred for PRs that touch dependencies)
On Windows cmd:
- Remove and recreate venv:
  - `rmdir /s /q .venv`
  - `py -3.12 -m venv .venv`  (fallback: `py -3.11 -m venv .venv`)
  - `.venv\Scripts\activate`
- Upgrade pip:
  - `python -m pip install --upgrade pip`

### V2 — Install minimal dependencies (source of truth)
- `pip install -r requirements.txt`

If working on dev/full tooling only:
- `pip install -r requirements-full.txt`

- Do not add new packages to `requirements.txt` unless required for the smoke test or canonical main path.


### V3 — Run tests (must pass)
- Primary:
  - `python -m pytest -q`
- If the repo only has a smoke test:
  - `python -m pytest -q tests/test_smoke.py`

### V4 — Run the canonical main path 
- Not yet defined.
- Do not invent a new pipeline/entrypoint. Ask which notebook/script should be treated as the main demo before adding new run commands.

## Archived / future verification (DO NOT FOLLOW YET)
- Future V4 idea: run a canonical demo entrypoint (script/module) after tests.
- Not currently defined; do not invent one.
- When defined, run the smallest reproducible end-to-end command for BunnyRAG and report outcome.
- If there is a script entrypoint, prefer it over notebooks:
  - `python <PATH_TO_ENTRYPOINT>.py`  (or `python -m <module>`)
- If only a notebook exists, document the notebook path and the minimal steps to run it.

### V5 — Report
After verification, report:
- Python version (`python -V`)
- Which install command was used (requirements vs requirements-full)
- Test command executed and whether it passed
- Main path command executed (if any) and whether it succeeded

## Coding conventions
- Prefer moving reusable logic out of notebooks into importable modules under `src/` (or a package dir).
- Keep notebooks as thin "driver" examples.
- Avoid adding new dependencies unless necessary; if you add one, justify it and document it.

## Data / secrets
- Do not commit large data files, credentials, API keys, or tokens.
- Ensure `.gitignore` covers: `.venv/`, `.env`, `__pycache__/`, `.ipynb_checkpoints/`, data output dirs.

## Documentation expectations
- If you change how to install/run, update README with exact commands.
- For new scripts/entrypoints: include usage examples.

## Preferred work style
- Propose a short plan before large refactors.
- Make one logical change per PR/commit series.
- When uncertain, ask for the "main path" (the canonical notebook/script to run end-to-end).
