from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(_repo_root()),
        capture_output=True,
        text=True,
        check=check,
    )


def _run_behavior_smoke(tmp_path: Path, mode: str) -> Path:
    script = _repo_root() / "synthetic_bunny" / "behavior_test_runner.py"
    out_dir = tmp_path / f"behavior_{mode}"
    proc = _run(
        [
            sys.executable,
            str(script),
            "--num-datasets",
            "1",
            "--queries-per-dataset",
            "1",
            "--n",
            "60",
            "--dim",
            "4",
            "--scale-prob",
            "0.6",
            "--vector-space",
            "sphere",
            "--seed-community-policy",
            "mixed",
            "--seed-k",
            "3",
            "--top-k",
            "5",
            "--lambdas",
            "0.0",
            "--random-trials",
            "3",
            "--max-generation-attempts",
            "5",
            "--max-query-attempts-per-dataset",
            "200",
            "--edge-weight-mode",
            mode,
            "--output-dir",
            str(out_dir),
        ]
    )
    assert proc.returncode == 0
    return out_dir


def test_behavior_runner_smoke_random_mode(tmp_path: Path) -> None:
    out_dir = _run_behavior_smoke(tmp_path, "random")
    rows_csv = out_dir / "behavior_results_rows.csv"
    metadata_json = out_dir / "behavior_metadata.json"
    assert rows_csv.exists()
    assert metadata_json.exists()

    metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
    assert metadata["params"]["edge_weight_mode"] == "random"
    assert metadata["dataset_meta"][0]["edge_weight_mode"] == "random"


def test_behavior_runner_smoke_query_aware_mode(tmp_path: Path) -> None:
    out_dir = _run_behavior_smoke(tmp_path, "query_aware")
    rows_csv = out_dir / "behavior_results_rows.csv"
    metadata_json = out_dir / "behavior_metadata.json"
    assert rows_csv.exists()
    assert metadata_json.exists()

    metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
    assert metadata["params"]["edge_weight_mode"] == "query_aware"
    assert metadata["dataset_meta"][0]["edge_weight_mode"] == "query_aware"
