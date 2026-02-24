from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_generate(
    output_graph: Path,
    output_vectors: Path,
    *,
    n: int = 30,
    dim: int = 4,
    scale_prob: float = 1.0,
    seed: int = 42,
    plot_path: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    script = _repo_root() / "synthetic_bunny" / "generate_synthetic_data.py"
    cmd = [
        sys.executable,
        str(script),
        "--n",
        str(n),
        "--dim",
        str(dim),
        "--scale-prob",
        str(scale_prob),
        "--seed",
        str(seed),
        "--output-path",
        str(output_graph),
        "--vectors-output-path",
        str(output_vectors),
    ]
    if plot_path is not None:
        cmd.extend(["--plot-path", str(plot_path)])

    env = dict(os.environ)
    # Use a repo-local matplotlib config dir so tests don't fail on locked home dirs.
    env["MPLCONFIGDIR"] = str(_repo_root() / "temp" / "matplotlib_test")

    return subprocess.run(
        cmd,
        check=check,
        cwd=str(_repo_root()),
        capture_output=True,
        text=True,
        env=env,
    )


def test_generate_outputs_and_schema(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"
    vectors_path = tmp_path / "vectors.json"

    _run_generate(graph_path, vectors_path, n=30, dim=4, scale_prob=1.0, seed=123)

    assert graph_path.exists()
    assert vectors_path.exists()

    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    vectors = json.loads(vectors_path.read_text(encoding="utf-8"))

    assert isinstance(graph.get("nodes"), dict)
    assert len(graph["nodes"]) == 30
    assert isinstance(graph.get("edges"), list)
    assert len(graph["edges"]) > 0

    first_edge = graph["edges"][0]
    assert isinstance(first_edge, list)
    assert len(first_edge) == 3
    assert isinstance(first_edge[2], dict)
    assert "weight" in first_edge[2]

    assert vectors["n"] == 30
    assert vectors["dim"] == 4
    assert vectors["seed"] == 123
    assert isinstance(vectors.get("vectors"), list)
    assert len(vectors["vectors"]) == 30
    assert len(vectors["vectors"][0]) == 4


def test_generate_reproducible_with_same_seed(tmp_path: Path) -> None:
    graph_a = tmp_path / "graph_a.json"
    vectors_a = tmp_path / "vectors_a.json"
    graph_b = tmp_path / "graph_b.json"
    vectors_b = tmp_path / "vectors_b.json"

    kwargs = dict(n=40, dim=5, scale_prob=0.5, seed=77)
    _run_generate(graph_a, vectors_a, **kwargs)
    _run_generate(graph_b, vectors_b, **kwargs)

    graph_payload_a = json.loads(graph_a.read_text(encoding="utf-8"))
    graph_payload_b = json.loads(graph_b.read_text(encoding="utf-8"))
    vectors_payload_a = json.loads(vectors_a.read_text(encoding="utf-8"))
    vectors_payload_b = json.loads(vectors_b.read_text(encoding="utf-8"))

    assert graph_payload_a == graph_payload_b
    assert vectors_payload_a == vectors_payload_b


def test_generate_changes_with_different_seed(tmp_path: Path) -> None:
    graph_a = tmp_path / "graph_a.json"
    vectors_a = tmp_path / "vectors_a.json"
    graph_b = tmp_path / "graph_b.json"
    vectors_b = tmp_path / "vectors_b.json"

    _run_generate(graph_a, vectors_a, n=35, dim=4, scale_prob=0.7, seed=1)
    _run_generate(graph_b, vectors_b, n=35, dim=4, scale_prob=0.7, seed=2)

    vectors_payload_a = json.loads(vectors_a.read_text(encoding="utf-8"))
    vectors_payload_b = json.loads(vectors_b.read_text(encoding="utf-8"))
    graph_payload_a = json.loads(graph_a.read_text(encoding="utf-8"))
    graph_payload_b = json.loads(graph_b.read_text(encoding="utf-8"))

    assert vectors_payload_a["vectors"] != vectors_payload_b["vectors"]
    assert graph_payload_a["edges"] != graph_payload_b["edges"]


def test_generate_disconnected_graph_fails(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"
    vectors_path = tmp_path / "vectors.json"

    proc = _run_generate(
        graph_path,
        vectors_path,
        n=200,
        dim=6,
        scale_prob=0.001,
        seed=42,
        check=False,
    )

    assert proc.returncode != 0
    assert "Generated graph is disconnected" in (proc.stdout + proc.stderr)


def test_generate_invalid_inputs_fail(tmp_path: Path) -> None:
    cases = [
        (0, 4, 0.1, "n must be positive"),
        (10, 1, 0.1, "dim must be at least 2"),
        (10, 4, -0.1, "scale_prob must be non-negative"),
    ]
    for i, (n, dim, scale_prob, expected_error) in enumerate(cases, start=1):
        proc = _run_generate(
            tmp_path / f"invalid_{i}_graph.json",
            tmp_path / f"invalid_{i}_vectors.json",
            n=n,
            dim=dim,
            scale_prob=scale_prob,
            seed=42,
            check=False,
        )
        assert proc.returncode != 0
        assert expected_error in (proc.stdout + proc.stderr)


def test_generate_plot_path_writes_png(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"
    vectors_path = tmp_path / "vectors.json"
    plot_path = tmp_path / "graph_plot.png"

    _run_generate(
        graph_path,
        vectors_path,
        n=40,
        dim=4,
        scale_prob=0.2,
        seed=42,
        plot_path=plot_path,
    )

    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
