from __future__ import annotations

import importlib.util
import math
import os
import random
import subprocess
import sys
import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_generator_module():
    module_path = _repo_root() / "Graph_Algorithm" / "random_spherical_graph_generator.py"
    spec = importlib.util.spec_from_file_location("random_spherical_graph_generator", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_generator_cli(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    script = _repo_root() / "Graph_Algorithm" / "random_spherical_graph_generator.py"
    env = dict(os.environ)
    env["MPLCONFIGDIR"] = str(_repo_root() / "temp" / "matplotlib_test_graph_generator_cli")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=str(_repo_root()),
        capture_output=True,
        text=True,
        check=check,
        env=env,
    )


def test_random_unit_vector_is_normalized() -> None:
    mod = _load_generator_module()
    rng = random.Random(123)
    vec = mod.random_unit_vector(6, rng)

    norm = math.sqrt(sum(x * x for x in vec))
    assert len(vec) == 6
    assert abs(norm - 1.0) < 1e-9
    assert all(x >= 0.0 for x in vec)


def test_random_spherical_graph_invalid_parameters_raise() -> None:
    mod = _load_generator_module()

    try:
        mod.random_spherical_graph(n=0, dim=3, scale_prob=0.1, seed=1)
        assert False, "Expected ValueError for n <= 0"
    except ValueError as exc:
        assert "n must be positive" in str(exc)

    try:
        mod.random_spherical_graph(n=10, dim=1, scale_prob=0.1, seed=1)
        assert False, "Expected ValueError for dim < 2"
    except ValueError as exc:
        assert "dim must be at least 2" in str(exc)

    try:
        mod.random_spherical_graph(n=10, dim=3, scale_prob=-0.1, seed=1)
        assert False, "Expected ValueError for negative scale_prob"
    except ValueError as exc:
        assert "scale_prob must be non-negative" in str(exc)


def test_random_spherical_graph_reproducible_and_bidirectional() -> None:
    mod = _load_generator_module()

    vectors_a, edges_a = mod.random_spherical_graph(
        n=30, dim=4, scale_prob=0.5, seed=99, bidirectional=True
    )
    vectors_b, edges_b = mod.random_spherical_graph(
        n=30, dim=4, scale_prob=0.5, seed=99, bidirectional=True
    )
    assert vectors_a == vectors_b
    assert edges_a == edges_b

    edge_set = {(src, dst, float(w)) for src, dst, w in edges_a}
    for src, dst, w in edges_a:
        assert (dst, src, float(w)) in edge_set


def test_random_spherical_graph_not_bidirectional_when_disabled() -> None:
    mod = _load_generator_module()
    _, edges = mod.random_spherical_graph(n=25, dim=4, scale_prob=1.0, seed=21, bidirectional=False)

    edge_set = {(src, dst) for src, dst, _ in edges}
    for src, dst, _ in edges:
        assert (dst, src) not in edge_set


def test_to_bunny_graph_json_and_component_sizes() -> None:
    mod = _load_generator_module()
    edges = [("0", "1", 0.8), ("1", "0", 0.8), ("2", "3", 0.4)]

    payload = mod.to_bunny_graph_json(4, edges)
    assert isinstance(payload["nodes"], dict)
    assert len(payload["nodes"]) == 4
    assert isinstance(payload["edges"], list)
    assert payload["edges"][0][2]["weight"] == 0.8

    sizes = mod.component_sizes(4, edges)
    assert sizes == [2, 2]


def test_visualize_graph_writes_png(tmp_path: Path) -> None:
    mod = _load_generator_module()
    vectors, edges = mod.random_spherical_graph(
        n=25, dim=4, scale_prob=0.8, seed=7, bidirectional=True
    )
    assert len(vectors) == 25

    mpl_config_dir = _repo_root() / "temp" / "matplotlib_test_graph_generator"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    old_mpl = os.environ.get("MPLCONFIGDIR")
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)
    try:
        out_png = tmp_path / "graph.png"
        mod.visualize_graph(
            n=25,
            edges=edges,
            output_path=str(out_png),
            title="test-graph",
            inter_community_weight_scale=0.2,
        )
    finally:
        if old_mpl is None:
            del os.environ["MPLCONFIGDIR"]
        else:
            os.environ["MPLCONFIGDIR"] = old_mpl

    assert out_png.exists()
    assert out_png.stat().st_size > 0


def test_main_cli_writes_graph_and_vectors_json(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"
    vectors_path = tmp_path / "vectors.json"

    proc = _run_generator_cli(
        [
            "--n",
            "30",
            "--dim",
            "4",
            "--scale-prob",
            "1.0",
            "--seed",
            "42",
            "--output-path",
            str(graph_path),
            "--vectors-output-path",
            str(vectors_path),
            "--no-bidirectional",
        ]
    )

    assert proc.returncode == 0
    assert graph_path.exists()
    assert vectors_path.exists()

    graph_payload = json.loads(graph_path.read_text(encoding="utf-8"))
    vectors_payload = json.loads(vectors_path.read_text(encoding="utf-8"))
    assert len(graph_payload["nodes"]) == 30
    assert isinstance(graph_payload["edges"], list)
    assert vectors_payload["n"] == 30
    assert vectors_payload["dim"] == 4
    assert vectors_payload["seed"] == 42
    assert "Wrote graph:" in proc.stdout
    assert "Wrote vectors:" in proc.stdout


def test_main_cli_fails_for_disconnected_graph(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"
    vectors_path = tmp_path / "vectors.json"

    proc = _run_generator_cli(
        [
            "--n",
            "20",
            "--dim",
            "4",
            "--scale-prob",
            "0",
            "--seed",
            "42",
            "--output-path",
            str(graph_path),
            "--vectors-output-path",
            str(vectors_path),
        ],
        check=False,
    )

    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "Generated graph is disconnected" in combined
