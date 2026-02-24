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


def _prepare_synthetic_inputs(tmp_path: Path) -> tuple[Path, Path]:
    graph_path = tmp_path / "graph.json"
    vectors_path = tmp_path / "vectors.json"
    script = _repo_root() / "synthetic_bunny" / "generate_synthetic_data.py"
    _run(
        [
            sys.executable,
            str(script),
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
        ]
    )
    return graph_path, vectors_path


def test_synthetic_graphrag_outputs_and_schema(tmp_path: Path) -> None:
    graph_path, vectors_path = _prepare_synthetic_inputs(tmp_path)
    out_path = tmp_path / "graphrag.json"
    script = _repo_root() / "synthetic_bunny" / "synthetic_graphrag.py"

    proc = _run(
        [
            sys.executable,
            str(script),
            "--graph-path",
            str(graph_path),
            "--vectors-path",
            str(vectors_path),
            "--query-random-points",
            "1",
            "--query-seed",
            "17",
            "--seed-k",
            "3",
            "--top-k",
            "8",
            "--max-distance",
            "6.0",
            "--output-path",
            str(out_path),
        ]
    )
    assert proc.returncode == 0
    assert out_path.exists()

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["query_mode"] == "random_sphere_points"
    assert payload["seed_k"] == 3
    assert payload["top_k"] == 8
    assert payload["max_distance"] == 6.0
    assert isinstance(payload["seed_nodes"], list)
    assert isinstance(payload["results"], list)
    assert len(payload["results"]) <= 8

    seed_set = set(payload["seed_nodes"])
    for row in payload["results"]:
        assert {"rank", "node_id", "graph_distance", "query_similarity"} <= set(row.keys())
        assert row["graph_distance"] <= 6.0
        assert row["node_id"] not in seed_set


def test_synthetic_graphrag_requires_exactly_one_query_mode(tmp_path: Path) -> None:
    graph_path, vectors_path = _prepare_synthetic_inputs(tmp_path)
    script = _repo_root() / "synthetic_bunny" / "synthetic_graphrag.py"

    both_proc = _run(
        [
            sys.executable,
            str(script),
            "--graph-path",
            str(graph_path),
            "--vectors-path",
            str(vectors_path),
            "--query-random-points",
            "1",
            "--query-vertices",
            "0,1,2",
            "--output-path",
            str(tmp_path / "out_both.json"),
        ],
        check=False,
    )
    assert both_proc.returncode != 0
    assert "Choose exactly one query mode" in (both_proc.stdout + both_proc.stderr)

    neither_proc = _run(
        [
            sys.executable,
            str(script),
            "--graph-path",
            str(graph_path),
            "--vectors-path",
            str(vectors_path),
            "--output-path",
            str(tmp_path / "out_neither.json"),
        ],
        check=False,
    )
    assert neither_proc.returncode != 0
    assert "Choose exactly one query mode" in (neither_proc.stdout + neither_proc.stderr)


def test_synthetic_graphrag_invalid_parameter_types_fail(tmp_path: Path) -> None:
    graph_path, vectors_path = _prepare_synthetic_inputs(tmp_path)
    script = _repo_root() / "synthetic_bunny" / "synthetic_graphrag.py"

    bad_topk = _run(
        [
            sys.executable,
            str(script),
            "--graph-path",
            str(graph_path),
            "--vectors-path",
            str(vectors_path),
            "--query-random-points",
            "1",
            "--top-k",
            "abc",
            "--output-path",
            str(tmp_path / "out_bad_topk.json"),
        ],
        check=False,
    )
    assert bad_topk.returncode != 0

    bad_max_distance = _run(
        [
            sys.executable,
            str(script),
            "--graph-path",
            str(graph_path),
            "--vectors-path",
            str(vectors_path),
            "--query-random-points",
            "1",
            "--max-distance",
            "not-a-float",
            "--output-path",
            str(tmp_path / "out_bad_distance.json"),
        ],
        check=False,
    )
    assert bad_max_distance.returncode != 0


def test_synthetic_graphrag_missing_or_malformed_inputs_fail(tmp_path: Path) -> None:
    graph_path, vectors_path = _prepare_synthetic_inputs(tmp_path)
    script = _repo_root() / "synthetic_bunny" / "synthetic_graphrag.py"

    missing_graph = _run(
        [
            sys.executable,
            str(script),
            "--graph-path",
            str(tmp_path / "missing_graph.json"),
            "--vectors-path",
            str(vectors_path),
            "--query-random-points",
            "1",
            "--output-path",
            str(tmp_path / "out_missing_graph.json"),
        ],
        check=False,
    )
    assert missing_graph.returncode != 0

    bad_vectors = tmp_path / "bad_vectors.json"
    bad_vectors.write_text('{"vectors":"oops"}', encoding="utf-8")
    malformed_vectors = _run(
        [
            sys.executable,
            str(script),
            "--graph-path",
            str(graph_path),
            "--vectors-path",
            str(bad_vectors),
            "--query-random-points",
            "1",
            "--output-path",
            str(tmp_path / "out_bad_vectors.json"),
        ],
        check=False,
    )
    assert malformed_vectors.returncode != 0
    assert "Vectors JSON must have 'vectors' as a list or dict" in (
        malformed_vectors.stdout + malformed_vectors.stderr
    )
