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


def test_lambda_sweep_outputs_and_schema(tmp_path: Path) -> None:
    graph_path, vectors_path = _prepare_synthetic_inputs(tmp_path)
    out_dir = tmp_path / "lambda_sweep"
    script = _repo_root() / "synthetic_bunny" / "synthetic_lambda_sweep.py"

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
            "7",
            "--lambdas",
            "0,0.1,0.2",
            "--graphrag-max-distance",
            "6.0",
            "--output-dir",
            str(out_dir),
        ]
    )
    assert proc.returncode == 0

    selected_path = out_dir / "synthetic_bunny_lambda_selected_nodes.json"
    graphrag_path = out_dir / "synthetic_graphrag_topk_selected_nodes.json"
    report_path = out_dir / "synthetic_bunny_lambda_sweep_report.txt"
    assert selected_path.exists()
    assert graphrag_path.exists()
    assert report_path.exists()

    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    assert sorted(selected.keys(), key=float) == ["0.000", "0.100", "0.200"]
    for _, rows in selected.items():
        assert isinstance(rows, list)
        assert len(rows) <= 7
        for row in rows:
            assert {"rank", "node_id", "utility_score", "query_similarity"} <= set(row.keys())
            assert {
                "avg_normalized_conductance",
                "avg_seed_cosine",
            } <= set(row.keys())

    graphrag = json.loads(graphrag_path.read_text(encoding="utf-8"))
    assert graphrag["top_k"] == 7
    assert graphrag["max_distance"] == 6.0
    assert isinstance(graphrag.get("seed_nodes"), list)
    assert len(graphrag.get("nodes", [])) <= 7

    report = report_path.read_text(encoding="utf-8")
    assert "Lambdas: [0.0, 0.1, 0.2]" in report


def test_lambda_sweep_requires_exactly_one_query_mode(tmp_path: Path) -> None:
    graph_path, vectors_path = _prepare_synthetic_inputs(tmp_path)
    script = _repo_root() / "synthetic_bunny" / "synthetic_lambda_sweep.py"

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
            "--output-dir",
            str(tmp_path / "out_both"),
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
            "--output-dir",
            str(tmp_path / "out_neither"),
        ],
        check=False,
    )
    assert neither_proc.returncode != 0
    assert "Choose exactly one query mode" in (neither_proc.stdout + neither_proc.stderr)


def test_lambda_sweep_invalid_parameter_types_fail(tmp_path: Path) -> None:
    graph_path, vectors_path = _prepare_synthetic_inputs(tmp_path)
    script = _repo_root() / "synthetic_bunny" / "synthetic_lambda_sweep.py"

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
            "--output-dir",
            str(tmp_path / "out_bad_topk"),
        ],
        check=False,
    )
    assert bad_topk.returncode != 0

    bad_lambdas = _run(
        [
            sys.executable,
            str(script),
            "--graph-path",
            str(graph_path),
            "--vectors-path",
            str(vectors_path),
            "--query-random-points",
            "1",
            "--lambdas",
            "0,foo,0.2",
            "--output-dir",
            str(tmp_path / "out_bad_lambdas"),
        ],
        check=False,
    )
    assert bad_lambdas.returncode != 0
    assert "Expected at least one numeric value" in (bad_lambdas.stdout + bad_lambdas.stderr) or "could not convert string to float" in (bad_lambdas.stdout + bad_lambdas.stderr)


def test_lambda_sweep_missing_or_malformed_inputs_fail(tmp_path: Path) -> None:
    graph_path, vectors_path = _prepare_synthetic_inputs(tmp_path)
    script = _repo_root() / "synthetic_bunny" / "synthetic_lambda_sweep.py"

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
            "--output-dir",
            str(tmp_path / "out_missing_graph"),
        ],
        check=False,
    )
    assert missing_graph.returncode != 0

    bad_vectors = tmp_path / "bad_vectors.json"
    bad_vectors.write_text('{"vectors":"not-a-list-or-dict"}', encoding="utf-8")
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
            "--output-dir",
            str(tmp_path / "out_bad_vectors"),
        ],
        check=False,
    )
    assert malformed_vectors.returncode != 0
    assert "Vectors JSON must have 'vectors' as a list or dict" in (
        malformed_vectors.stdout + malformed_vectors.stderr
    )


def test_lambda_sweep_rejects_non_positive_seedk_topk(tmp_path: Path) -> None:
    graph_path, vectors_path = _prepare_synthetic_inputs(tmp_path)
    script = _repo_root() / "synthetic_bunny" / "synthetic_lambda_sweep.py"

    bad_seedk = _run(
        [
            sys.executable,
            str(script),
            "--graph-path",
            str(graph_path),
            "--vectors-path",
            str(vectors_path),
            "--query-random-points",
            "1",
            "--seed-k",
            "0",
            "--output-dir",
            str(tmp_path / "out_bad_seedk"),
        ],
        check=False,
    )
    assert bad_seedk.returncode != 0
    assert "--seed-k must be a positive integer." in (bad_seedk.stdout + bad_seedk.stderr)

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
            "-1",
            "--output-dir",
            str(tmp_path / "out_bad_topk"),
        ],
        check=False,
    )
    assert bad_topk.returncode != 0
    assert "--top-k must be a positive integer." in (bad_topk.stdout + bad_topk.stderr)


def test_lambda_sweep_reproducible_for_same_inputs_and_seeds(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"
    vectors_path = tmp_path / "vectors.json"
    generate_script = _repo_root() / "synthetic_bunny" / "generate_synthetic_data.py"
    sweep_script = _repo_root() / "synthetic_bunny" / "synthetic_lambda_sweep.py"

    _run(
        [
            sys.executable,
            str(generate_script),
            "--n",
            "35",
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

    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    base_cmd = [
        sys.executable,
        str(sweep_script),
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
        "--lambdas",
        "0,0.1,0.2",
        "--graphrag-max-distance",
        "6.0",
    ]
    _run([*base_cmd, "--output-dir", str(out_a)])
    _run([*base_cmd, "--output-dir", str(out_b)])

    selected_a = json.loads((out_a / "synthetic_bunny_lambda_selected_nodes.json").read_text(encoding="utf-8"))
    selected_b = json.loads((out_b / "synthetic_bunny_lambda_selected_nodes.json").read_text(encoding="utf-8"))
    graphrag_a = json.loads((out_a / "synthetic_graphrag_topk_selected_nodes.json").read_text(encoding="utf-8"))
    graphrag_b = json.loads((out_b / "synthetic_graphrag_topk_selected_nodes.json").read_text(encoding="utf-8"))

    assert selected_a == selected_b
    assert graphrag_a == graphrag_b
