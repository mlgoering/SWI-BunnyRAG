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


def _prepare_sweep_outputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    graph_path = tmp_path / "graph.json"
    vectors_path = tmp_path / "vectors.json"
    generate_script = _repo_root() / "synthetic_bunny" / "generate_synthetic_data.py"
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

    sweep_dir = tmp_path / "lambda_sweep"
    sweep_script = _repo_root() / "synthetic_bunny" / "synthetic_lambda_sweep.py"
    _run(
        [
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
            "6",
            "--lambdas",
            "0,0.2,0.4",
            "--graphrag-max-distance",
            "6.0",
            "--output-dir",
            str(sweep_dir),
        ]
    )
    return (
        graph_path,
        sweep_dir / "synthetic_bunny_lambda_selected_nodes.json",
        sweep_dir / "synthetic_graphrag_topk_selected_nodes.json",
    )


def _prepare_sweep_outputs_all_variants(tmp_path: Path) -> tuple[Path, Path, Path]:
    graph_path = tmp_path / "graph_all.json"
    vectors_path = tmp_path / "vectors_all.json"
    generate_script = _repo_root() / "synthetic_bunny" / "generate_synthetic_data.py"
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

    sweep_dir = tmp_path / "lambda_sweep_all"
    sweep_script = _repo_root() / "synthetic_bunny" / "synthetic_lambda_sweep.py"
    _run(
        [
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
            "6",
            "--lambdas",
            "0,0.2,0.4",
            "--graphrag-max-distance",
            "6.0",
            "--edge-weight-mode",
            "all",
            "--output-dir",
            str(sweep_dir),
        ]
    )
    return (
        graph_path,
        vectors_path,
        sweep_dir / "synthetic_lambda_sweep_variants.json",
    )


def test_visualize_lambda_sweep_writes_html(tmp_path: Path) -> None:
    graph_path, selected_nodes_path, graphrag_path = _prepare_sweep_outputs(tmp_path)
    out_html = tmp_path / "viz.html"
    script = _repo_root() / "synthetic_bunny" / "visualize_lambda_sweep.py"

    proc = _run(
        [
            sys.executable,
            str(script),
            "--graph-path",
            str(graph_path),
            "--selected-nodes-path",
            str(selected_nodes_path),
            "--graphrag-path",
            str(graphrag_path),
            "--output-html",
            str(out_html),
            "--cluster-separation",
            "0.1",
            "--title",
            "Test Visualization",
        ]
    )
    assert proc.returncode == 0
    assert out_html.exists()
    html = out_html.read_text(encoding="utf-8")
    assert "Plotly.newPlot" in html
    assert "GraphRAG" in html
    assert ("\\u03bb = +0.000" in html) or ("λ = +0.000" in html)
    assert ("\\u03bb = +0.200" in html) or ("λ = +0.200" in html)
    assert ("\\u03bb = +0.400" in html) or ("λ = +0.400" in html)


def test_visualize_lambda_sweep_missing_files_fail(tmp_path: Path) -> None:
    script = _repo_root() / "synthetic_bunny" / "visualize_lambda_sweep.py"
    proc = _run(
        [
            sys.executable,
            str(script),
            "--graph-path",
            str(tmp_path / "missing_graph.json"),
            "--selected-nodes-path",
            str(tmp_path / "missing_selected.json"),
            "--graphrag-path",
            str(tmp_path / "missing_graphrag.json"),
            "--output-html",
            str(tmp_path / "out.html"),
        ],
        check=False,
    )
    assert proc.returncode != 0


def test_visualize_lambda_sweep_malformed_graph_fails(tmp_path: Path) -> None:
    # Graph has wrong schema type for nodes/edges.
    bad_graph = tmp_path / "bad_graph.json"
    bad_graph.write_text('{"nodes":[],"edges":{}}', encoding="utf-8")

    selected = tmp_path / "selected.json"
    selected.write_text('{"0.000":[]}', encoding="utf-8")
    graphrag = tmp_path / "graphrag.json"
    graphrag.write_text('{"seed_nodes":[],"nodes":[]}', encoding="utf-8")

    script = _repo_root() / "synthetic_bunny" / "visualize_lambda_sweep.py"
    proc = _run(
        [
            sys.executable,
            str(script),
            "--graph-path",
            str(bad_graph),
            "--selected-nodes-path",
            str(selected),
            "--graphrag-path",
            str(graphrag),
            "--output-html",
            str(tmp_path / "out.html"),
        ],
        check=False,
    )
    assert proc.returncode != 0
    assert "Graph JSON must contain 'nodes' (dict) and 'edges' (list)." in (
        proc.stdout + proc.stderr
    )


def test_visualize_lambda_sweep_variants_mode_writes_html(tmp_path: Path) -> None:
    graph_path, vectors_path, variants_path = _prepare_sweep_outputs_all_variants(tmp_path)
    out_html = tmp_path / "viz_variants.html"
    script = _repo_root() / "synthetic_bunny" / "visualize_lambda_sweep.py"

    proc = _run(
        [
            sys.executable,
            str(script),
            "--graph-path",
            str(graph_path),
            "--variants-path",
            str(variants_path),
            "--vectors-path",
            str(vectors_path),
            "--output-html",
            str(out_html),
        ]
    )
    assert proc.returncode == 0
    assert out_html.exists()
    html = out_html.read_text(encoding="utf-8")
    assert "3D Vector Space" in html
    assert "Graph: random" in html
    assert "Graph: cosine_beta" in html
    assert "Graph: query_aware" in html
    assert "GraphRAG" in html


def test_visualize_lambda_sweep_empty_mode_payload_still_writes_html(tmp_path: Path) -> None:
    # Valid minimal graph.
    graph_payload = {"nodes": {"0": "0", "1": "1"}, "edges": [["0", "1", {"weight": 1.0}]]}
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph_payload), encoding="utf-8")

    # Empty selected payload + graphrag nodes missing node_id => GraphRAG mode
    # still exists (with zero selected nodes), so visualization should still render.
    selected_path = tmp_path / "selected.json"
    selected_path.write_text("{}", encoding="utf-8")
    graphrag_path = tmp_path / "graphrag.json"
    graphrag_path.write_text('{"seed_nodes":[],"nodes":[{}]}', encoding="utf-8")

    script = _repo_root() / "synthetic_bunny" / "visualize_lambda_sweep.py"
    proc = _run(
        [
            sys.executable,
            str(script),
            "--graph-path",
            str(graph_path),
            "--selected-nodes-path",
            str(selected_path),
            "--graphrag-path",
            str(graphrag_path),
            "--output-html",
            str(tmp_path / "out.html"),
        ],
        check=False,
    )
    assert proc.returncode == 0
    html = (tmp_path / "out.html").read_text(encoding="utf-8")
    assert "GraphRAG" in html
