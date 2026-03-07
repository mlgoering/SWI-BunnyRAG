from pathlib import Path


def test_smoke_load_graph(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    bunny_dir = repo_root / "Bunny_Rags"
    assert bunny_dir.exists(), f"Missing directory: {bunny_dir}"

    monkeypatch.syspath_prepend(str(bunny_dir))
    from builder import CausalGraphBuilder  # noqa: E402

    graph_path = bunny_dir / "causal_math_graph_llm.json"
    assert graph_path.exists(), f"Missing graph file: {graph_path}"

    builder = CausalGraphBuilder()
    builder.load(str(graph_path))  # don't over-specify return value

    graph = builder.get_graph()
    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0
