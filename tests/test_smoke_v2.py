from pathlib import Path


def test_smoke_bunny_and_causal_v2(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    bunny_dir = repo_root / "Bunny Rags"
    graph_dir = repo_root / "Graph Algorithm"
    kb_path = repo_root / "Data generation" / "wiki_math_knowledge_base_api.json"

    assert bunny_dir.exists(), f"Missing directory: {bunny_dir}"
    assert graph_dir.exists(), f"Missing directory: {graph_dir}"
    assert kb_path.exists(), f"Missing KB file: {kb_path}"

    bunny_graph = bunny_dir / "causal_math_graph_llm.json"
    causal_graph = graph_dir / "causal_math_graph_llm.json"
    assert bunny_graph.exists(), f"Missing Bunny graph: {bunny_graph}"
    assert causal_graph.exists(), f"Missing Causal graph: {causal_graph}"

    monkeypatch.syspath_prepend(str(bunny_dir))
    monkeypatch.syspath_prepend(str(graph_dir))

    from bunny_chain import BunnyRAGChain  # noqa: E402
    from causal_chain import CausalRAGChain  # noqa: E402

    bunny_chain = BunnyRAGChain(
        graph_path=str(bunny_graph),
        knowledge_base_path=str(kb_path),
    )
    bunny_result = bunny_chain.explore_and_query(
        query="What happens when the circumcenter is on the side of the triangle?",
        top_k=3,
        labda=0.02,
        include_context=False,
    )

    assert bunny_result["graph_path"] == str(bunny_graph)
    assert bunny_result["knowledge_chunks_loaded"] > 0
    assert isinstance(bunny_result["results"], list)

    causal_chain = CausalRAGChain(
        graph_state_path=str(causal_graph),
        knowledge_base_path=str(kb_path),
    )
    causal_result = causal_chain.run(
        "What happens when the circumcenter is on the side of the triangle?"
    )

    assert causal_result["knowledge_chunks_loaded"] > 0
    assert "context_text" in causal_result
    assert "final_prompt" in causal_result
