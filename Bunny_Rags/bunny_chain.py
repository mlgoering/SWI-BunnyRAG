import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from builder import CausalGraphBuilder
from bunny_retriever import BunnyPathRetriever

logger = logging.getLogger(__name__)


def _extract_documents_from_json(data: Any) -> List[str]:
    docs: List[str] = []
    text_keys = ("raw_text", "text", "content", "chunk_text", "chunk")

    def collect_from_item(item: Any) -> None:
        if not isinstance(item, dict):
            return
        for key in text_keys:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                docs.append(value.strip())
                break

    if isinstance(data, list):
        for item in data:
            collect_from_item(item)
        return docs

    if isinstance(data, dict):
        for key in ("documents", "chunks", "data", "items"):
            value = data.get(key)
            if isinstance(value, list):
                for item in value:
                    collect_from_item(item)
                if docs:
                    return docs

    return docs


class BunnyRAGChain:
    """BunnyRAG chain with configurable graph and text-chunk inputs."""

    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        graph_path: Optional[str] = "causal_math_graph_llm.json",
        knowledge_base_path: Optional[str] = None,
    ):
        self.builder = CausalGraphBuilder(model_name=model_name)
        self.graph_path: Optional[str] = None
        self.documents: List[str] = []
        self.retriever: Optional[BunnyPathRetriever] = None

        if graph_path:
            self.load_graph(graph_path)
        else:
            self.retriever = BunnyPathRetriever(self.builder)

        if knowledge_base_path:
            self.load_knowledge_base(knowledge_base_path)

    def load_graph(self, graph_path: str) -> bool:
        logger.info("Loading graph from %s", graph_path)
        loaded = self.builder.load(graph_path)
        self.retriever = BunnyPathRetriever(self.builder)
        if loaded is False:
            logger.warning("Failed to load graph at %s", graph_path)
            return False
        self.graph_path = graph_path
        return True

    def load_knowledge_base(self, knowledge_base_path: str, limit: Optional[int] = None) -> int:
        path = Path(knowledge_base_path)
        if not path.exists():
            raise FileNotFoundError(f"Knowledge base file not found: {knowledge_base_path}")

        data = json.loads(path.read_text(encoding="utf-8"))
        docs = _extract_documents_from_json(data)

        if limit is not None:
            docs = docs[:limit]

        self.documents = docs
        logger.info("Loaded %d text chunks from %s", len(self.documents), knowledge_base_path)
        return len(self.documents)

    def _get_context_for_node(self, node_text: str, window_size: int = 250) -> str:
        if not self.documents:
            return "No knowledge-base text loaded."

        query = node_text.lower()
        for doc in self.documents:
            doc_lower = doc.lower()
            idx = doc_lower.find(query)
            if idx != -1:
                start = max(0, idx - window_size)
                end = min(len(doc), idx + window_size)
                return f"...{doc[start:end]}..."
        return "Context not found in loaded text chunks."

    def explore_and_query(
        self,
        query: str,
        top_k: int = 5,
        labda: float = 0.02,
        include_context: bool = True,
    ) -> Dict[str, Any]:
        if not self.retriever:
            self.retriever = BunnyPathRetriever(self.builder)
        if not self.graph_path:
            raise ValueError("No graph_path set. Call load_graph(...) first.")

        ranked = self.retriever.retrieve_nodes_part2(
            query=query,
            top_k=top_k,
            labda=labda,
            json_path=self.graph_path,
        )

        results: List[Dict[str, Any]] = []
        for node_id, score in ranked:
            node_text = self.builder.node_text.get(node_id, node_id)
            entry: Dict[str, Any] = {
                "node_id": node_id,
                "node_text": node_text,
                "score": float(score),
            }
            if include_context:
                entry["context"] = self._get_context_for_node(node_text)
            results.append(entry)

        return {
            "query": query,
            "graph_path": self.graph_path,
            "knowledge_chunks_loaded": len(self.documents),
            "results": results,
        }
