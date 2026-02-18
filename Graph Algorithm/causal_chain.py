import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from builder import CausalGraphBuilder
from retriever import CausalPathRetriever

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


class CausalRAGChain:
    """Graph/Causal RAG chain with configurable graph and text-chunk inputs."""

    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        graph_state_path: Optional[str] = None,
        knowledge_base_path: Optional[str] = None,
    ):
        self.logger = logging.getLogger(__name__)
        self.builder = CausalGraphBuilder(model_name=model_name, normalize_nodes=True)
        self.retriever = CausalPathRetriever(self.builder)
        self.documents: List[str] = []

        if graph_state_path:
            self.load_graph_state(graph_state_path)
        if knowledge_base_path:
            self.load_knowledge_base_text(knowledge_base_path)

    def load_graph_state(self, filepath: str) -> bool:
        self.logger.info("Loading graph state from %s", filepath)
        if not Path(filepath).exists():
            self.logger.warning("Graph file not found: %s", filepath)
            return False
        loaded = self.builder.load(filepath)
        self.retriever = CausalPathRetriever(self.builder)
        if loaded is False:
            self.logger.warning("Failed to parse graph file: %s", filepath)
            return False
        return True

    def save_graph_state(self, filepath: str) -> None:
        self.builder.save(filepath)

    def load_knowledge_base_text(self, json_path: str, limit: Optional[int] = None) -> int:
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Knowledge base file not found: {json_path}")

        data = json.loads(path.read_text(encoding="utf-8"))
        docs = _extract_documents_from_json(data)
        if limit is not None:
            docs = docs[:limit]

        self.documents = docs
        self.logger.info("Loaded %d text chunks from %s", len(self.documents), json_path)
        return len(self.documents)

    def ingest_wiki_knowledge(
        self,
        json_path: str,
        limit: Optional[int] = None,
        auto_save_path: Optional[str] = None,
    ) -> int:
        self.load_knowledge_base_text(json_path, limit=limit)
        if not self.documents:
            self.logger.warning("No text chunks available for ingestion from %s", json_path)
            return 0

        self.logger.info("Indexing %d documents into graph...", len(self.documents))
        self.builder.index_documents(self.documents, show_progress=False)
        self.retriever = CausalPathRetriever(self.builder)

        if auto_save_path:
            self.save_graph_state(auto_save_path)
        return len(self.documents)

    def _get_context_for_path(self, path: List[str], window_size: int = 300) -> str:
        if not self.documents:
            return "No knowledge-base text loaded."

        path_keywords = [node.lower() for node in path]
        best_snippet = ""
        max_matches = 0

        for doc in self.documents:
            doc_lower = doc.lower()
            matches = sum(1 for keyword in path_keywords if keyword in doc_lower)
            if matches >= 2 and matches > max_matches:
                max_matches = matches
                first_pos = doc_lower.find(path_keywords[0])
                if first_pos != -1:
                    start = max(0, first_pos - window_size)
                    end = min(len(doc), first_pos + window_size * 2)
                    best_snippet = f"...{doc[start:end]}..."

        if best_snippet:
            return best_snippet
        return "Context not found in loaded text chunks."

    def run(
        self,
        query: str,
        max_paths: int = 5,
        min_path_length: int = 2,
        max_path_length: int = 4,
    ) -> Dict[str, Any]:
        paths = self.retriever.retrieve_paths(
            query,
            max_paths=max_paths,
            min_path_length=min_path_length,
            max_path_length=max_path_length,
        )

        context_blocks: List[str] = []
        for i, path in enumerate(paths):
            arrow_chain = " -> ".join(path)
            source_snippet = self._get_context_for_path(path)
            context_blocks.append(
                f"PATH {i + 1}: {arrow_chain}\nSOURCE CONTEXT: {source_snippet}\n"
            )

        paths_context_text = "\n".join(context_blocks)
        if not paths_context_text:
            paths_context_text = "No direct causal paths found in the knowledge graph."

        prompt = f"""You are a Causal AI Expert.
Using the provided Causal Paths and their Source Context, write a coherent, detailed answer.
Do not just list the paths; weave them into a narrative explanation.

USER QUERY: {query}

=== RETRIEVED CAUSAL EVIDENCE ===
{paths_context_text}
=================================

ANSWER:"""

        return {
            "query": query,
            "paths": paths,
            "context_text": paths_context_text,
            "final_prompt": prompt,
            "knowledge_chunks_loaded": len(self.documents),
        }
