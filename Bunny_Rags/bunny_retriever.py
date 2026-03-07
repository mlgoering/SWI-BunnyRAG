import networkx as nx
import json
import numpy as np
import scipy.sparse as sp
from typing import List, Tuple, Dict, Set, Optional, Any, Union
import torch
from sentence_transformers import util
import logging
import heapq

logger = logging.getLogger(__name__)

class BunnyPathRetriever:
    """Retrieve relevant causal paths from the causal graph"""
    
    def __init__(self, builder: 'CausalGraphBuilder'):
        """
        Initialize the causal path retriever
        
        Args:
            builder: CausalGraphBuilder containing the graph and embeddings
        """
        self.graph = builder.get_graph()
        self.node_embeddings = builder.node_embeddings
        self.node_text = builder.node_text
        self.encoder = builder.encoder
        self.builder = builder

    def retrieve_nodes(self, 
                       query: str, 
                       top_k: int = 5, 
                       threshold: float = 0.5) -> List[Tuple[str, float]]:
        """Retrieve the most semantically relevant nodes to a query"""
        if not self.encoder or not self.node_embeddings:
            logger.warning("Encoder or node embeddings not available")
            return []
            
        try:
            # Encode query
            q_emb = self.encoder.encode(query, convert_to_tensor=True)
            
            # Calculate similarity with all nodes
            scores = {}
            for node_id, emb in self.node_embeddings.items():
                try:
                    sim = util.pytorch_cos_sim(q_emb, emb).item()
                    if sim >= threshold:
                        scores[node_id] = sim
                except Exception as e:
                    logger.error(f"Error calculating similarity for node {node_id}: {e}")
            
            # Sort by similarity score (highest cosine similarity first)
            return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
        except Exception as e:
            logger.error(f"Error retrieving nodes: {e}")
            return []

    # def retrieve_path_nodes(self, 
    #                         query: str, 
    #                         top_k: int = 5, 
    #                         max_hops: int = 2,
    #                         include_similar: bool = True) -> List[str]:
    #     """Retrieve a set of nodes related to the query through causal paths"""
    #     top_nodes = self.retrieve_nodes(query, top_k=top_k)
    #     path_nodes = set()
    #     seed_nodes = [node_id for node_id, _ in top_nodes]
        
    #     for node_id in seed_nodes:
    #         path_nodes.add(node_id)
    #         path_nodes.update(self._get_descendants(node_id, max_hops))
    #         path_nodes.update(self._get_ancestors(node_id, max_hops))
        
    #     if include_similar and len(seed_nodes) > 0 and self.encoder:
    #         try:
    #             seed_embs = [self.node_embeddings[n] for n in seed_nodes if n in self.node_embeddings]
    #             if seed_embs:
    #                 avg_emb = torch.mean(torch.stack(seed_embs), dim=0)
    #                 for node_id, emb in self.node_embeddings.items():
    #                     if node_id not in path_nodes:
    #                         sim = util.pytorch_cos_sim(avg_emb, emb).item()
    #                         if sim > 0.8:
    #                             path_nodes.add(node_id)
    #         except Exception as e:
    #             logger.error(f"Error including similar nodes: {e}")
        
    #     return list(path_nodes)

    # def _get_descendants(self, node: str, max_hops: int) -> Set[str]:
    #     """Get all descendants within max_hops"""
    #     descendants = set()
    #     current = {node}
    #     for _ in range(max_hops):
    #         next_nodes = set()
    #         for n in current:
    #             try:
    #                 next_nodes.update(self.graph.successors(n))
    #             except Exception:
    #                 pass
    #         descendants.update(next_nodes)
    #         current = next_nodes
    #         if not current:
    #             break
    #     return descendants

    # def _get_ancestors(self, node: str, max_hops: int) -> Set[str]:
    #     """Get all ancestors within max_hops"""
    #     ancestors = set()
    #     current = {node}
    #     for _ in range(max_hops):
    #         next_nodes = set()
    #         for n in current:
    #             try:
    #                 next_nodes.update(self.graph.predecessors(n))
    #             except Exception:
    #                 pass
    #         ancestors.update(next_nodes)
    #         current = next_nodes
    #         if not current:
    #             break
    #     return ancestors

    def build_effective_resistance(self, json_path: str):
        """Helper to compute effective resistance matrix R"""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        labels = list(data["nodes"].keys())
        label_to_id = {lab: i for i, lab in enumerate(labels)}
        n = len(labels)
        
        rows, cols, vals = [], [], []
        for src, dst, meta in data["edges"]:
            w = float(meta.get("weight", 1.0))
            i, j = label_to_id[src], label_to_id[dst]
            rows.append(i)
            cols.append(j)
            vals.append(w)
        
        A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        A_und = A + A.T
        A_und.setdiag(0)
        A_und.eliminate_zeros()
        
        deg = np.asarray(A_und.sum(axis=1)).ravel()
        L = sp.diags(deg) - A_und
        L_dense = L.toarray()
        L_plus = np.linalg.pinv(L_dense)
        
        diag = np.diag(L_plus)
        R = diag[:, None] + diag[None, :] - 2 * L_plus
        return label_to_id, R

    def retrieve_nodes_part2(self, 
                             query: str, 
                             top_k: int = 15,
                             labda: float = 0, 
                             json_path: str = "causal_math_graph_llm.json") -> List[Tuple[str, float]]:
        """Advanced retrieval using Effective Resistance and Semantic Similarity"""
        if not self.encoder or not self.node_embeddings:
            return []

        label_to_id, R = self.build_effective_resistance(json_path)
        seed_list = self.retrieve_nodes(query)
        seed_ids = [
            node_sp for node_sp, _ in seed_list
            if node_sp in label_to_id and node_sp in self.node_embeddings
        ]

        if not seed_ids:
            return []

        seed_set = set(seed_ids)
        candidate_pair_data = {}
        all_raw_conductances = []
        resistance_zero_eps = 1e-12

        # First pass: compute pairwise raw conductance C(v,s)=1/R(v,s) and semantic similarity.
        for node_id, emb in self.node_embeddings.items():
            if node_id not in label_to_id or node_id in seed_set:
                # Do not score selected source nodes themselves.
                continue

            i = label_to_id[node_id]
            pair_data = []

            for node_sp in seed_ids:
                j = label_to_id[node_sp]
                resistance = float(R[i, j])

                # Infinity resistance => no reward.
                if not np.isfinite(resistance):
                    raw_conductance = 0.0
                elif resistance <= resistance_zero_eps:
                    raise ValueError(
                        f"Zero effective resistance encountered between candidate '{node_id}' "
                        f"and source '{node_sp}'."
                    )
                else:
                    raw_conductance = 1.0 / resistance
                    all_raw_conductances.append(raw_conductance)

                sim = util.pytorch_cos_sim(self.node_embeddings[node_sp], emb).item()
                # Keep semantic penalty bounded in [0, lambda].
                sim = max(0.0, min(1.0, sim))
                pair_data.append((raw_conductance, sim))

            candidate_pair_data[node_id] = pair_data

        if not candidate_pair_data:
            return []

        min_c = min(all_raw_conductances) if all_raw_conductances else 0.0
        max_c = max(all_raw_conductances) if all_raw_conductances else 0.0
        denom = max_c - min_c

        scores = {}
        for node_id, pair_data in candidate_pair_data.items():
            normalized_conductances = []
            similarities = []

            for raw_conductance, sim in pair_data:
                if denom > 0.0:
                    c_norm = (raw_conductance - min_c) / denom
                elif max_c > 0.0:
                    c_norm = 1.0
                else:
                    c_norm = 0.0

                normalized_conductances.append(c_norm)
                similarities.append(sim)

            avg_conductance = float(np.mean(normalized_conductances))
            avg_similarity = float(np.mean(similarities))

            # Utility score: maximize normalized conductance reward, penalize semantic similarity.
            score_i = avg_conductance - labda * avg_similarity
            scores[node_id] = score_i

        # Higher score is better in the utility framework.
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # def retrieve_paths(self, 
    #                   query: str, 
    #                   max_paths: int = 15,
    #                   min_path_length: int = 2,
    #                   max_path_length: int = 4) -> List[List[str]]:
    #     """
    #     Retrieve complete causal paths relevant to the query
        
    #     Args:
    #         query: User query
    #         max_paths: Maximum number of paths to return
    #         min_path_length: Minimum path length to include
    #         max_path_length: Maximum path length to consider
            
    #     Returns:
    #         List of paths, where each path is a list of node IDs
    #     """
    #     # Get relevant nodes
    #     relevant_nodes = self.retrieve_nodes_part2(
    #         query=query, 
    #         top_k=15, 
    #         labda=0, 
    #         json_path="causal_math_graph_llm.json" # Ensure this matches your filename
    #     )
        
    #     if len(relevant_nodes) < 2:
    #         return []
        
    #     # Find all simple paths between relevant nodes
    #     paths = []
    #     for i, src in enumerate(relevant_nodes):
    #         for tgt in relevant_nodes[i+1:]:
    #             if src != tgt:
    #                 try:
    #                     # Try to find paths in both directions
    #                     for direction in [(src, tgt), (tgt, src)]:
    #                         start, end = direction
    #                         for path in nx.all_simple_paths(
    #                             self.graph, start, end, 
    #                             cutoff=max_path_length
    #                         ):
    #                             if len(path) >= min_path_length:
    #                                 # Convert IDs to text for readability
    #                                 text_path = [self.node_text.get(n, n) for n in path]
    #                                 paths.append((path, text_path))
    #                 except (nx.NetworkXNoPath, nx.NodeNotFound):
    #                     continue
        
    #     # Sort paths by relevance (currently length, could be improved)
    #     sorted_paths = sorted(paths, key=lambda p: len(p[0]))
        
    #     # Return the text version of the paths
    #     return [text_path for _, text_path in sorted_paths[:max_paths]]

    def get_causal_explanation(self, query: str) -> str:
        # Placeholder: retrieve_paths implementation needed or uncommented
        return f"Causal explanation for {query} would go here."

    def highlight_subgraph(self, query: str) -> nx.DiGraph:
        nodes = self.retrieve_path_nodes(query)
        return self.graph.subgraph(nodes)
