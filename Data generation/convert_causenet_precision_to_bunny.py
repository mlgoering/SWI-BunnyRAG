import argparse
import bz2
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


class DisjointSetUnion:
    def __init__(self) -> None:
        self.parent: List[int] = []
        self.rank: List[int] = []

    def add(self) -> int:
        idx = len(self.parent)
        self.parent.append(idx)
        self.rank.append(0)
        return idx

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def _clean_concept(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.strip().split())
    if not cleaned:
        return None
    return cleaned.lower()


def _extract_pair(item: Dict[str, Any]) -> Optional[Tuple[str, str, Optional[float]]]:
    relation = item.get("causal_relation", item)
    if not isinstance(relation, dict):
        return None

    cause = relation.get("cause")
    effect = relation.get("effect")

    if isinstance(cause, dict):
        cause = cause.get("concept", cause.get("text"))
    if isinstance(effect, dict):
        effect = effect.get("concept", effect.get("text"))

    cause_text = _clean_concept(cause)
    effect_text = _clean_concept(effect)
    if not cause_text or not effect_text:
        return None

    confidence = relation.get("confidence", relation.get("score", relation.get("weight")))
    confidence_value: Optional[float]
    if isinstance(confidence, (int, float)):
        confidence_value = float(confidence)
    else:
        confidence_value = None

    return cause_text, effect_text, confidence_value


def _iter_pairs_from_jsonl_bz2(
    input_path: Path,
) -> Iterable[Tuple[str, str, Optional[float]]]:
    with bz2.open(input_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            pair = _extract_pair(obj)
            if pair:
                yield pair


def _largest_component_nodes(
    input_path: Path,
    remove_self_loops: bool,
) -> Set[str]:
    dsu = DisjointSetUnion()
    node_to_id: Dict[str, int] = {}

    def get_id(node: str) -> int:
        if node not in node_to_id:
            node_to_id[node] = dsu.add()
        return node_to_id[node]

    for cause, effect, _ in _iter_pairs_from_jsonl_bz2(input_path):
        if remove_self_loops and cause == effect:
            continue
        cause_id = get_id(cause)
        effect_id = get_id(effect)
        dsu.union(cause_id, effect_id)

    if not node_to_id:
        return set()

    root_sizes: Dict[int, int] = defaultdict(int)
    for node, idx in node_to_id.items():
        root_sizes[dsu.find(idx)] += 1

    largest_root = max(root_sizes.items(), key=lambda kv: kv[1])[0]
    return {
        node
        for node, idx in node_to_id.items()
        if dsu.find(idx) == largest_root
    }


def convert_records(
    input_path: Path,
    keep_nodes: Optional[Set[str]],
    remove_self_loops: bool,
) -> Dict[str, Any]:
    nodes: Dict[str, str] = {}
    variants: Dict[str, List[str]] = {}
    edge_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    edge_confidences: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    removed_self_loops = 0
    removed_outside_component = 0

    for cause, effect, confidence in _iter_pairs_from_jsonl_bz2(input_path):
        if remove_self_loops and cause == effect:
            removed_self_loops += 1
            continue
        if keep_nodes is not None and (cause not in keep_nodes or effect not in keep_nodes):
            removed_outside_component += 1
            continue

        nodes[cause] = cause
        nodes[effect] = effect
        edge_key = (cause, effect)
        edge_counts[edge_key] += 1
        if confidence is not None:
            edge_confidences[edge_key].append(confidence)

    edges: List[List[Any]] = []
    for (cause, effect), count in edge_counts.items():
        confs = edge_confidences[(cause, effect)]
        if confs:
            weight = sum(confs) / len(confs)
        else:
            weight = float(count)
        if weight <= 0:
            continue
        edges.append([cause, effect, {"weight": float(weight)}])

    edges.sort(key=lambda e: (e[0], e[1]))
    return {
        "nodes": nodes,
        "variants": variants,
        "edges": edges,
        "_conversion_stats": {
            "removed_self_loops": removed_self_loops,
            "removed_outside_largest_component": removed_outside_component,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert CauseNet precision JSONL.BZ2 into BunnyRAG graph JSON format."
    )
    parser.add_argument(
        "--input",
        default="data/external/causenet/causenet-precision.jsonl.bz2",
        help="Path to CauseNet precision .jsonl.bz2 file.",
    )
    parser.add_argument(
        "--output",
        default="Bunny Rags/causenet_precision_bunny_graph.json",
        help="Output path for Bunny-compatible graph JSON.",
    )
    parser.add_argument(
        "--no-largest-component-filter",
        action="store_true",
        help="Disable largest connected component filtering.",
    )
    parser.add_argument(
        "--keep-self-loops",
        action="store_true",
        help="Keep self-loop edges (default removes them).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    remove_self_loops = not args.keep_self_loops
    filter_lcc = not args.no_largest_component_filter

    keep_nodes: Optional[Set[str]] = None
    if filter_lcc:
        keep_nodes = _largest_component_nodes(
            input_path=input_path,
            remove_self_loops=remove_self_loops,
        )
        if not keep_nodes:
            raise RuntimeError("Largest connected component detection found no nodes.")

    converted = convert_records(
        input_path=input_path,
        keep_nodes=keep_nodes,
        remove_self_loops=remove_self_loops,
    )

    stats = converted.pop("_conversion_stats")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(converted, indent=2), encoding="utf-8")

    print(f"Largest-component filter enabled: {filter_lcc}")
    if filter_lcc:
        print(f"Largest-component nodes retained: {len(converted['nodes'])}")
    print(f"Self-loop removal enabled: {remove_self_loops}")
    print(f"Removed self-loops: {stats['removed_self_loops']}")
    if filter_lcc:
        print(f"Removed edges outside largest component: {stats['removed_outside_largest_component']}")
    print(f"Output nodes: {len(converted['nodes'])}")
    print(f"Output edges: {len(converted['edges'])}")
    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
