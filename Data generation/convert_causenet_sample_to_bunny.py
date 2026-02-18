import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


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
    if isinstance(confidence, (int, float)):
        confidence_value: Optional[float] = float(confidence)
    else:
        confidence_value = None

    return cause_text, effect_text, confidence_value


def convert_records(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    nodes: Dict[str, str] = {}
    variants: Dict[str, List[str]] = {}
    edge_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    edge_confidences: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    for item in records:
        pair = _extract_pair(item)
        if not pair:
            continue

        cause, effect, confidence = pair
        nodes[cause] = cause
        nodes[effect] = effect

        edge_key = (cause, effect)
        edge_counts[edge_key] += 1
        if confidence is not None:
            edge_confidences[edge_key].append(confidence)

    edges: List[List[Any]] = []
    for (cause, effect), count in edge_counts.items():
        if edge_confidences[(cause, effect)]:
            weight = sum(edge_confidences[(cause, effect)]) / len(edge_confidences[(cause, effect)])
        else:
            weight = float(count)
        if weight <= 0:
            continue
        edges.append([cause, effect, {"weight": float(weight)}])

    edges.sort(key=lambda e: (e[0], e[1]))
    return {"nodes": nodes, "variants": variants, "edges": edges}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert CauseNet sample JSON into BunnyRAG graph JSON format."
    )
    parser.add_argument(
        "--input",
        default="data/external/causenet/causenet-sample.json",
        help="Path to CauseNet sample JSON.",
    )
    parser.add_argument(
        "--output",
        default="Bunny Rags/causenet_sample_bunny_graph.json",
        help="Output path for BunnyRAG-compatible graph JSON.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected input JSON to be a list of CauseNet records.")

    converted = convert_records(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(converted, indent=2), encoding="utf-8")

    print(f"Input records: {len(data)}")
    print(f"Output nodes: {len(converted['nodes'])}")
    print(f"Output edges: {len(converted['edges'])}")
    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
