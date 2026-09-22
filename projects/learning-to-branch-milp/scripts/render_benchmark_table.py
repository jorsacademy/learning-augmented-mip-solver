from __future__ import annotations

import argparse
import re
from pathlib import Path

SUMMARY_RE = re.compile(
    r"^(?P<metric>[a-zA-Z0-9_]+): mean=(?P<mean>-?[0-9.]+) "
    r"std=(?P<std>-?[0-9.]+) ci95=±(?P<ci>-?[0-9.]+) seeds=(?P<n>\d+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render benchmark stdout as a Markdown table.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def parse_sections(text: str) -> dict[str, dict[str, tuple[float, float]]]:
    sections: dict[str, dict[str, tuple[float, float]]] = {}
    current: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections.setdefault(current, {})
            continue
        if current is None:
            continue
        match = SUMMARY_RE.match(line)
        if match:
            sections[current][match.group("metric")] = (
                float(match.group("mean")),
                float(match.group("ci")),
            )
    return sections


def fmt(value: tuple[float, float] | None, *, percent: bool = False) -> str:
    if value is None:
        return "—"
    mean, ci = value
    suffix = "%" if percent else ""
    return f"{mean:.3f} ± {ci:.3f}{suffix}"


def render(sections: dict[str, dict[str, tuple[float, float]]]) -> str:
    policies = [
        "most_fractional",
        "pseudocost",
        "reliability",
        "strong",
        "learned_mse",
        "learned_listwise",
        "learned_gnn",
    ]
    lines = [
        "# Research benchmark summary",
        "",
        "Values are seed-level means with Student-t 95% confidence-interval half-widths.",
        "",
        "| Policy | B&B nodes | LP solves | Wall sec/instance | Node reduction vs most-frac | Node reduction vs reliability |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for policy in policies:
        metrics = sections.get(policy)
        if not metrics:
            continue
        vs_mf = sections.get(f"{policy}_vs_most_fractional", {}).get(
            "paired_node_reduction_percent"
        )
        vs_rel = sections.get(f"{policy}_vs_reliability", {}).get(
            "paired_node_reduction_percent"
        )
        if policy == "most_fractional":
            vs_mf = (0.0, 0.0)
        if policy == "reliability":
            vs_rel = (0.0, 0.0)
        lines.append(
            "| "
            + " | ".join(
                [
                    policy,
                    fmt(metrics.get("nodes_processed")),
                    fmt(metrics.get("lp_solves")),
                    fmt(metrics.get("wall_seconds_per_instance")),
                    fmt(vs_mf, percent=True),
                    fmt(vs_rel, percent=True),
                ]
            )
            + " |"
        )

    agreement = sections.get("heldout_strong_branching_agreement", {})
    if agreement:
        lines.extend(
            [
                "",
                "## Held-out strong-branching top-1 agreement",
                "",
                "| Learned policy | Agreement |",
                "|---|---:|",
            ]
        )
        for name in ("learned_mse", "learned_listwise", "learned_gnn"):
            if name in agreement:
                lines.append(f"| {name} | {fmt(agreement[name], percent=False)} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    sections = parse_sections(args.input.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(sections), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
