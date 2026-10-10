#!/usr/bin/env python3
"""Reproduce Promise Switch primary human-vs-judge reliability statistics."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DIR = (
    ROOT / "notes" / "promise_switch" / "validation"
    / "promise_switch_message_codebook_v1" / "public"
)
FIELDS = (
    "explicit_promise",
    "commitment_present",
    "statement_type",
    "promised_action",
    "conditionality",
    "refusal_present",
    "ambiguous_response",
)
EXPECTED_IDS = tuple(f"PSV{index:03d}" for index in range(1, 131))
PUBLIC_COLUMNS = ("public_item_id", *FIELDS)


def read_public_labels(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PUBLIC_COLUMNS:
            raise ValueError(
                f"{path.name} must contain exactly these columns: {PUBLIC_COLUMNS}"
            )
        rows = list(reader)
    identifiers = [row["public_item_id"] for row in rows]
    if len(rows) != 130:
        raise ValueError(f"{path.name} must contain exactly 130 labels")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{path.name} contains duplicate public item IDs")
    if tuple(sorted(identifiers)) != EXPECTED_IDS:
        raise ValueError(f"{path.name} must contain PSV001 through PSV130 exactly once")
    for row in rows:
        if not re.fullmatch(r"PSV\d{3}", row["public_item_id"]):
            raise ValueError(f"invalid public item ID in {path.name}")
        for field in FIELDS:
            if not row[field]:
                raise ValueError(f"missing {field} label for {row['public_item_id']}")
    return rows


def cohens_kappa(first: Iterable[str], second: Iterable[str]) -> float | None:
    labels_a, labels_b = list(first), list(second)
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("kappa requires equal, non-empty matched samples")
    categories = set(labels_a) | set(labels_b)
    n = len(labels_a)
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    expected = sum(
        (labels_a.count(category) / n) * (labels_b.count(category) / n)
        for category in categories
    )
    if math.isclose(expected, 1.0, abs_tol=1e-12):
        return None
    return (observed - expected) / (1 - expected)


def calculate_reliability(
    human_rows: list[dict[str, str]], judge_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    human = {row["public_item_id"]: row for row in human_rows}
    judge = {row["public_item_id"]: row for row in judge_rows}
    if set(human) != set(judge):
        missing = sorted(set(human) - set(judge))
        extra = sorted(set(judge) - set(human))
        raise ValueError(f"human/judge public ID mismatch; missing={missing}, extra={extra}")
    output: list[dict[str, Any]] = []
    for field in FIELDS:
        human_labels = [human[item_id][field] for item_id in EXPECTED_IDS]
        judge_labels = [judge[item_id][field] for item_id in EXPECTED_IDS]
        agreement = sum(a == b for a, b in zip(human_labels, judge_labels))
        kappa = cohens_kappa(human_labels, judge_labels)
        explanation = ""
        if kappa is None:
            explanation = (
                "undefined because both coders used one identical category; "
                "expected agreement equals 1"
            )
        elif field == "ambiguous_response":
            human_yes = Counter(human_labels)["yes"]
            judge_yes = Counter(judge_labels)["yes"]
            explanation = (
                f"human coder marked {human_yes} messages ambiguous; "
                f"judge marked {judge_yes}"
            )
        output.append({
            "variable": field,
            "n": len(EXPECTED_IDS),
            "raw_agreement_count": agreement,
            "raw_agreement_percent": 100 * agreement / len(EXPECTED_IDS),
            "cohens_kappa": kappa,
            "kappa_status": "undefined_expected_agreement_one" if kappa is None else "defined",
            "explanation": explanation,
        })
    return output


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "variable", "n", "raw_agreement_count", "raw_agreement_percent",
        "cohens_kappa", "kappa_status", "explanation",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                "raw_agreement_percent": f"{row['raw_agreement_percent']:.10f}",
                "cohens_kappa": "" if row["cohens_kappa"] is None else f"{row['cohens_kappa']:.12g}",
            })


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--human-labels", type=Path, default=PUBLIC_DIR / "human_labels_n130.csv")
    cli.add_argument("--judge-labels", type=Path, default=PUBLIC_DIR / "judge_labels_n130.csv")
    cli.add_argument("--output", type=Path, default=PUBLIC_DIR / "reliability_summary_n130.csv")
    return cli


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    human = read_public_labels(args.human_labels)
    judge = read_public_labels(args.judge_labels)
    summary = calculate_reliability(human, judge)
    write_summary(args.output, summary)
    for row in summary:
        kappa = "undefined" if row["cohens_kappa"] is None else f"{row['cohens_kappa']:.3f}"
        print(
            f"{row['variable']}: {row['raw_agreement_count']}/{row['n']} "
            f"({row['raw_agreement_percent']:.1f}%), kappa={kappa}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
