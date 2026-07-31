#!/usr/bin/env python3
"""Validate canonical documentation, registry sources, and generated results."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_DOCS = [
    "README.md",
    "AGENTS.md",
    "PROJECT_SPEC.md",
    "RESULTS.md",
    "EXPERIMENT_LOG.md",
    "experiments/registry.yaml",
    "experiments/schema.yaml",
    "docs/archive/README.md",
    "docs/operations/README.md",
    "docs/operations/current_status.md",
]

FORBIDDEN_CURRENT_DOCS = [
    "WORKLOG.md",
    "BASELINE_RUN_SUMMARY.md",
    "conpareAGENTS.md",
    "comparePROJECT_SPEC.md",
    "paper/PAPER_CONTEXT.md",
    ".ipynb_checkpoints/PROJECT_SPEC-checkpoint.md",
    ".ipynb_checkpoints/WORKLOG-checkpoint.md",
    ".ipynb_checkpoints/AGENTS-checkpoint.md",
]

SCALING_SOURCE_TYPES = {"scaling_pipeline", "scaling_average_rows", "scaling_json"}


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)


def validate_seed_template(
    entry: dict, source: dict, label: str, errors: list[str]
) -> None:
    paths = [
        ROOT / source["path_template"].format(seed=seed)
        for seed in source.get("expected_seeds", [])
    ]
    existing = [path.is_file() for path in paths]
    if not existing or not any(existing):
        fail(f"{entry['id']} has no existing {label} source", errors)
    if entry.get("status") == "complete" and (not existing or not all(existing)):
        fail(f"complete entry {entry['id']} has missing {label} source files", errors)


def main() -> None:
    errors: list[str] = []
    for relative in REQUIRED_DOCS:
        if not (ROOT / relative).is_file():
            fail(f"missing required document: {relative}", errors)
    for relative in FORBIDDEN_CURRENT_DOCS:
        if (ROOT / relative).exists():
            fail(f"stale current-document duplicate remains: {relative}", errors)

    registry: dict = {"experiments": []}
    registry_path = ROOT / "experiments" / "registry.yaml"
    if registry_path.is_file():
        registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        entries = registry.get("experiments", [])
        ids = [entry.get("id") for entry in entries]
        if len(ids) != len(set(ids)):
            fail("duplicate experiment ids in registry", errors)
        for entry in entries:
            source = entry.get("source", {})
            source_type = source.get("type")
            if source_type == "pipeline":
                existing = []
                for seed in source.get("expected_seeds", []):
                    path = ROOT / source["path_template"].format(seed=seed)
                    existing.append(path.is_file())
                if not any(existing):
                    fail(f"{entry['id']} has no existing pipeline source", errors)
                if entry.get("status") == "complete" and not entry.get("allow_partial") and not all(existing):
                    fail(f"complete entry {entry['id']} has missing source files", errors)
            elif source_type == "average_row":
                if not (ROOT / source["path"]).is_file():
                    fail(f"missing average-row source for {entry['id']}: {source['path']}", errors)
            elif source_type == "evidence":
                for relative in source.get("paths", []):
                    if not (ROOT / relative).is_file():
                        fail(f"missing evidence source for {entry['id']}: {relative}", errors)
            elif source_type in SCALING_SOURCE_TYPES:
                validate_seed_template(entry, source, "metric", errors)
                communication = entry.get("communication", {}).get("source")
                if communication:
                    communication = {
                        **communication,
                        "expected_seeds": source.get("expected_seeds", []),
                    }
                    validate_seed_template(entry, communication, "communication", errors)
            else:
                fail(f"unknown source type for {entry.get('id')}: {source_type}", errors)

    canonical_csv = ROOT / "experiments" / "generated" / "canonical_results.csv"
    if canonical_csv.is_file():
        with canonical_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            fail("canonical_results.csv is empty", errors)
        for row in rows:
            if row["actual_status"] == "complete" and row["missing_seeds"] not in {"[]", ""}:
                fail(f"complete result has missing seeds: {row['id']}", errors)
    else:
        fail("canonical_results.csv has not been generated", errors)

    scaling_csv = ROOT / "experiments" / "generated" / "iclr2027_scaling_results.csv"
    if scaling_csv.is_file():
        with scaling_csv.open("r", encoding="utf-8", newline="") as handle:
            scaling_rows = list(csv.DictReader(handle))
        if not scaling_rows:
            fail("iclr2027_scaling_results.csv is empty", errors)
    elif any(
        entry.get("table_section") == "scaling"
        for entry in registry.get("experiments", [])
    ):
        fail("iclr2027_scaling_results.csv has not been generated", errors)

    agents_text = (ROOT / "AGENTS.md").read_text(encoding="utf-8") if (ROOT / "AGENTS.md").is_file() else ""
    if "当前实验结果快照" in agents_text:
        fail("AGENTS.md still contains a mutable result snapshot section", errors)
    if "统一 ConvNet packet encoder 是当前主方法" in agents_text:
        fail("AGENTS.md contains a stale main-method statement", errors)

    if errors:
        print("[docs] validation failed:")
        for error in errors:
            print(f"- {error}")
        sys.exit(1)
    print(f"[docs] validation passed: {len(REQUIRED_DOCS)} canonical docs")


if __name__ == "__main__":
    main()
