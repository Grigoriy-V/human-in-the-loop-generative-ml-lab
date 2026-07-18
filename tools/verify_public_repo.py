#!/usr/bin/env python3
"""Cheap, dependency-free verification for the repository's public evidence."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
LEADERS = (
    ROOT / "reports" / "agent_execution_ledger.jsonl",
    ROOT / "reports" / "experiment_ledger.jsonl",
)
PUBLIC_DOCS = (
    ROOT / "README.md",
    ROOT / "docs" / "portfolio_case_study.md",
    ROOT / "docs" / "technical_retrospective.md",
    ROOT / "docs" / "reproducibility.md",
    ROOT / "docs" / "operations_guide.md",
    ROOT / "docs" / "agent_orchestration.md",
    ROOT / "ML_PROJECT_ROADMAP.md",
)
REQUIRED_EVIDENCE = (
    ROOT / "LICENSE",
    ROOT / "THIRD_PARTY.md",
    ROOT / "PROJECT_LOG.md",
    ROOT / "reports" / "afhq_cat_sit_b_128_repa_early_stop_results.md",
    ROOT / "reports" / "portfolio_claim_evidence_matrix.md",
    ROOT / "reports" / "public_repo_readiness_audit.md",
    ROOT / "reports" / "agent_execution_ledger.schema.json",
    ROOT / "reports" / "experiment_ledger.schema.json",
    ROOT / "docs" / "assets" / "portfolio_ml_progression.svg",
    ROOT / "docs" / "assets" / "portfolio_afhq_metrics.svg",
    ROOT / "docs" / "assets" / "portfolio_afhq_fixed_seed_comparison.png",
    ROOT / "docs" / "assets" / "portfolio_agent_pipeline.svg",
)
PORTFOLIO_VISUALS = tuple(path for path in REQUIRED_EVIDENCE if "portfolio_" in path.name)
VISUAL_SIZE_CAP_BYTES = 2 * 1024 * 1024
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
CODE_SPAN = re.compile(r"`[^`]*`")
EXTERNAL_SCHEMES = ("http:", "https:", "mailto:", "tel:", "data:")
FORBIDDEN_SUFFIXES = {".ckpt", ".pt", ".pth", ".safetensors", ".onnx", ".npy", ".npz"}


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def check_ledgers(errors: list[str]) -> None:
    for ledger in LEADERS:
        seen: set[str] = set()
        try:
            lines = ledger.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            errors.append(f"missing/unreadable ledger {relative(ledger)}: {exc}")
            continue
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"invalid JSONL {relative(ledger)}:{line_number}: {exc.msg}")
                continue
            event_id = event.get("event_id") if isinstance(event, dict) else None
            if not isinstance(event_id, str) or not event_id.strip():
                errors.append(f"missing event_id in {relative(ledger)}:{line_number}")
            elif event_id in seen:
                errors.append(f"duplicate event_id {event_id!r} in {relative(ledger)}:{line_number}")
            else:
                seen.add(event_id)


def markdown_target(raw_target: str) -> str:
    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")]
    else:
        target = target.split(maxsplit=1)[0]
    return unquote(target)


def check_links(errors: list[str]) -> None:
    for document in PUBLIC_DOCS:
        try:
            text = document.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"missing/unreadable public document {relative(document)}: {exc}")
            continue
        text = CODE_SPAN.sub("", text)
        for match in MARKDOWN_LINK.finditer(text):
            target = markdown_target(match.group(1))
            if not target or target.startswith("#") or target.lower().startswith(EXTERNAL_SCHEMES):
                continue
            path_part = target.split("#", 1)[0].split("?", 1)[0]
            if not path_part or path_part.startswith("/"):
                continue
            candidate = (document.parent / PurePosixPath(path_part)).resolve()
            try:
                candidate.relative_to(ROOT)
            except ValueError:
                errors.append(f"out-of-repository link in {relative(document)}: {target}")
                continue
            if not candidate.is_file():
                errors.append(f"broken tracked-file link in {relative(document)}: {target}")


def check_required_evidence(errors: list[str]) -> None:
    for path in (*PUBLIC_DOCS, *REQUIRED_EVIDENCE):
        if not path.is_file():
            errors.append(f"required public evidence missing: {relative(path)}")
    for visual in PORTFOLIO_VISUALS:
        if visual.is_file() and visual.stat().st_size > VISUAL_SIZE_CAP_BYTES:
            errors.append(
                f"portfolio visual exceeds {VISUAL_SIZE_CAP_BYTES // (1024 * 1024)} MiB cap: "
                f"{relative(visual)} ({visual.stat().st_size} bytes)"
            )


def check_repository_identity(errors: list[str]) -> None:
    stale_identifier = "Grigoriy-V/ML_U-Net_test"
    for document in PUBLIC_DOCS:
        try:
            text = document.read_text(encoding="utf-8")
        except OSError:
            continue
        if stale_identifier in text:
            errors.append(
                f"stale GitHub repository identifier in {relative(document)}: "
                f"{stale_identifier}"
            )


def tracked_files() -> list[str] | None:
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return [item for item in result.stdout.decode("utf-8", errors="replace").split("\0") if item]


def check_forbidden_artifacts(errors: list[str]) -> str:
    tracked = tracked_files()
    if tracked is None:
        return "git unavailable; skipped tracked-artifact check"
    for name in tracked:
        path = PurePosixPath(name)
        parts = {part.lower() for part in path.parts}
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden tracked model/cache binary: {name}")
        elif "outputs" in parts or "datasets" in parts or "checkpoints" in parts:
            errors.append(f"forbidden tracked generated-artifact path: {name}")
        elif "evaluation" in parts and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden tracked evaluation binary: {name}")
    return f"checked {len(tracked)} tracked files"


def main() -> int:
    errors: list[str] = []
    check_ledgers(errors)
    check_links(errors)
    check_required_evidence(errors)
    check_repository_identity(errors)
    git_summary = check_forbidden_artifacts(errors)
    if errors:
        print(f"public evidence verification FAILED ({len(errors)} issue(s))")
        for error in errors:
            print(f"- {error}")
        return 1
    print(
        "public evidence verification passed: "
        f"{len(LEADERS)} ledgers, {len(PUBLIC_DOCS)} public docs, "
        f"{len(REQUIRED_EVIDENCE)} required evidence files; {git_summary}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
