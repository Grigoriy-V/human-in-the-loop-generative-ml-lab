"""Build compact portfolio visuals from existing reports and evaluator artifacts.

This script is deliberately presentation-only: it reads stored JSON/PNG outputs,
does not import model code, and never invokes training, sampling, or evaluation.
"""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
EVAL = ROOT / "evaluation" / "afhq_cat_baseline_repa_early_stop_20k"
FONT = r"C:\Windows\Fonts\arial.ttf"
BOLD = r"C:\Windows\Fonts\arialbd.ttf"
NAVY, INK, MUTED, LINE = "#102A43", "#243B53", "#627D98", "#D9E2EC"
BLUE, TEAL, ORANGE, RED, GREEN = "#247BA0", "#2A9D8F", "#F4A261", "#E76F51", "#2F855A"


def svg(path: Path, width: int, height: int, body: str) -> None:
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
        '<style>text{font-family:Arial,Helvetica,sans-serif}.title{font-size:28px;font-weight:700;fill:#102A43}.sub{font-size:16px;fill:#627D98}.label{font-size:16px;fill:#243B53}.small{font-size:13px;fill:#627D98}.bold{font-weight:700}</style>'
        f'<rect width="100%" height="100%" fill="#FFFFFF"/>{body}</svg>', encoding="utf-8"
    )


def progression() -> None:
    stages = [
        ("CIFAR-10", "DDPM / U-Net", "32×32 · completed", BLUE),
        ("Tiny ImageNet", "U-Net", "64×64 · partial 150k / 400k", ORANGE),
        ("Imagenette", "Latent SiT + REPA", "128×128 · cross-step", TEAL),
        ("AFHQ Cats", "SiT-B/2 + REPA", "128×128 · quick-200", TEAL),
        ("Frozen result", "Early-stop REPA", "raw 20k · FID/KID winner", GREEN),
    ]
    xs = [85, 350, 615, 880, 1145]
    body = '<text x="70" y="55" class="title">ML progression: from pixel diffusion to controlled model selection</text>'
    body += '<text x="70" y="85" class="sub">Learning milestones are intentionally labelled where evidence is partial, cross-step, or quick-protocol.</text>'
    body += '<line x1="130" y1="210" x2="1270" y2="210" stroke="#BCCCDC" stroke-width="6" stroke-linecap="round"/>'
    for i, ((name, method, note, color), x) in enumerate(zip(stages, xs)):
        if i < len(stages) - 1:
            body += f'<path d="M{x+42} 210 H{xs[i+1]-42}" stroke="#9FB3C8" stroke-width="3" marker-end="url(#arrow)"/>'
        body += f'<circle cx="{x}" cy="210" r="31" fill="{color}"/><text x="{x}" y="216" text-anchor="middle" fill="white" font-size="18" font-weight="700">{i+1}</text>'
        body += f'<rect x="{x-105}" y="260" width="210" height="126" rx="12" fill="#F0F4F8" stroke="#D9E2EC"/>'
        body += f'<text x="{x}" y="294" text-anchor="middle" class="label bold">{html.escape(name)}</text><text x="{x}" y="321" text-anchor="middle" class="label">{html.escape(method)}</text><text x="{x}" y="350" text-anchor="middle" class="small">{html.escape(note)}</text>'
    body += '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#9FB3C8"/></marker></defs>'
    body += '<rect x="70" y="430" width="1260" height="64" rx="10" fill="#FFF7ED"/><text x="95" y="458" class="label bold">Reading rule:</text><text x="95" y="480" class="small">Do not treat Tiny ImageNet as completed, Imagenette cross-step as a same-step ablation, or AFHQ quick-200 as full evaluation.</text>'
    svg(ASSETS / "portfolio_ml_progression.svg", 1400, 540, body)


def metrics() -> None:
    with (EVAL / "metrics.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    names = {"baseline_raw_20k": "Baseline", "repa_raw_20k": "Always-on REPA", "early_stop_raw_20k": "Early-stop REPA"}
    colors = {"baseline_raw_20k": BLUE, "repa_raw_20k": ORANGE, "early_stop_raw_20k": GREEN}
    metric_specs = [("fid", "FID ↓", "lower is better"), ("kid", "KID ↓", "lower is better"), ("precision", "Precision ↑", "higher is better"), ("recall", "Recall ↑", "higher is better")]
    body = '<text x="65" y="52" class="title">AFHQ Cats: raw 20k comparison</text><text x="65" y="80" class="sub">Fixed quick-200 protocol · seeds 1000–1199 · Heun-50 · CFG 1.0 · lower/higher direction shown per metric</text>'
    for j, (key, title, direction) in enumerate(metric_specs):
        x, y, w, h = 65 + j * 330, 125, 290, 305
        vals = [float(r[key]) for r in rows]
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * .22 or .05
        lo, hi = lo - pad, hi + pad
        body += f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="#F8FAFC" stroke="#D9E2EC"/>'
        body += f'<text x="{x+20}" y="{y+34}" class="label bold">{title}</text><text x="{x+20}" y="{y+57}" class="small">{direction}</text>'
        for i, r in enumerate(rows):
            val = float(r[key]); bar = 150 * (val-lo)/(hi-lo)
            yy = y + 98 + i * 65
            body += f'<text x="{x+20}" y="{yy+13}" class="small">{names[r["variant"]]}</text><rect x="{x+20}" y="{yy+22}" width="{bar:.1f}" height="18" rx="5" fill="{colors[r["variant"]]}"/><text x="{x+185}" y="{yy+37}" class="label bold">{val:.5g}</text>'
        winner = min(rows, key=lambda r: float(r[key])) if key in ("fid", "kid") else max(rows, key=lambda r: float(r[key]))
        body += f'<text x="{x+20}" y="{y+285}" class="small">Best: {names[winner["variant"]]}</text>'
    body += '<rect x="65" y="463" width="1270" height="55" rx="10" fill="#EEF7F3"/><text x="87" y="496" class="label">Decision: early-stop REPA was frozen for its FID/KID lead; baseline retains the precision/recall lead. Full-1000 was not run.</text>'
    svg(ASSETS / "portfolio_afhq_metrics.svg", 1400, 555, body)


def comparison() -> None:
    variants = [("baseline_raw_20k", "Baseline"), ("repa_raw_20k", "Always-on REPA"), ("early_stop_raw_20k", "Early-stop REPA")]
    cell, cols, sample_count = 128, 8, 8
    canvas = Image.new("RGB", (cols * cell + 260, 3 * cell + 150), "white")
    draw = ImageDraw.Draw(canvas)
    title, small, label = ImageFont.truetype(BOLD, 28), ImageFont.truetype(FONT, 16), ImageFont.truetype(BOLD, 18)
    draw.text((26, 18), "AFHQ Cats — fixed-seed comparison", font=title, fill="#102A43")
    draw.text((26, 57), "Canonical evaluator outputs · earliest fixed seeds 1000–1007 · raw 20k · Heun-50 · CFG 1.0", font=small, fill="#627D98")
    for r, (variant, name) in enumerate(variants):
        src = Image.open(EVAL / "variants" / variant / "grid.png").convert("RGB")
        # The evaluator grid is ordered row-major with ten columns.  The top row is seeds 1000–1009.
        for c in range(sample_count):
            crop = src.crop((c * 130, 0, c * 130 + 128, 128)).resize((cell, cell), Image.Resampling.LANCZOS)
            canvas.paste(crop, (250 + c * cell, 105 + r * cell))
        src.close()
        draw.text((22, 151 + r * cell), name, font=label, fill="#243B53")
    for c in range(sample_count):
        draw.text((250 + c * cell + 26, 86), str(1000 + c), font=small, fill="#627D98")
    draw.text((26, 504), "Selection rule: fixed evaluator ordering, no hand-picked samples. These visuals are illustrative; model selection used metrics above.", font=small, fill="#627D98")
    canvas.save(ASSETS / "portfolio_afhq_fixed_seed_comparison.png", optimize=True)


def pipeline() -> None:
    boxes = [
        (65, 155, 190, 96, "Human / user", "approves direction\nand long runs", BLUE),
        (315, 155, 220, 96, "Root supervisor", "roadmap · task spec\nreview · decision", NAVY),
        (595, 155, 245, 96, "Role routing", "Luna: clerical · Terra: routine\nSol: explicit high-risk approval", TEAL),
        (900, 155, 220, 96, "Bounded work", "implementation / validation\nmanual long-run gate", ORANGE),
        (1180, 155, 160, 96, "Evidence", "tests / evaluator\nartifacts", GREEN),
    ]
    body = '<text x="65" y="52" class="title">Human-supervised agent workflow</text><text x="65" y="80" class="sub">Agents execute bounded tasks; the human retains authority over direction and long training/evaluation.</text>'
    body += '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#627D98"/></marker></defs>'
    for i, (x, y, w, h, head, sub, color) in enumerate(boxes):
        body += f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="#FFFFFF" stroke="{color}" stroke-width="3"/><text x="{x+w/2}" y="{y+35}" text-anchor="middle" class="label bold">{head}</text>'
        for n, line in enumerate(sub.split("\n")):
            body += f'<text x="{x+w/2}" y="{y+61+n*19}" text-anchor="middle" class="small">{line}</text>'
        if i < len(boxes)-1: body += f'<path d="M{x+w+8} {y+48} H{boxes[i+1][0]-10}" stroke="#627D98" stroke-width="3" marker-end="url(#arrow)"/>'
    body += '<path d="M1260 260 V325 H425 V260" fill="none" stroke="#627D98" stroke-width="3" marker-end="url(#arrow)"/>'
    body += '<rect x="315" y="335" width="440" height="92" rx="12" fill="#F0F4F8" stroke="#9FB3C8"/><text x="535" y="370" text-anchor="middle" class="label bold">Append-only audit trail</text><text x="535" y="397" text-anchor="middle" class="small">experiment ledger: material ML operations · agent ledger: lifecycle + review</text>'
    body += '<rect x="810" y="335" width="530" height="92" rx="12" fill="#FFF7ED" stroke="#F4A261"/><text x="1075" y="370" text-anchor="middle" class="label bold">Manual gate is a control, not a footnote</text><text x="1075" y="397" text-anchor="middle" class="small">Long training/evaluation is prepared by agents and launched only by the human.</text>'
    svg(ASSETS / "portfolio_agent_pipeline.svg", 1400, 475, body)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    progression(); metrics(); comparison(); pipeline()
    for path in ASSETS.glob("portfolio_*"):
        print(f"{path.relative_to(ROOT)}\t{path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
