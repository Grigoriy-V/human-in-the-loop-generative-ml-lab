# Human-in-the-Loop Generative ML Lab

An evidence-led learning project that grew from a PyTorch DDPM into a latent diffusion/flow workflow with controlled agent assistance. It shows how model work, evaluation discipline, and human-supervised orchestration can reinforce one another.

The human defined the learning goals, approved scope and model decisions, and manually gated long training/evaluation. Agents performed bounded implementation, investigation, validation, and documentation tasks under an append-only audit trail.

![Portfolio progression from DDPM to AFHQ evaluation](docs/assets/portfolio_ml_progression.svg)

## What was built

Two connected pillars make up the case study.

1. **Generative ML:** class-conditioned DDPM/U-Net work progressed to SiT (a transformer-based latent flow model), a fixed-protocol evaluator, REPA (a frozen-teacher representation-alignment auxiliary loss), and a bounded AFHQ Cats model-selection decision.
2. **Human-supervised orchestration:** a supervisor defines scope and makes final decisions; workers execute narrowly bounded tasks; long GPU runs remain manual; separate append-only ledgers record ML operations and agent execution.

## Selected result: AFHQ Cats

For a fixed held-out AFHQ Cats **quick-200** comparison (200 seeds `1000–1199`, Heun-50, CFG 1.0), the raw early-stop REPA 20k checkpoint was selected on FID/KID. It used REPA through 10k steps, then continued to 20k without it.

| Raw 20k variant | FID ↓ | KID ↓ | Precision ↑ | Recall ↑ |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 48.051 | 0.02052 | **0.340** | **0.754** |
| Always-on REPA | 52.384 | 0.02531 | 0.310 | 0.722 |
| Early-stop REPA | **45.787** | **0.01692** | 0.280 | 0.732 |

![AFHQ Cats quick-200 metric comparison](docs/assets/portfolio_afhq_metrics.svg)

Early-stop improves FID by 4.71% and KID by 17.53% versus the baseline in this protocol, but the baseline remains stronger on precision and recall. This is a bounded selection decision—not a universal winner, statistical-significance claim, or full evaluation: **full-1000 was deliberately not run**. Read the [result report](reports/afhq_cat_sit_b_128_repa_early_stop_results.md) and [claim-to-evidence matrix](reports/portfolio_claim_evidence_matrix.md).

![Illustrative eight-seed AFHQ Cats grid: baseline, always-on REPA, and early-stop REPA; selection used quick-200 metrics](docs/assets/portfolio_afhq_fixed_seed_comparison.png)

The eight-seed grid is illustrative; model selection used the quick-200 metrics above, not this grid.

## Engineering highlights

- A CIFAR-10 DDPM baseline completed 200k steps. Under the fixed local benchmark, the five-run optimized median reached **2,272.92 images/s** versus the single historical **1,787.57** result (+27.15%); a repeated three-run A0 control measured **1,812.39 images/s** median (+25.41%). This is not a production-performance claim. [Evidence](reports/cifar10_optimization_report.md)
- The evaluator fixes seeds, sampler, CFG, VAE, reference split, and feature extractor; it reports FID, KID, precision, recall, and failure diagnostics. [Evaluator setup](reports/imagenette_sit_evaluator_setup.md)
- Checkpoint hashes, deterministic sampling checks, configuration snapshots, and decisions are preserved in compact reports and append-only ledgers. [Experiment ledger](reports/experiment_ledger.jsonl)

![Human-supervised agent pipeline](docs/assets/portfolio_agent_pipeline.svg)

## Read the story at the right depth

- [Portfolio case study](docs/portfolio_case_study.md) — the integrated 5–7 minute narrative.
- [Technical retrospective](docs/technical_retrospective.md) — design choices, failures, and lessons.
- [Agent orchestration reference](docs/agent_orchestration.md) — roles, controls, and audit lifecycle.
- [Reproducibility guide](docs/reproducibility.md) — supported evidence, a cheap inspection path, and explicit limits.
- [Operations guide](docs/operations_guide.md) — historical Windows-oriented commands; it is not a portability promise.
- [Third-party materials](THIRD_PARTY.md) — dataset/model attribution and use boundaries.

## Evidence map

| Question | Compact evidence |
| --- | --- |
| Why early-stop REPA was selected | [AFHQ result report](reports/afhq_cat_sit_b_128_repa_early_stop_results.md) → [evaluation config](mini_diffusion/configs/evaluation/afhq_cat_baseline_repa_early_stop_20k.yaml) → experiment event `afhq-cats-repa-early-stop-20k-freeze-closeout-20260718` |
| What claims are safe | [Claim-to-evidence matrix](reports/portfolio_claim_evidence_matrix.md) |
| How experiments evolved | [Project log](PROJECT_LOG.md) and [roadmap](ML_PROJECT_ROADMAP.md) |
| How agents were controlled | [Orchestration policy](docs/agent_orchestration.md) and [agent execution ledger](reports/agent_execution_ledger.jsonl) |

## Scope and availability

This repository tracks code, compact reports, configurations, hashes, and selected visuals. Datasets, checkpoints, caches, logs, and full evaluation outputs are intentionally ignored. The observed environment is Windows with Python 3.12 and an RTX 4090; dependencies are not fully pinned and no cross-platform or cloud claim is made.

Training and evaluation can be expensive and are human-gated. Start with the [reproducibility guide](docs/reproducibility.md), not a long-run command. Repository-owned code is available under the [MIT License](LICENSE); third-party datasets, weights, and checkpoints remain subject to their own terms in [THIRD_PARTY.md](THIRD_PARTY.md). The public repository is [Grigoriy-V/human-in-the-loop-generative-ml-lab](https://github.com/Grigoriy-V/human-in-the-loop-generative-ml-lab).

For a no-download, no-GPU check of public packaging and evidence integrity, run `python tools/verify_public_repo.py` with Python 3.11+; it does not reproduce ML metrics. Details and limits are in the [reproducibility guide](docs/reproducibility.md).
