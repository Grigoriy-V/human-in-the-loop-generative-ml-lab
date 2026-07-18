# Reproducibility and evidence guide

## What this repository supports today

The observed development environment was **Windows**, **Python 3.12**, and an **NVIDIA RTX 4090**. Some reports additionally record PyTorch/CUDA versions for their specific run. This is an observed local environment, not a portable support matrix: dependencies are not fully pinned, there is no lockfile or CI workflow, and GPU/model-download availability varies.

The repository tracks source code, configurations, compact reports, selected curated visuals, and append-only evidence ledgers. It intentionally ignores datasets, checkpoints, latent/feature caches, training logs, and full evaluation output. Where a local artifact mattered to a decision, its report records a repository-relative location and/or SHA-256.

## Portable evidence verification

From a fresh clone, run this from the repository root with Python 3.11 or newer:

```powershell
python tools/verify_public_repo.py
```

It uses only the Python standard library and no network, GPU, model imports, datasets, checkpoints, or downloaded dependencies. On success it prints one concise `public evidence verification passed` line; on failure it lists actionable JSONL, public-link, required-evidence, visual-size, or tracked-artifact issues and exits nonzero. This path was run during portfolio packaging.

It validates packaging and evidence integrity only. It does **not** reproduce numerical ML results, train, sample, benchmark, or evaluate a model. For historical Windows-oriented operational commands, see the [operations guide](operations_guide.md).

## Training and evaluation boundaries

Training, sampling, cache preparation, benchmark, and evaluation commands may require a CUDA-capable environment, local datasets/checkpoints, downloaded model weights, and substantial time. They are expensive and human-gated by project policy; this guide does not authorize or imply running them. Historical commands are preserved in the [operations guide](operations_guide.md) and the relevant reports, with no claim that they were re-tested during packaging.

The current AFHQ Cats decision is based on a quick-200 protocol, not a full-1000 confirmation. The frozen checkpoint, dataset, and full evaluator artifacts are local/ignored; the tracked [result report](../reports/afhq_cat_sit_b_128_repa_early_stop_results.md) is the public evidence entry point.

## Public packaging status

The repository now includes an [MIT license](../LICENSE) and documented [third-party attribution and use boundaries](../THIRD_PARTY.md). Final public-evidence verification and review passed; the earlier [readiness audit](../reports/public_repo_readiness_audit.md) remains the pre-remediation snapshot. This closes packaging readiness, not full ML reproducibility: numerical results still depend on ignored datasets, checkpoints, caches, the recorded local environment, and human-gated expensive runs.
