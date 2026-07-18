# Public Repository Readiness Audit

**Audit type:** static, read-only-first portfolio audit  
**Snapshot commit:** `6ad4a9d` (`docs: add portfolio claim evidence matrix`)  
**Scope:** tracked repository content and Git metadata only. No training, evaluation, tests, sampling, benchmark, dependency install, or network access was run.  
**Audience:** a Head of AI reviewing a public GitHub repository.

## Decision

**No-go for public portfolio release today.** The implementation and evidence base are credible, and no tracked checkpoint, dataset, or obvious credential value was found. However, two presentation/release blockers must be resolved first: the repository has no license, and the top-level README is an outdated, Windows-specific operator manual rather than an executive entry point for the current ML plus agent-orchestration case.

After the P1 items below, the repository can be a strong public case study. This is a packaging decision, not a request to run more ML work.

## Findings

| Severity | Finding and evidence | Why it matters publicly | Recommended remediation |
| --- | --- | --- | --- |
| P1 | **No repository license.** No tracked `LICENSE*`, `COPYING*`, or `NOTICE*` file; AFHQ's CC BY-NC 4.0 is mentioned only in [README.md](../README.md), and other third-party inputs are cited ad hoc. | A reviewer cannot tell what they may reuse, and code/data/model licensing boundaries are unclear. | Add a deliberate code license. Add a short third-party attribution and use-policy section covering CIFAR-10, Tiny ImageNet, Imagenette, AFHQ, SD VAE, and DINOv2. State explicitly that datasets and checkpoints are not distributed. Confirm the intended public/commercial-use posture before choosing the code license. |
| P1 | **README is mispositioned and stale for the active story.** [README.md](../README.md) is titled “Mini Diffusion” and has no portfolio, case-study, AFHQ result, orchestration, human-gate, or agent wording. It contains 38 Windows-path references and leads with a local `D:\ML\My_first_model` setup. The current project direction and freeze are in [ML_PROJECT_ROADMAP.md](../ML_PROJECT_ROADMAP.md) and [reports/afhq_cat_sit_b_128_repa_early_stop_results.md](afhq_cat_sit_b_128_repa_early_stop_results.md). | The first 60 seconds communicate an early educational DDPM, not the evidence-backed `Human-in-the-Loop Generative ML Lab` that the portfolio stage intends to show. It also risks readers reproducing obsolete or costly commands. | Replace the top-level README with an executive overview: outcome, exact AFHQ decision and limitation, architecture/pipeline diagram, safe verification path, links to evidence and technical retrospective, constraints, and reproducibility/data/model disclosure. Move the long command reference to a dedicated operational guide. |
| P1 | **Reproducibility is not portable or fully pinned.** [requirements.txt](../requirements.txt) has unpinned packages and no lockfile or environment specification. [README.md](../README.md) separately installs a CUDA-indexed PyTorch wheel and asks for Python 3.12 on Windows, while many commands assume PowerShell and a local `.venv`. No CI workflow, container, or cross-platform note is tracked. | A reviewer cannot reliably distinguish a small safe validation from GPU training, or reproduce the environment outside the author’s Windows/RTX 4090 setup. | Publish a supported-environment matrix (OS, Python, Torch/CUDA, GPU expectation), a pinned/locked CPU-safe test environment, and a separate optional GPU install path. Add one clearly labelled no-download, no-GPU verification command and expected result. Mark all training/evaluation commands as expensive and human-gated. |
| P1 | **The repository's only large visual is 14.3 MiB and does not show the current winner.** [docs/assets/imagenette_sit_baseline100_vs_repa350_ema_heun50.png](../docs/assets/imagenette_sit_baseline100_vs_repa350_ema_heun50.png) is 14,975,675 bytes; the only other tracked visual is an older Tiny ImageNet grid. The current AFHQ winner’s result report has no tracked representative visual. | The main portfolio claim lacks an immediately visible result; a 14 MiB PNG makes GitHub browsing slower and distracts from the selected experiment. | Create a small, labelled AFHQ comparison chart and representative paired grid under `docs/assets/`, with protocol/limitations in the caption. Optimize or replace the Imagenette PNG if it remains useful. Do not add full evaluation outputs or checkpoints. |
| P2 | **Evidence is strong but scattered.** The raw history is distributed among [PROJECT_LOG.md](../PROJECT_LOG.md), `reports/`, `evaluation/`, two append-only ledgers, and the new claim matrix. Relative Markdown links currently resolve, but the README does not navigate to this evidence. | A senior reviewer will not infer the experiment lineage unaided, and metrics can be read without their protocol/limitation. | Add a short “evidence map” in the README: current result → report → configuration → ledger event → visual. Keep the raw ledgers as audit material, not primary reading. |
| P2 | **Current claim boundaries need to be carried into the public narrative.** The AFHQ result is a quick-200 comparison: early-stop raw wins FID/KID, while baseline remains better on precision/recall; full-1000 was intentionally skipped. See [reports/afhq_cat_sit_b_128_repa_early_stop_results.md](afhq_cat_sit_b_128_repa_early_stop_results.md) and [reports/portfolio_claim_evidence_matrix.md](portfolio_claim_evidence_matrix.md). | “Best model” without this scope would overclaim. The same concern applies to the cross-step Imagenette comparison. | Describe the winner as the selected result **for this bounded quick-200 AFHQ decision**, state the precision/recall limitation, and say full-1000 was not run. Preserve the matrix’s forbidden-claim language in the technical retrospective. |
| P2 | **Generated-output policy is good but release guidance is incomplete.** [.gitignore](../.gitignore) excludes `.venv/`, `datasets/`, `outputs/`, `evaluation/`, checkpoints, and common local caches; `git check-ignore` confirms representative paths are ignored. A few evaluation manifests/reports are intentionally tracked. | The policy protects the remote now, but contributors need a concise explanation of which small artifacts are allowed and why. | Document “tracked evidence vs. ignored generated artifacts” in the README/contributing guidance. Keep only compact reports, manifests, metrics, and curated visuals in Git. |
| P2 | **Static secret scan found sensitive-name terms, not an apparent credential value.** Tracked-file scanning produced only semantic terms (for example model tokens, agent-policy wording, and DINO feature-cache code); no `.env`/credential file is tracked. | This is a positive result, but a public release still needs an intentional pre-push secret scan. | Keep secret patterns in `.gitignore`, add a documented pre-release secret scan, and review GitHub secret-scanning alerts after the repository is public. Do not commit access tokens or downloaded model credentials. |
| P2 | **Git is locally in progress rather than release-clean.** At audit time, the working tree contained packaging-stage report/ledger changes from other approved workers; `git diff --check` was clean. `main` is ahead of `origin/main`. | This is normal during packaging, but the release snapshot should be one intentional, reviewable commit series. | Before publishing, require `git status --short` to be empty, review the final diff, and push only after the P1 release checklist passes. |

## What already meets the bar

- Tracked content is compact: about 15.9 MiB total; no tracked dataset, checkpoint, cache, archive, or full evaluation output was detected.
- The repository records commands, hashes, metrics, and decisions in structured reports and ledgers. This is unusually good evidence discipline for an educational project.
- All checked tracked Markdown relative links resolve.
- `.gitignore` correctly excludes generated ML artifacts, including `datasets/`, `outputs/`, `evaluation/`, `.venv/`, and checkpoint extensions.
- A small safe path exists conceptually in [README.md](../README.md): the CIFAR debug configuration uses fake data and the test command targets `mini_diffusion/tests`. It still needs the portable, pinned setup and expected-output framing above.

## Release gate

Proceed to the final public-readiness review only when all of the following are true:

1. A deliberate code license and third-party data/model attribution policy are present.
2. The root README tells the current ML plus orchestration story in under three minutes, links to evidence, and labels limitations.
3. A portable, pinned, inexpensive verification path is documented separately from GPU training/evaluation.
4. Current AFHQ visuals and a compact metric comparison are tracked under `docs/assets/` and embedded from the README/case study.
5. Claims match [reports/portfolio_claim_evidence_matrix.md](portfolio_claim_evidence_matrix.md), including the quick-200 and full-1000 limitations.
6. The release commit is clean and receives one final secret/link/render review.

## Audit limitations

This audit did not execute code or verify external URLs, licenses, model-download access, GPU compatibility, or actual secret values. The secret result is a static tracked-file check only; it is not a substitute for a hosted secret scanner or a legal review.
