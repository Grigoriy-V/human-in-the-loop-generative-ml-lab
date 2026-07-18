# Portfolio Content Gap Audit

**Date:** 2026-07-18  
**Scope:** static, evidence-backed audit of the public portfolio story. No ML operation, code inspection beyond reported evidence, or public-narrative edit was performed.

## Executive verdict

The public package is sufficient for a portfolio conversation now: it explains the DDPM-to-latent-SiT progression, gives a bounded AFHQ selection result, shows methodological caution, and makes the human gate legible. Its main omission is upstream of the metrics: a technical reader cannot yet see that the AFHQ comparison rests on a deliberately controlled image-preparation and cache-provenance pipeline. Add that once, concisely, in the case study; use the retrospective for the supporting technical detail. Do not add more metric tables or hyperparameter inventory to the README.

The existing visual evidence is sufficient for the stated story: the README/case show the AFHQ metric chart and explicitly illustrative fixed-seed grid, and the README links the Imagenette side-by-side grid. It should not be represented as a comprehensive qualitative study.

## Ranked additions

### P1 — must add

1. **Controlled AFHQ data preparation and held-out isolation**

   - **Target:** `docs/portfolio_case_study.md`, immediately after the AFHQ outcome paragraph in **ML outcomes**.
   - **Proposed content (2 sentences):** “The AFHQ comparison used 5,153 training cats and a separate 500-image held-out split. Each training source produced four deterministic crop/flip variants, while evaluation used one deterministic center-square crop with no training augmentation; manifests and hashes linked each latent to its source and prevented the held-out cache from entering training.”
   - **Evidence:** `reports/afhq_cat_sit_b_128_setup.md` (Runtime Completion; Design; Dataset Status), `reports/afhq_cat_sit_b_128_cache_stats.md`, and `reports/experiment_ledger.jsonl` event `afhq-cats-data-cache-benchmark-smoke-20260718`.
   - **Hiring signal:** demonstrates data-split discipline, reproducibility, and awareness that generative-model evidence begins with data provenance rather than only a loss or a grid.
   - **Overclaim/overload risk:** do not call the four variants independent examples, claim a generalization guarantee, or include hashes/count tables in the case study.

### P2 — useful if keeping the retrospective technical

2. **Teacher/student view alignment for REPA**

   - **Target:** `docs/technical_retrospective.md`, in the REPA section after its first explanatory paragraph.
   - **Proposed content (3 sentences):** “For AFHQ REPA, the frozen DINOv2 teacher was fed the exact cached crop/flip used to produce each VAE latent, then resized to 224 with the teacher normalization. Feature-cache preparation revalidated path, augmentation seed/index, split, and source hash before writing features. This prevents a representation-alignment loss from comparing a latent from one view with teacher features from another.”
   - **Evidence:** `reports/afhq_cat_sit_b_128_repa_readiness.md` (Teacher Feature Cache; REPA) and `reports/afhq_cat_sit_b_128_setup.md` (Design).
   - **Hiring signal:** makes a subtle but high-value correctness decision visible: teacher/student alignment and cache provenance were treated as experimental controls.
   - **Overclaim/overload risk:** say “prevents mixing” rather than “proves no leakage”; do not introduce DINO tensor shapes, projector dimensions, or feature-cache byte size here.

3. **One compact objective-transition sentence**

   - **Target:** `docs/technical_retrospective.md`, end of **Scale cautiously: Tiny ImageNet and latent SiT**.
   - **Proposed content (2 sentences):** “The modelling objective changed with the architecture: the earlier DDPM trained a noise predictor, whereas SiT trained a velocity field and sampled it by numerical integration. REPA was an auxiliary representation-alignment term on top of that unchanged flow loss, not a replacement sampling method.”
   - **Evidence:** `reports/cifar10_baseline.md`, `reports/imagenette_sit_s_128_repa_setup.md` (Scope and implementation), and `docs/technical_retrospective.md` existing SiT/REPA descriptions.
   - **Hiring signal:** helps an interviewer see conceptual command of the DDPM → flow/SiT → REPA progression without requiring a paper-style derivation.
   - **Overclaim/overload risk:** avoid equations, scheduler settings, and claims that one objective is intrinsically superior.

4. **Observed operating envelope, only as a decision example**

   - **Target:** `docs/technical_retrospective.md`, AFHQ section, one sentence after data/cache discussion.
   - **Proposed content (1 sentence):** “On the recorded RTX 4090 cache benchmark, batch 128 with accumulation two was selected over physical batch 256 because it was faster (2,152.02 vs 2,057.33 images/s) and used less peak VRAM (5.27 vs 8.78 GB).”
   - **Evidence:** `reports/afhq_cat_sit_b_128_setup.md` (Runtime Completion; Performance Check) and `reports/experiment_ledger.jsonl` event `afhq-cats-data-cache-benchmark-smoke-20260718`.
   - **Hiring signal:** shows a concrete throughput/memory choice, rather than generic familiarity with batch size and accumulation.
   - **Overclaim/overload risk:** retain the local-hardware boundary; omit CUDA/PyTorch versions and all optimiser flags.

### P3 — keep out of the public narrative (or leave in reports)

- Projector dimensions/parameter counts, DINO feature tensor sizes, cache byte totals, exact cache fingerprints, checkpoint SHA-256 values, and full optimiser/scheduler settings are excellent evidence but belong in reports/configs.
- The full AFHQ VAE-ceiling numbers, 10k full-1000 evaluation, nearest-neighbour counts, and repeated quick-run duration are useful audit detail but would distract from the final early-stop selection story.
- CIFAR and Tiny ImageNet implementation minutiae should remain selective: the current sampler failure/resume lesson and honest partial stop already communicate the relevant judgment.
- Do not elevate the roughly 5x token observation. Its current placement as a user-reported, non-instrumented operational observation is correct; the next credible public statement requires matched-workload telemetry.
- Do not add an agent-ledger failure/correction anecdote unless it is a real, legible lifecycle example with a clear consequence. The current policy, human gate, separate ledgers, and explicit limits are enough; inventing procedural drama would weaken trust.

## Augmentation and preprocessing facts

### AFHQ Cats — confirmed

| Topic | Confirmed public-safe fact | Evidence |
| --- | --- | --- |
| Split | 5,153 train cats; 500 separate held-out cats. Training does not open the test cache. | `reports/afhq_cat_sit_b_128_setup.md`; `reports/afhq_cat_sit_b_128_cache_stats.md` |
| Train transform | Four deterministic variants per source using `RandomResizedCrop(scale=0.85-1.0, ratio=1.0)` with alternate horizontal flip. | `reports/afhq_cat_sit_b_128_setup.md` (Design; Final Pre-10k Preflight) |
| Evaluation transform | One deterministic center-square crop per source, without train augmentation. | `reports/afhq_cat_sit_b_128_setup.md` (Design) |
| Cache provenance | Manifest records relative path, augmentation seed, split, source/pixel/latent SHA-256; creation fails when the four train variants are not pixel-distinct. | `reports/afhq_cat_sit_b_128_setup.md` (Design); `reports/afhq_cat_sit_b_128_cache_stats.md` |
| REPA alignment | Teacher features use the exact cached crop/flip associated with the latent, then teacher-specific resize/normalisation; cache preparation cross-checks path, augmentation seed/index, split, and source hash. | `reports/afhq_cat_sit_b_128_repa_readiness.md` (REPA; Teacher Feature Cache) |

**Not confirmed for public wording:** a measured augmentation ablation, a causal estimate of augmentation benefit, a claim that the variants remove all leakage/memorisation risk, or an exact per-variant crop/flip distribution beyond the reported deterministic recipe.

### Imagenette — confirmed

| Topic | Confirmed public-safe fact | Evidence |
| --- | --- | --- |
| Train preprocessing | Deterministic RGB resize/center-crop to 128 px; cache seed 123; 9,469 FP16 latent items. | `reports/imagenette_sit_s_128_baseline_100k.md` (Frozen cache metadata) |
| Augmentation | No augmentation or horizontal flip was active for the DINO feature cache; the teacher repeats the deterministic 128 px crop used for the VAE cache. | `reports/imagenette_sit_s_128_repa_setup.md` (Teacher cache) |
| Teacher preprocessing | Teacher input is resized to 224 px with bicubic antialiasing and ImageNet normalisation after reproducing the latent-cache crop. | `reports/imagenette_sit_s_128_repa_setup.md` (Teacher cache) |
| Pairing controls | Feature construction checks the exact source relative path and label selected by the latent cache. | `reports/imagenette_sit_s_128_repa_setup.md` (Teacher cache; Verification coverage) |

**Not confirmed for public wording:** the exact resize interpolation/crop geometry before the documented 128 px result, a separately held-out Imagenette split/count in the public reports reviewed here, or an augmentation ablation. Do not infer AFHQ-style multi-view augmentation for Imagenette.

## Duplication audit

| Material | README (2–3 min) | Case study (5–7 min) | Retrospective (8–12 min) | Recommendation |
| --- | --- | --- | --- | --- |
| AFHQ quick-200 metrics and caveat | Necessary headline table. | Necessary interpretation and selection reasoning. | Should reference rather than repeat all numbers. | Keep current hierarchy; add only the data-pipeline paragraph to the case. |
| DDPM → SiT progression | One sentence in “What was built.” | Four-stage narrative. | Technical explanation of model/objective/cache transition. | Keep; add the objective-transition sentence only in retrospective. |
| Agent orchestration | Brief pillar and diagram. | Ownership, human gate, two ledgers. | Reference policy, not another full explanation. | No addition needed. |
| Token observation | Omitted from headline. | Correctly quarantined as operational and anecdotal. | No need to repeat. | Keep exactly once in the case study. |
| Data preparation/provenance | Currently only implicit via links/scope. | Missing. | Cache discussion exists but misses AFHQ augmentation and exact teacher-view pairing. | Add P1 once in case, P2 detail once in retrospective. |
| Licensing/availability | Scope link to attribution. | Not needed. | Not needed. | Leave details in `THIRD_PARTY.md` and README link; do not duplicate terms. |

## Suggested final size

Apply only P1 plus P2 items 2–4: approximately **150–190 net words** total (about 45–55 in the case study and 105–135 in the retrospective). The README should receive **0 net words**. This preserves the intended reading hierarchy while making data discipline and ML judgment easier to discuss in an interview.

## Audit limits

This review cites tracked documentation and ledger records. It did not re-run ML commands, inspect local datasets/checkpoints, validate external licence pages, or establish causal effects of augmentation, REPA, throughput choices, or token routing.
