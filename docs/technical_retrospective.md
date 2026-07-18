# Technical retrospective: learning a generative ML workflow

This is a selective technical account of what changed, what stopped, and why. It is not a research paper and it does not replace the append-only experiment record.

![Technical progression from pixel DDPM to latent SiT, evaluator, and AFHQ early-stop selection](assets/portfolio_ml_progression.svg)

## 1. Start with a complete, inspectable loop

The first target was a class-conditioned CIFAR-10 DDPM in plain PyTorch: forward noising, a U-Net noise predictor, classifier-free guidance, EMA, checkpoint/resume, and deterministic sampling. The useful lesson was not merely that a model could train. It was that a generative system needs a reliable loop around the model: a short debug path, an overfit check, predictable sample generation, saved RNG state, and a way to investigate failures.

An early sampler issue produced intermittent black/white outputs. The repair clipped predicted x0 before the posterior mean, kept reverse sampling FP32, introduced dedicated seeded generators, and added finite/saturation diagnostics. The outcome was a reproducible sampling path rather than a claim that a single visual grid proved quality. [Baseline report](../reports/cifar10_baseline.md)

The same stage included constrained performance work. On the fixed local benchmark protocol, optimisation increased measured throughput from 1,787.57 to 2,272.92 images/s (+27.15%) while preserving the specified model/objective semantics. This is a hardware- and protocol-bound result, not a general deployment claim. [Optimisation report](../reports/cifar10_optimization_report.md)

This first stage set a pattern for later work: separate correctness, performance, and quality questions. A throughput benchmark should not include sampling or logging overhead if the purpose is to compare training paths. A sampling fix should not be presented as a quality improvement without a controlled evaluation. A checkpoint that loads is not enough; resume needs to restore the state that defines the continuation.

## 2. Scale cautiously: Tiny ImageNet and latent SiT

Tiny ImageNet was the first scale-up exercise: 64×64 inputs, a larger U-Net, physical-batch versus accumulation choices, and data-loader concerns on Windows. The run stopped at 150k/400k steps (37.5%). Keeping the checkpoint and report while declining to present it as a finished quality result was deliberate scope control. [Partial milestone](../reports/tiny_imagenet_partial.md)

Stopping did not mean the run was useless. It exposed the memory/throughput trade-off between a larger physical batch and gradient accumulation, and it provided experience with a more demanding dataset path. But finishing the configured budget would have answered a lower-priority question than moving to latent transformers and a formal evaluator. The retained checkpoint is therefore evidence of a partial engineering milestone, not a quality claim and certainly not “37.5% accuracy.”

The next architectural change was from pixel DDPM/U-Net to a latent SiT-S/2 velocity model on Imagenette. A frozen SD VAE and cached latents made the diffusion/flow model operate on latent tensors; Heun sampling and transformer blocks replaced the earlier pixel-space reverse process. This transition introduced a more realistic set of concerns: cache provenance, decoder boundaries, raw-versus-EMA selection, and a more expensive evaluator. [SiT evaluator setup](../reports/imagenette_sit_evaluator_setup.md)

The pixel-to-latent move changed where errors could enter. In the DDPM path, the train and sample loop worked directly in image space. In the latent path, preprocessing had to select an image deterministically, encode it with the frozen VAE and its configured scale, preserve the association between latent and label, and later decode generated latents through the same model boundary. Cached latents saved repeated encoder work, but made the cache fingerprint and source manifest part of the experiment definition. A model checkpoint without the matching cache/config context was no longer sufficient evidence.

The flow formulation also changed the sampling contract. The SiT predicts a velocity field and the sampler integrates it with Euler or Heun steps, rather than applying the earlier DDPM posterior update. Fixed sampler type and step count therefore became mandatory comparison fields. The evaluator was built to hold those choices constant rather than allowing an improved sampler setting to masquerade as an improved checkpoint.

The modelling objective changed with the architecture: DDPM trained a noise predictor, whereas SiT trained a velocity field and sampled it through numerical integration. REPA added auxiliary representation alignment to the unchanged flow objective; it was not a replacement sampling method.

## 3. Compare representation alignment without erasing the caveat

REPA was introduced as a frozen DINOv2-teacher alignment term with a student projector. On Imagenette, the headline result is useful but narrow: under a fixed 1,000-sample EMA Heun-50 protocol, REPA at 350k versus a 100k baseline improved FID (130.96 vs 149.95), KID (0.05199 vs 0.06103), target accuracy (29.5% vs 26.9%), and recall (72.1% vs 51.0%), with slightly lower precision (23.9% vs 24.6%).

For AFHQ REPA, the frozen DINOv2 teacher received the exact cached crop/flip used for each VAE latent, followed by a 224-pixel bicubic resize and ImageNet normalization. Feature preparation verified path, augmentation seed/index, split, and source hash before writing features. This prevents mixing teacher and student views; it does not prove absence of data leakage.

The comparison is cross-step. There is no 350k non-REPA control, so it cannot establish that REPA converges faster or wins at the same training budget. That distinction influenced the next experiment: AFHQ Cats would use a direct, unified comparison rather than an aesthetic impression of different runs.

That caveat is an example of why experiment labels matter. “REPA 350k versus baseline 100k” is supported; “REPA converges faster” is not. More steps, a different representation objective, and a different point on the optimisation trajectory are confounded. The result was still useful—it justified studying when alignment should be active—but it could not settle the general REPA question.

## 4. Fixed comparison protocol and the AFHQ early-stop decision

The evaluator fixes the held-out reference split, seed range, sampler, CFG, VAE, and feature extractor. It reports FID, KID, precision, recall, finite/black-white/low-detail diagnostics, duplicates, and nearest-neighbour artifacts. It also records checkpoint hashes before and after evaluation and tests fixed-seed determinism. Those checks do not prove perceptual quality or absence of memorisation, but they reduce obvious comparison drift. [AFHQ result report](../reports/afhq_cat_sit_b_128_repa_early_stop_results.md)

For AFHQ Cats, three raw 20k variants were compared under the same quick protocol: held-out data, 200 fixed seeds (1000–1199), Heun-50, and CFG 1.0.

On the recorded local RTX 4090 cache benchmark, batch 128 with accumulation two was selected over physical batch 256 because it was faster (2,152.02 vs 2,057.33 images/s) and used less peak VRAM (5.27 vs 8.78 GB); this is protocol-bound, not a general hardware claim.

| Variant | FID ↓ | KID ↓ | Precision ↑ | Recall ↑ |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 48.051 | 0.02052 | **0.340** | **0.754** |
| Always-on REPA | 52.384 | 0.02531 | 0.310 | 0.722 |
| REPA 10k, then off | **45.787** | **0.01692** | 0.280 | 0.732 |

The early-stop design was motivated by a simple hypothesis: use alignment early for structure, then remove it to allow later texture/detail learning. It won FID/KID in this controlled quick comparison, whereas always-on REPA was excluded from finalist consideration. The supervisor froze the raw early-stop 20k checkpoint (SHA-256 `300b5600…16e57bd0`) as the current parent artifact.

The caveat is part of the decision, not a footnote: baseline remained better in precision and recall, full-1000 was deliberately skipped, EMA was not selected, and a redundant run with the same seed set is not an independent replicate. The outcome is a bounded model-selection result rather than a universal claim.

![AFHQ Cats metrics under the fixed quick-200 protocol](assets/portfolio_afhq_metrics.svg)

![Illustrative eight-seed AFHQ Cats grid for baseline, always-on REPA, and early-stop REPA; selection used quick-200 metrics](assets/portfolio_afhq_fixed_seed_comparison.png)

This eight-seed grid is illustrative. The selection decision came from the quick-200 metric protocol, not from choosing among these displayed samples.

Raw-versus-EMA was another decision rather than a default. EMA often improves sampling once its moving average represents a useful region of training, but the AFHQ reports show that the available EMA variants lagged the raw checkpoints. Early training history can make EMA slow to reflect a rapidly improving model. The project therefore evaluated and reported weight choice instead of assuming that “EMA is always better”; raw 20k became canonical for this stage.

FID/KID and precision/recall were read together. Early-stop moved the distribution-level distance metrics in the preferred direction, while baseline retained better estimated fidelity/coverage balance on precision and recall. That is enough to choose a candidate for the current objective, but not enough to erase the baseline or claim significance. Full-1000, sampler ablation, and CFG sweep were consciously omitted once the supervisor judged the bounded question answered.

## 5. Reproducibility habits: evidence, hashes, and resume discipline

The project treats checkpoints as evidence-bearing artifacts. Reports record the originating command where available, config/cache fingerprints, hash values, global step, and finite-state inspection. Resume checks restore model, optimizer, scheduler, and RNG state; evaluation is designed not to mutate the frozen checkpoint. Deterministic fixed-seed probes make it possible to notice accidental sampling drift.

Resume and fork semantics needed special care. A normal resume is expected to restore the optimiser, learning-rate scheduler, global step, and Python/NumPy/Torch/CUDA RNG states along with raw and EMA weights. The early-stop experiment was different: it intentionally forked from the REPA 10k checkpoint into an isolated output lineage, removed REPA/projector state, and continued without modifying the parent artifact. Distinct output directories and post-run hashes reduced the risk of silently overwriting the baseline or confusing checkpoints with different training objectives.

Gradient accumulation introduced a more subtle reporting risk. AFHQ used a physical batch of 128 with two accumulation steps to preserve an effective batch of 256 while retaining throughput and VRAM headroom. A later report identified that one TensorBoard loss tag summed microbatch flow losses while another represented only the final microbatch. The values were finite and useful as diagnostics, but not directly comparable as a clean per-update mean. The correct response was to document the limitation and require corrected/relabelled logging before another resume—not reinterpret the tag after the fact.

This is not a claim of universal reproducibility. The primary evidence was collected locally on Windows with an RTX 4090; packages are not fully pinned; datasets, checkpoints, caches, and full evaluation outputs are intentionally excluded from Git. What is tracked is the code, compact reports, configurations, hashes, and selected small visuals. [Reproducibility guide](reproducibility.md)

## 6. Human-agent lifecycle and corrections

The agent workflow was added to support this evidence discipline. The human approves direction and manually gates long runs. A supervisor scopes each task and reviews its result; a worker can change only its approved slice. Agent execution and material ML operations are recorded in separate append-only JSONL ledgers. Every worker task has a started event and one terminal event; a supervisor review is a distinct later action. [Lifecycle policy](agent_orchestration.md)

The orchestration itself also surfaced process corrections: timestamps must be captured programmatically in UTC; ledger corrections must be appended rather than rewriting history; and review decisions must remain separate from worker completion. These are modest controls, but they make a local learning project easier to audit.

The workflow is semi-automatic and human-gated. It is neither autonomous experimentation nor a production multi-agent platform. Luna, Terra, and Sol are routing profiles for increasingly demanding bounded tasks, not independent owners of ML direction.

![Human-supervised lifecycle with bounded workers and separate evidence ledgers](assets/portfolio_agent_pipeline.svg)

Audit corrections were treated as evidence about the process itself. A malformed or incomplete historical record is not silently edited, because that would make the ledger look cleaner while reducing trust. A new correction event identifies the problem and preserves the original. Programmatic UTC capture avoids agents typing plausible-looking timestamps. Terminal worker events keep `supervisor_decision` null because completion and acceptance are different facts.

Long runs stay manual for the same reason. They consume significant local GPU time, can create large ignored artifacts, and establish a checkpoint lineage that later evaluations depend on. An agent may prepare an exact command and acceptance criteria, but the human explicitly decides when to spend that compute and launches it. This makes the control boundary visible and prevents “finish the task” from silently expanding into an expensive experiment.

## 7. Costs, limits, and what I would improve next

The project incurred local GPU work, model/data downloads, and evaluation cost; it does not yet have a portable lockfile or per-agent token telemetry. A preliminary account usage view appeared roughly 5× lower after routing/orchestration setup, but that is only a user-reported, anecdotal, directional observation. There is no controlled before/after measurement, exact token count, causal attribution, or per-agent cost record.

Next, I would instrument raw input, cached input, output, and reasoning-token counts for every root and worker run, together with model/profile, task identifier, start/end time, result status, and workload class. A useful comparison needs matched tasks and the same observation window; it should separate routing effects from caching, prompt size, model choice, and changes in task complexity. Only then would token cost per accepted outcome become a defensible metric.

Portfolio packaging and public readiness are complete. The current frozen checkpoint remains the intended parent for a separately scoped Cats-to-all-AFHQ transfer experiment, which still requires separate human authorization and has not started.

For the compact portfolio narrative, see the [case study](portfolio_case_study.md). For the source-of-truth claim boundaries, see the [claim matrix](../reports/portfolio_claim_evidence_matrix.md).
