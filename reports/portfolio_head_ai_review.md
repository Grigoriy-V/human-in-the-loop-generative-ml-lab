# Head of AI portfolio review

**Review type:** independent, static, final-style review of the portfolio package
**Review snapshot:** 2026-07-18; documentation and tracked evidence only
**Reviewer lens:** Head of AI assessing an Applied AI / AI Engineer candidate project

## Executive verdict: conditionally ready

The portfolio is ready to show in a conversation or private review. It makes a credible, unusually well-evidenced case that the author can build a generative-ML workflow, make bounded experimental decisions, and use agents without surrendering ownership. The public release remains conditional on one **separate legal/release choice**: choose and add a project-code license. That is the sole remaining P1 release gate; it is not a defect in the ML narrative or the content package.

I found no P0 issue and no P1 content defect. The current materials correctly call the SiT path a latent **flow/velocity** model rather than falsely presenting it as DDPM diffusion, distinguish quick-200 from full evaluation, preserve the precision/recall trade-off, and label the approximate 5x token observation as anecdotal rather than a result.

## What creates hiring signal

- **Decision quality rather than metric theatre.** The AFHQ selection is useful because it says both why early-stop REPA won (FID/KID) and why it is not a blanket winner (baseline precision/recall, no full-1000, same-seed rerun is not a replicate). See [README](../README.md#selected-result-afhq-cats), [case study](../docs/portfolio_case_study.md#ml-outcomes), and the [AFHQ report](afhq_cat_sit_b_128_repa_early_stop_results.md).
- **Real experimental maturity.** The project records partial/negative outcomes rather than laundering them into successes: Tiny ImageNet stopped at 150k/400k; the Imagenette claim is explicitly cross-step; always-on REPA was excluded. That is stronger evidence of judgment than a longer list of runs.
- **Human ownership is explicit and believable.** The opening README plus [case-study ownership section](../docs/portfolio_case_study.md#what-i-personally-owned) makes the author’s decisions, manual long-run gate, and model-selection authority clear. Agents are described as bounded execution, not autonomous researchers.
- **Orchestration is a real supporting system, not a buzzword.** Separate append-only experiment and agent ledgers, role routing, bounded task specifications, and supervisor-only decisions are concrete controls. The [pipeline visual](../docs/assets/portfolio_agent_pipeline.svg) expresses this in seconds.
- **Evidence navigation is unusually good for a portfolio repo.** The README’s result table, evidence map, reports, hashes, configuration link, curated visual, and the no-download verifier create a reviewable trail without committing datasets or checkpoints.

## Findings and remediation

### P1 — release gate, not a content defect

| Location | Finding | Concrete remediation |
| --- | --- | --- |
| Repository root; [README scope](../README.md#scope-and-availability); [THIRD_PARTY](../THIRD_PARTY.md#purpose-and-boundary) | No project-code license is tracked. Third-party attribution is now present and appropriately cautious, but it does not grant rights to the project code. | The owner must choose the intended public/commercial posture and add the corresponding `LICENSE`. Do not infer a license from dataset/model notices. Re-run the public verifier and final release checks after this owner decision. |

### P2 — improvements that would sharpen an already credible presentation

| Location | Finding | Concrete remediation |
| --- | --- | --- |
| [README opening and What was built](../README.md#what-was-built) | First-minute clarity is strong, but SiT and REPA appear before a non-specialist reader gets a one-line expansion. A Head of AI will know them; an adjacent manager may not. | Add a parenthetical once: “SiT, a transformer-based latent flow model” and “REPA, a frozen-teacher representation-alignment auxiliary loss.” Keep the rest compact. |
| [README engineering highlights](../README.md#engineering-highlights) | The 27.15% throughput result is defensible, but the word “historically” makes the comparator slightly less immediate than “repeated control.” The report contains both. | Prefer one short qualifier such as “versus the historical baseline; repeated A0 was also measured” or link directly to the table. Do not promote this result above the model-selection story. |
| [case study Operational observation](../docs/portfolio_case_study.md#operational-observation-not-a-result) and [retrospective Cost and telemetry](../docs/technical_retrospective.md#7-cost-and-token-telemetry-what-is-known-and-what-is-not) | The token paragraph is exceptionally honest, but it is not yet a portfolio strength. Repeating it in two long documents makes the unmeasured number slightly more salient than necessary. It does not currently harm credibility because each use states the boundary. | Keep the case-study paragraph as a short operational note and retain the detailed instrumentation plan in the retrospective. Never put “5x” in the README, title, résumé bullet, or visual until matched-workload telemetry exists. |
| [fixed-seed comparison](../docs/assets/portfolio_afhq_fixed_seed_comparison.png) | The grid is legible, labelled as canonical/fixed-seed, and visually useful. It is nevertheless easy for a fast reviewer to read as the selection evidence rather than an illustration. | Make the existing bottom caption even more prominent in alt text/copy adjacent to the image: model selection used the quick-200 metrics; this eight-seed grid is illustrative. No hand-picked-samples claim is already good. |
| [reproducibility guide](../docs/reproducibility.md#what-can-be-verified-from-a-fresh-clone) | The safe verifier is a strong public-repo check, but reviewers must still understand it validates evidence integrity, not training reproducibility or metrics. The guide says this clearly. | Preserve that sentence in any future README shortening. Add a release badge only if CI is actually enabled and passing on the public remote. |

## Rubric scores

| Criterion | Score / 5 | Evidence-based assessment |
| --- | ---: | --- |
| First 60-second executive clarity | 4.5 | README leads with outcome, ownership, the AFHQ table, limitations, and visuals; minor acronym expansion would make it broader-audience friendly. |
| Technical credibility and depth | 4.5 | Direct reports, fixed protocols, hashes, raw/EMA distinction, failure diagnosis, and a defensible flow-vs-DDPM explanation are present. |
| Human ownership vs. agent contribution | 5.0 | The human’s direction, approval, manual launch, and final decision authority are repeated consistently and supported by policy/ledgers. |
| Experiment design and evaluation maturity | 4.5 | Controlled AFHQ comparison, caveats, diagnostics, and explicit stops are excellent; quick-200/no replicate limits are correctly retained. |
| Orchestration novelty and honesty | 4.5 | The controls and separation of audit records are concrete; the text avoids falsely claiming autonomous research or production MLOps. |
| Claim/evidence traceability | 5.0 | Claim matrix, result reports, config references, event IDs, hashes, and static verifier give a reviewer a clear trail. |
| Visual quality and narrative fit | 4.5 | Four visuals are clean, readable at normal GitHub scale, protocol-labelled, and support—not replace—the metric claims. The fixed-seed grid needs its illustrative role kept prominent. |
| Reproducibility / public-repo readiness | 3.5 | Strong evidence package and no-download verification; intentionally incomplete environment pinning, local-only artifacts, and the missing code license prevent a full score. |
| Concision and readability | 4.5 | README is skimmable; case study and retrospective are worth reading because they add decisions and caveats rather than duplicate raw ledgers. |
| Hiring signal for Applied AI / AI Engineer | 4.5 | Strong signal for an experimentation-minded Applied AI/ML engineer: implementation, evaluation, artifacts, and operational controls. It is not presented as production-scale research infrastructure, which improves trust. |

## Visual and claim checks

- **ML progression SVG:** accurate sequence and excellent caveat banner: partial Tiny ImageNet, cross-step Imagenette, quick-protocol AFHQ. It does not overstate completion.
- **AFHQ metrics SVG:** values, metric directions, the FID/KID vs precision/recall split, protocol, and skipped full-1000 are all visible. Bars are a qualitative aid with exact labels, so no misleading scale claim is implied.
- **AFHQ fixed-seed grid:** clear model/seed labels; it truthfully states canonical fixed seeds and that visuals are illustrative. It should remain paired with the metric table.
- **Agent workflow SVG:** accurately represents the documented authority flow and manual gate. It does not call workers independent decision-makers.
- **Terminology:** DDPM is used for the early pixel model; later SiT is called a latent flow/velocity path. This is accurate and avoids the common “everything is diffusion” conflation.
- **Token observation:** safe as written: it is user-reported, directional, without exact counters or causal attribution. It would become credibility-damaging only if promoted to an efficiency result before instrumentation.

## Suggested verbal pitch (30 seconds)

“I built this as a hands-on generative-ML lab: starting with a PyTorch DDPM, moving to a latent SiT flow model, and then building an evaluator so selection was based on a fixed protocol rather than a pretty grid. On AFHQ Cats, an early-stop representation-alignment variant won FID and KID, while I kept the baseline’s precision/recall advantage and the skipped full evaluation visible. I also built a human-gated agent workflow around the work: agents can execute bounded tasks, but I approve experiments, launch long GPU runs, and make the final decision from an auditable evidence trail.”

## Likely interview questions

1. “Why did you freeze early-stop REPA when baseline precision and recall were higher, and what would make you reverse that decision?”
2. “How would you design the next AFHQ run so the REPA conclusion is statistically and experimentally stronger without wasting GPU budget?”
3. “What evidence would you collect to prove orchestration actually improves token cost or research throughput, instead of simply moving work between agents?”

## Static checks performed

- `python tools/verify_public_repo.py` — passed: 2 ledgers, 7 public docs, 10 required evidence files, 138 tracked files.
- Markdown/link coverage is included in the verifier; referenced portfolio documentation and evidence paths resolved.
- `git diff --check` — passed.
- Static tracked-file check confirmed no project-code `LICENSE`, `COPYING`, or `NOTICE` file. This is the sole remaining release choice noted above.

## Review limits

This was a static review. It did not rerun ML training, evaluation, tests, downloads, external license verification, or inspect ignored checkpoints/evaluation outputs. The findings rely on the tracked reports, configuration references, ledgers, and curated visual assets.
