# ML Experiment Logging

`reports/experiment_ledger.jsonl` is the append-only machine-readable record of material ML operations. One line is one factually occurred event, validated by `reports/experiment_ledger.schema.json`.

## Event Model

Every event has `schema_version`, a unique `event_id`, and `timestamp_utc`. Historical entries may use the UTC time of the Git commit that preserved the evidence; their `notes` must say so. `experiment_id` groups all events for one scoped run.

`event_type` describes the operation: `experiment_created`, `data_preflight`, `cache_created`, `smoke_test`, `benchmark`, `training_milestone`, `evaluation`, `decision`, `experiment_closed`, or `correction`. `status` is one of `completed`, `failed`, `skipped`, or `pending`.

`git_commit`, `config_path`, `config_sha256`, `dataset_fingerprint`, `checkpoint_path`, `checkpoint_step`, and `checkpoint_sha256` bind the event to reproducible inputs. Paths are always repository-relative. `exact_command` contains the command actually run; use `null` when historical evidence did not retain one rather than reconstructing it as fact.

`runtime` always contains `device`, `gpu`, `dtype`, `batch`, `effective_batch`, and `duration_seconds`; unavailable values are `null`. `metrics` stores measured scalar or structured results. `artifacts` stores repository-relative report, grid, cache, or output paths. `decision` is `continue`, `stop`, `change`, `freeze`, or `null`; `decision_reason` explains it. `notes` captures bounded context and uncertainty.

## Example

```json
{"schema_version":"1.0","event_id":"example-cache-001","timestamp_utc":"2026-07-18T10:00:00Z","experiment_id":"example","event_type":"cache_created","status":"completed","git_commit":"abc123","config_path":"mini_diffusion/configs/example.yaml","config_sha256":"...","dataset_fingerprint":"...","checkpoint_path":null,"checkpoint_step":null,"checkpoint_sha256":null,"exact_command":".\\.venv\\Scripts\\python.exe mini_diffusion\\prepare_latents.py --config mini_diffusion\\configs\\example.yaml","runtime":{"device":"cuda","gpu":"NVIDIA GeForce RTX 4090","dtype":"bf16","batch":32,"effective_batch":32,"duration_seconds":18.4},"metrics":{"cached_latents":1000},"artifacts":{"cache":"outputs/example/latents/train.pt"},"decision":"continue","decision_reason":"Cache validation passed.","notes":null}
```

## Append-Only Rules

- Append a new JSON object after an operation actually completes, fails, is skipped, or is explicitly pending. Do not rewrite or delete earlier lines.
- Do not record planned commands as `completed`. A plan belongs in a report or issue until it runs.
- Do not invent unavailable values. Store `null` and state the limit in `notes`.
- Never write absolute paths, tokens, credentials, or personal data.
- Keep generated checkpoints, datasets, and full evaluation outputs out of Git; ledger entries may reference them by relative path and SHA-256.
- Correct an error with a new `correction` event. Its `notes` must name the earlier `event_id`, describe the error, and give the corrected value. The original event remains unchanged.

## Agent And Skill Use

Agents append the event before their final response after a material ML operation. Skills can query the JSONL by `experiment_id`, checkpoint step, decision, or metric to build reproducible comparisons. A future ML Training Playbook will derive runbooks, gate checks, checkpoint selection, and agent orchestration state from this ledger rather than from prose-only reports.

## Standard Experiment Comparisons

Standard experiment comparisons run through one comparison CLI with a frozen protocol config. Use individual manual evaluation commands only to diagnose a case the comparison CLI does not support; do not use them to assemble a reported comparison.

For the AFHQ Cats baseline versus REPA 10k/20k quick protocol:

```powershell
.\.venv\Scripts\python.exe mini_diffusion\evaluate_comparison.py --config mini_diffusion\configs\evaluation\afhq_cat_baseline_vs_repa_10k_20k.yaml
```

The CLI verifies the checkpoint inputs, uses shared fixed seeds and held-out reference features, writes JSON/CSV/Markdown plus paired grids, verifies that checkpoint hashes did not change, and appends its completed event to the ledger.
