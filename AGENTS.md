# Project Agent Rules

- After every completed test suite, benchmark, evaluation, or training milestone, update `PROJECT_LOG.md` and the relevant report with the commands actually run, material results, and the decision taken. Do not log minor intermediate steps.
- Do not add checkpoints, full evaluation outputs, datasets, or other large generated artifacts to Git. Small representative grids may be added only under `docs/assets/` when needed by a report.
- Before planning an ML experiment, read `ML_PROJECT_ROADMAP.md`. After an agreed roadmap change, update only that file; do not create duplicate roadmap files.

## Supervisor And Worker Roles

- The root/main agent acts only as the project supervisor. It reads `ML_PROJECT_ROADMAP.md`, reports, the experiment ledger, checkpoint metadata, and Git state; defines the next step; writes the worker task specification and acceptance criteria; reviews reported evidence; and makes the final continue/stop/change/freeze decision.
- The root/main agent must not edit implementation code or experiment configs and must not directly run tests, benchmarks, dataset preparation, cache creation, training, sampling, or evaluation commands.
- Delegate all code inspection and modification, routine investigation, commands, tests, benchmarks, ML runs, and experiment logging to a worker subagent. Use a lower-cost worker model for routine bounded work when available; escalate the worker model only when task complexity requires it.
- Use one primary worker for a write-heavy or ML experiment task. Do not run multiple workers against the same code paths, output directory, checkpoint lineage, or experiment artifacts concurrently.
- Every worker task must specify scope, files and artifacts in bounds, commands or milestones allowed, required stop conditions, reporting requirements, and acceptance criteria. The worker must not broaden the experiment or change the roadmap without supervisor approval.
- The worker is responsible for updating `PROJECT_LOG.md`, the relevant report, and `reports/experiment_ledger.jsonl` for every material ML operation it actually runs. The supervisor verifies those records before accepting the milestone.
- The supervisor may perform read-only inspection of reports, ledger entries, checkpoint metadata, generated summaries, and Git status/log/diff statistics. If implementation-level verification is needed, delegate that verification to a worker rather than inspecting or modifying the code directly.

## Agent Orchestration And Audit

- Project agent limits are defined in `.codex/config.toml`; executable profiles are in `.codex/agents/`; detailed lifecycle and failure rules are in `docs/agent_orchestration.md`.
- Use `luna_clerk` at `none` only for deterministic clerical work, `terra_worker` at `low` for the default bounded implementation/validation work, and `sol_specialist` at `high` only with explicit supervisor approval for complex or high-risk work. If Luna is unavailable in the current session, fall back to Terra at `low`.
- A worker must not change its model or reasoning level, delegate, or broaden scope without supervisor approval. Use one write-heavy worker for overlapping mutable scope; long training and evaluation remain human-gated.
- Every worker task appends a `started` event and one terminal event to `reports/agent_execution_ledger.jsonl`, validated by `reports/agent_execution_ledger.schema.json`. Worker `started`, `completed`, `failed`, and `interrupted` events must set `supervisor_decision` to `null`; only the supervisor appends a later `reviewed` event with an explicit decision. Link real ML operations to their `reports/experiment_ledger.jsonl` event IDs; do not invent token or credit data.

## ML Experiment Logging

For every material ML operation that actually runs -- dataset preparation, cache creation, benchmark, smoke test, training milestone, evaluation, comparison, checkpoint freeze, or experiment closeout -- the agent must append a structured event to `reports/experiment_ledger.jsonl` before its final response.

Rules:
- Record only actions and results that actually occurred.
- Clearly distinguish completed, failed, skipped, and pending work.
- Include exact commands, config/data/checkpoint hashes, runtime context, metrics, artifacts, and the resulting continue/stop/change/freeze decision when applicable.
- Use repository-relative paths and never record secrets.
- Never edit or delete previous ledger events. Add a correction event when earlier information is wrong.
- Planned commands must not be recorded as completed.
- The final response must state which ledger events were appended.
- This requirement applies to ML experiment operations, not ordinary documentation-only or unrelated code tasks.
