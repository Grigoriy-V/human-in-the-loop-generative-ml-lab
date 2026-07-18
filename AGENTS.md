# Project Agent Rules

- After every completed test suite, benchmark, evaluation, or training milestone, update `PROJECT_LOG.md` and the relevant report with the commands actually run, material results, and the decision taken. Do not log minor intermediate steps.
- Do not add checkpoints, full evaluation outputs, datasets, or other large generated artifacts to Git. Small representative grids may be added only under `docs/assets/` when needed by a report.
- Before planning an ML experiment, read `ML_PROJECT_ROADMAP.md`. After an agreed roadmap change, update only that file; do not create duplicate roadmap files.

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
