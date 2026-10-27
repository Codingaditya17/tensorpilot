# Baseline: TensorPilot v1.0.0-rc2-3 7B BF16 session on one RTX 4090

Immutable evidence from the first full 7B session (2026-10-27 UTC, Pod `wydrcdr7xnqbwf`, RTX 4090, driver 595.91.07, image
`cuda-experiment@sha256:7e65d7ca…c446e56`): the mandatory probe gate, the serial baseline, continuous batching up to 1,024 clients,
the crash experiment, and **the failed vLLM stage**. The quality evaluation was not run.

- The four archives here are exactly as exported from the Pod (`*.sha256` sidecars, `SHA256SUMS` for all five).
- The fifth archive, `7b-bf16-prod-20261027T001850Z.tar.gz` (162 MB, the batching stage), is **not committed**; `SHA256SUMS` records its
  hash and size. Keep the original: `report.md`, `report-tables.json` and `acceptance-verdict.*` were generated from it.
- `summary.json` is a derived table (matched serial/batch points, the scale sweep). `report.md` is `tensorpilot.report_v10` output and
  `acceptance-verdict.*` is `tensorpilot.acceptance` output, both run on the five archives with no ledger, no archive list and no
  billing figure (the ledger was not copied from the Pod), so the budget and archive-verification items are INCONCLUSIVE there.
- The vLLM archive holds `engine.log`, which contains the failure's root cause: Triton could not find a C compiler. Analysis and the
  fix: [`docs/v10-7b-results.md`](../../../docs/v10-7b-results.md).

Do not edit or replace these files.
