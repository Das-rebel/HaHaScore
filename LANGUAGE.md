# HaHaScore Architecture Glossary

This is the vocabulary used in architecture discussions. Terms from the `improve-codebase-architecture` skill adapted to this repo.

- **Module** — Anything with an interface and an implementation. In HaHaScore: `JesterSource` (data loader), `RobertaRegressor` (regression head), `laugho_cv.repeated_cv` (validation harness).
- **Interface** — Everything a caller must know to use a module. For `JesterSource`: constructor takes `output_dir`, exposes `write_csv(csv_path) -> (csv_path, n_in, n_out)`, writes 3-column CSV.
- **Implementation** — The code inside. For `JesterSource`: HTTP calls to HuggingFace, pyarrow parquet reads, CSV writes.
- **Depth** — Leverage at the interface. A deep module (`JesterSource.write_csv`) hides HTTP retries, parquet parsing, NaN filtering, and CSV writing behind a single function call.
- **Seam** — Where an interface lives. The seam between `JesterSource` and Kaggle kernels is a Python import — swap the adapter to use AudioSet instead of Jester without touching kernels.
- **Adapter** — A concrete thing satisfying an interface at a seam. `JesterSource` is one adapter; a hypothetical `AudioSetSource` would be a second adapter at the same seam.
- **Leverage** — What callers get from depth. Adding a 4th dataset = 1 new Source, not 4 edits to Kaggle kernels.
- **Locality** — What maintainers get from depth. Schema bug fix in `JesterSource` (commit 5378590, `rating` not `Z_rating`) now lands in ONE place instead of 4.

## The deletion test

Before adding a module, ask: if you deleted it, would complexity vanish (pass-through) or reappear across N callers (earning its keep)? If it reappears, the module was earning its keep.

## Deep modules in HaHaScore

| Module | Interface | Why deep |
|---|---|---|
| `JesterSource.write_csv` | `(csv_path, n_in, n_out)` | Hides HTTP, parquet, NaN filter, CSV writing, 5 shards |
| `RobertaRegressor` | `(freeze_backbone, use_fp16, use_grad_checkpointing)` | Hides backbone + head + fp16 + grad_checkpointing config |
| `kaggle_run.py --mode X` | `m3_recipe / 5x3cv / finetuned` | Hides all Kaggle-kernel differences behind a CLI flag |
| `laugho_cv.repeated_cv` | (task adapter) | Hides classification vs regression bootstrap CI logic |

## Shallow modules to avoid

- Copy-pasted kernel scripts (4 files with 80% boilerplate overlap)
- A 3rd task in `laugho_cv.py` via copy-paste
- A 5th dataset via copy-paste in Kaggle kernels
