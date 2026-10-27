# Reproduction

Every number and figure in `paper/main.tex` comes from these steps. Run them from the repository root, in this order.
The step numbers are fixed labels and are not consecutive. Raw data: [`data/README.md`](../data/README.md).

```bash
uv sync
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
# selection series (MIT, AskUbuntu, Enron): fixed settings of the detectors
for s in 01_graphs 02_moments 03_main_table 08_baselines 09_tire; do uv run python repro/$s.py; done
for s in 03_main_table 08_baselines 09_tire; do uv run python repro/$s.py --corrected; done
# the 13 datasets, the source papers' own code, planted changes
uv run python repro/10_bench_data.py && uv run python repro/10_bench_data.py MIT_PC ENRON_PC
uv run python repro/11_bench_scores.py && uv run python repro/11_bench_scores.py MIT_PC ENRON_PC
for s in 13_bench_g0 12_bench_eval 14_plants 17_peelclauset; do uv run python repro/$s.py; done
uv run python repro/16_identity.py checks bench plants synth new plantsnew --workers 6
uv run python repro/19_activity.py
uv run python repro/22_settings.py --sections ABCDEFGHJKLNMR --sgnn
uv run python repro/17_peelclauset.py --why
for s in 23_inference 30_fig_main; do uv run python repro/$s.py; done
cd paper && latexmk -pdf main.tex
```

Steps 03, 12, 14, 16, 17, 19, 22 and 23 print their tables with `--summary`. Expensive results are cached under
`repro/data/` with keys that hash the inputs and the code, so a rerun recomputes only what changed. Steps use the CPU;
most need under 6 GB per process.

| Step | What it does | Time (first run) |
|---|---|---|
| `01_graphs`, `02_moments` | graphs and per-snapshot counts of the selection series | 4 min |
| `03_main_table`, `08_baselines`, `09_tire` | tuned F1 and the fixed setting of each detector | 6 min |
| `10_bench_data` | builds 12 of the 13 datasets (AskUbuntu comes from steps 01 and 02) and Peel and Clauset's series | 4 min |
| `11_bench_scores` | scores every detector at its fixed setting | 16 min |
| `12_bench_eval` | event percentile (EP) per dataset; source papers' metrics | 1 min |
| `13_bench_g0` | runs LAD's and s-GNN's own code and the SCPD port `methods/scpd.py` on their own data | 4 min (s-GNN: 24 min) |
| `14_plants` | planted rewiring of real networks (Figure 2a) | 2 h on 14 processes |
| `16_identity` | node-aligned detectors, turnover, plants, block models (Figure 2) | 7 h on 8–14 processes |
| `17_peelclauset` | Peel and Clauset's constructions and GHRG | GHRG fits, then minutes |
| `19_activity` | structural statistics; EP by activity change (Figure 1b) | 1 min |
| `22_settings` | other setting rules, count variants, LAD eigenvalues, residual design | hours |
| `23_inference` | pooled EP, intervals, tests, bounds, subsets, event types | 3 min |
| `30_fig_main` | Figures 1 and 2 | 5 s |

Shared code: `common.py` (paths, data, F1, selection, cache keys), `evaluate.py` (windows, EP, bootstrap),
`inference.py` (paired bootstraps, sign-flip tests, Holm, bounds), `rows.py` (datasets, settings, features),
`plants.py` (planted changes and structural statistics), `figstyle.py`. Detectors are in `methods/`.

## Where the numbers come from

| Paper | Source |
|---|---|
| Pooled EP, intervals, tests, bounds, subsets, event types (EP by event type: Table 3; large or extreme events by type, `event_types.by_stratum`: App. D), causal counts, turnover, the common tolerance (App. D), Tables 3, 7, 8, 9 (abstract, Sec. 5, App. D) | `23_inference` → `results/inference.json` |
| Table 2 (datasets), per-dataset EP, source-paper metrics, reproduction checks (App. B), s-GNN's mean adjusted F1 over 16 runs (Sec. 4; `bench_g0.json` `sgnn.mean`) | `12_bench_eval`, `13_bench_g0` → `results/bench.json`, `bench_g0.json` |
| Fixed settings and tuned F1 of the detectors (Sec. 4, App. A) | `03_main_table`, `08_baselines`, `09_tire` → `results/{main_table,baselines,tire,bench_settings}.json` |
| Reselection with the calendar MIT dates (App. A) | the same steps with `--corrected` → `results/{main_table,baselines,tire}_corrected.json` |
| Other setting rules (A), count variants (E), Table 6 (F), s-GNN tuning (I), identity rows under the rules of A (J), score test (K), NetSimile and other baselines (L), LAD eigenvalues (M), grid sizes (N), residual design (R) | `22_settings` → `results/settings.json`, section in brackets |
| Node-aligned detectors and turnover (Sec. 5.1, App. D) | `16_identity` → `results/identity.json`, `identity_new.json` |
| Block models (Sec. 5.2, App. E) | `16_identity` → `results/identity_synth.json`, `identity_new.json` |
| Planted rewiring (Sec. 5.2, App. E, Table 10) | `14_plants`, `16_identity` → `results/plants.json`, `identity_plants.json`, `identity_plants_new.json` |
| Structural statistics, EP by activity change (Figure 1b; Sec. 5.1: the share of large or extreme events, `C_stratum`, and of event-free windows each ranked like an event among the event-free windows that do not overlap it, `C_null_share` (also per count), and their per-dataset rows in Table 5; App. D: extreme minus ordinary EP within datasets, `C_within`, post hoc) | `19_activity` → `results/activity.json` |
| Peel and Clauset's series (App. B) | `17_peelclauset` → `results/peelclauset.json`; the pool with both series: `results/inference.json` (`peel_clauset`) |
| Figures 1 and 2 | `30_fig_main` → `paper/figures/fig_main.pdf`, `fig_structure.pdf` (its log prints every plotted value) |

The `spec` strings inside `inference.json`, `peelclauset.json` and `settings.json` name the specs by working names;
the specs are the files in `prereg/specs/`.

## Setup notes

* s-GNN (`13_bench_g0 --sgnn`, `22_settings --sgnn`) needs a DGL environment in `data/raw/sgnn/venv`:
  `uv venv --python 3.11 data/raw/sgnn/venv && uv pip install --python data/raw/sgnn/venv/bin/python torch==2.1.2
  dgl==1.1.3 numpy==1.26.4 scipy==1.12.0 networkx==3.6.1 scikit-learn pytorch-lightning==1.6.5 torchmetrics==0.11.4
  "setuptools<70" ruptures matplotlib psutil pandas==2.2.3`.
* `17_peelclauset` compiles the authors' GHRG code with `g++`; `--validate` runs their Python 2 code under
  `uv run --python 3.8 --with dendropy==4.0.2`.
* `--corrected` reselects the settings with the corrected MIT dates; the paper's fixed settings were selected with
  the earlier dates (`config.SELECTION_CPS`).

`prereg/` holds:

* `PROTOCOL.md`, `DEVIATIONS.md`, `INTERPRETATIONS.md`: the construction and event rules that the dataset builders
  follow;
* `LABEL_AUDIT.md`, `scc_audit_events.json`: the label audit and the audited stablecoin events (step 23);
* `IDENTITY.md`, `TURNOVER.md`: the checks and reading rules of step 16 and the turnover rule of step 23;
* `results/EEG.json`: the reference statistics of the EEG build check (`data/downloaders/chbmit.py`);
* `specs/`: the analysis specs, indexed in [`prereg/specs/README.md`](prereg/specs/README.md).
