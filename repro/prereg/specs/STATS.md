# Statistics lane: analysis spec (written 2026-10-26, before any of these analyses ran)

Ledger items: L12, L22, L23, L24, L26, L27, L28, L29, L30, L31, L33, L67, L68.
Code: `repro/inference.py` (new shared module: the statistics below, one implementation), `repro/23_inference.py`
(benchmark inference over every row), `repro/plants.py` + `repro/14_plants.py` (paired plant metrics),
`repro/21_sealed.py` (sealed-test inference). Nothing here changes a score series or an EP value.

## 1. Benchmark B (step 23)

Rows. Core rows of step 12 (the PRIMARY list that step 12 writes to bench.json), the identity rows of step 16
(hamming, jaccard, deltacon, chenzhang, nbs, frz) and the wiring rows of step 19 (transitivity, assortativity,
fair_pair). Mask, events, tolerances and null windows: step 12's common mask, unchanged. Check S0: the per-event EP
of every core row equals bench.json (4 decimals) and the pooled EP equals bench.json pooled to 1e-12; the identity
and wiring rows equal identity.json / activity.json per-event EP to 4 decimals.

Estimand E1 (primary, as in the paper): mean over the datasets of a pool of the per-dataset mean EP (equal weight),
conditional on these datasets and on their null windows. E2: event-weighted mean over the pooled events.

Pools: all ten; >= 10 evaluable events; held-out 7 (without MIT, ENRON, AskUbuntu, the three F1-selection sets);
without post-hoc-labelled datasets (of the ten only MAG); interaction (8) and correlation (PM, EEG); drop each one
dataset; Peel-Clauset: all ten plus MIT_PC and ENRON_PC, and the ten with MIT_PC / ENRON_PC replacing MIT / ENRON.

Intervals (all paired: one resample serves every row; B = 10,000, seed 20261023):
  (a) event level: events resampled within each dataset (the paper's interval; conditional);
  (b) two-stage dataset level: datasets resampled, then events within each drawn dataset;
  (c) dataset only: datasets resampled, per-dataset means fixed (R3's interval).
Per dataset (small n): exact sign-flip test of the paired per-event differences (all 2^n sign patterns for
n <= 16, else 200,000 random patterns), two-sided p; 95% interval by inverting the test on a grid of shifts
(step 0.001 in [-1, 1]); with n <= 5 no shift can be rejected at 5% (minimum p = 2 / 2^n >= 0.0625), and the
interval is reported as not bounded. Coverage statement: a simulation draws n events (n = 2, 4, 5, 6, 9, 10, 20)
from the centred pooled per-event differences counts - SWORD-exact of the four datasets with >= 10 events
(4,000 replications each) and reports the coverage of the percentile bootstrap (B = 2,000) and of the
sign-flip interval at the true mean 0. Reading: report whatever coverage comes out.

Contrasts: counts minus each other row. Family F1 = the Table 2 detectors (exact, exact_z, cusum, bocpd, mmd, lad,
scpd, tire); family F2 = F1 plus the six identity and three wiring rows. Two-sided bootstrap p =
min(1, 2 min(P(d* <= 0), P(d* >= 0))) under each interval type; Holm adjustment within each family; simultaneous
intervals at level 1 - 0.05 / |family| (Bonferroni). Rank: in each draw the rows of F2 plus counts are ranked by
pooled EP; P(counts rank 1) and the 95% range of every row's rank.
Reading rule: a sentence "counts rank above X" is supported at the dataset level only if the Holm-adjusted
two-stage p of counts - X is below 0.05; otherwise "not distinguishable from X".

Also moved here from step 18 (PRIMARY / rows_of of step 12): detection rate within tol at fixed null quantiles on
benchmark B (evaluate.delay, alpha 0.01 and 0.05, H = 3 tol), unchanged definition.

## 2. Planted changes (step 14)

Per (background, cell, row), trials = (seed, offset, onset t0); pos = max score on [t0, t0 + 5) in the spliced
series, neg = the same window in the original series (same t0).
  paired AUC = mean over trials of 1[pos > neg] + 1/2 1[pos = neg];
  block bootstrap: unit = (segment, period index i of t0 = s0 + 20 + o + 40 i): all offsets and seeds of one
  period form one block (overlapping windows and the shared background stay together); B = 2,000; percentile 95%;
  increment = mean(pos - neg) / SD(neg over the onsets of all offsets);
  tie fraction = share of trials with pos = neg; zero fraction = share of pos and neg window maxima equal to 0.
  The seeds randomize only the plant; the background is fixed (stated).
Pre-set rules: a row is "not evaluable" in a cell if its tie fraction is >= 0.5. A structural cell is a null cell if
its median moved-edge share is < 0.01; null cells leave every count. fair_pair vs SWORD: paired AUC difference with
the same block resample; a cell counts for fair_pair (SWORD) if the 95% interval of the difference lies above
(below) 0. Cross-matched table: transitivity, assortativity and fair_pair in every cell; matched = assortativity on
assort, transitivity on triclose, fair_pair on both. The unpaired window-max AUC stays in the output as `auc`
(renamed "unpaired" in the text) so that the frozen decision rules P1-P4 keep their definition.
Attribution of a plant-caused SWORD alarm (D_planted >= theta > D_original, alpha 0.05): (i) largest term of the
planted split (the old label: names the background term that the plant pushed over theta); (ii) counterfactual:
the term with the largest increase of the planted over the original term magnitude at the same t, and the mean
increase of each term.

## 3. Sealed tests (step 21)

Fact: the best baseline is chosen by its point EP on the sealed set and held fixed in the bootstrap. Added:
(a) the maximum over the frozen baselines recomputed in every two-stage resample (same subject and event draws for
every row); (b) candidate minus each frozen baseline with Holm; (c) subject-weighted EP (mean over subjects of the
per-subject mean) as co-primary, same two-stage resample; (d) cluster-only bootstrap (subjects only, Davison and
Hinkley 1997) as sensitivity; (e) a-priori power of the pass rule (lower 95% bound > 0) assuming the true lead equals
the development estimate, normal approximation with the sealed SE = (hi - lo) / 3.92: power =
Phi(d_dev / SE - 1.96), and the minimum detectable lead at 80% power, 2.80 SE; (f) development minus sealed lead,
z = (d_dev - d_sealed) / sqrt(SE_dev^2 + SE_sealed^2), two-sided p. Development estimates: FRZ-12 on Siena: FRZ-12 -
tire on benchmark B's EEG (3 subjects); IG-S on Helsinki: its development leads on CHB-new and on Siena (each
reported); SWORD-exact on CHB-new: SWORD-exact - best baseline on benchmark B's EEG.
Wording rule: a sealed lead whose 95% interval contains 0 is "not confirmed"; if the interval also contains the
development estimate, the text must not say the development lead "did not hold" or "shrank".

## 4. Alarm counts (L12)

Every alarm count counts distinct alarms (the union over probe seeds) or uses exact moments; the numbers come from
results/activity.json E_table1_alarms_secondary (step 19).

## Deviations (logged as they happened)

- S1 (before the first run): the sign-flip test enumerates all patterns for n <= 12 (not 16) and uses the identity
  plus 5,000 random patterns above (not 200,000): the 16-event enumeration over the shift grid needs about 0.5 GB per
  call. The coverage simulation therefore reports the sign-flip coverage for n <= 12 only (n = 20: percentile only).
- S2: the sealed-test sensitivities (`inference` blocks of sealed.json) use inference.cluster_draws, a vectorized
  two-stage bootstrap with the same distribution as evaluate.boot_draws but its own random stream (the stream of
  step 18's former boot2, which step 18 now imports from inference.py). The frozen interval of each test stays the
  evaluate.boot_draws interval; the event-weighted "fixed best" interval of the inference block can differ from it
  in the third decimal (CHB SWORD-exact: [-0.159, +0.016] in the block vs [-0.159, +0.015] frozen).
- S3: the Peel-Clauset pools of step 23 use step 12's common mask (core rows); step 17's own re-report uses a mask
  over every step-17 row (GHRG and P&C baselines included). Both are reported.
- S4 (hygiene): step 16 held a copy of step 14's row scoring (pl_feats, per_segment, pl_scores) and check C3
  compared the copy with plants.json. The scoring now lives once in plants.py (feats, per_segment, score_all, SET,
  BURN), used by steps 14 and 16; C3 is retired because no copy is left (the shared scorer reproduces step 16's cached
  base series bit for bit on MIT and HT09, and step 14's rerun from the renamed caches is identical).
