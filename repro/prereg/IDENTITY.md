# Identity-aware detectors: specification (written 2026-10-24, before any score of these detectors was computed)

Step `repro/16_identity.py`; outputs `results/identity.json` (benchmark B), `results/identity_plants.json`,
`results/identity_synth.json`; caches under `repro/data/identity/`.

## Questions

Q1. Does a detector that uses vertex identity beat the two counts on benchmark B (same masks, burn-in, EP and
    bootstrap as step 12)?
Q2. Do identity-aware detectors see the planted wiring changes of step 14 (same plants, seeds, onsets, trials)?
Q3. Is what they see the persistence break of the plant design (every planted snapshot rewired independently)?
    A persistent variant keeps edge persistence inside the planted block; every row is scored on both.
Q4. Do the detectors see block-model changes at matched expected degree, in the source papers' synthetic settings
    (LAD / SCPD SBM generator; s-GNN merge and swap generators)?

## Detectors (one setting each, fixed here)

All use the protocol's window rule: test window = the w snapshots before t, reference = the w_ref before it,
w = 2, w_ref = 10 (the default window of `evaluate.window_scores`); pair presence without self-loops.

* `hamming`: L1 distance of the window-mean adjacencies, sum_e |x_e / w - y_e / w_ref| (Hamming distance of edge
  sets for single snapshots; the two-sample CUSUM statistic at a fixed split up to a constant).
* `jaccard`: weighted Jaccard distance of the two window means.
* `deltacon`: DeltaCon (Koutra, Vogelstein, Faloutsos, SDM 2013; Koutra et al., TKDD 2016) between the two window
  means: FaBP affinities S = [I + eps^2 D - eps A]^{-1} with eps = 1 / (1 + max degree) of each graph, the grouped
  version with g = 10 random vertex groups (fixed seed 0), root-Euclidean distance of sqrt(S E).
* `chenzhang`: graph-based two-sample statistic of Chen and Zhang (Ann. Statist. 2015) on the w + w_ref snapshots
  of the two windows: MST of the Hamming distances between snapshots, R = number of MST edges between the windows,
  Z = -(R - E R) / sd(R) under the permutation null (their Lemma), at the fixed split of the protocol.
* `nbs`: the CUSUM inner product of Network Binary Segmentation (Wang, Yu, Rinaldo, Ann. Statist. 2021) at the fixed
  split: two independent copies (snapshots at even and at odd time index), Y^c = sqrt(nL nR / (nL + nR)) (mean_R -
  mean_L) per pair for copy c, D_t = sum_e Y^A_e Y^B_e.

## Analyses

A. Benchmark B: every dataset of step 12; the mask of step 12 (`evaluate.common_mask` over its primary rows and
   burn-in); EP per event; pooled EP with equal weight per dataset and the bootstrap of `evaluate.boot_pooled`
   (seed 20261023, 10,000 draws), all ten datasets and the >= 10-event subset; paired contrasts row - counts;
   cyclic-shift p with the shifts of step 12 (same generator seeds).
B. Plants: the 14 cells of step 14 on PS, HT09, PM, MIT, seeds 0-9, offsets 0-39, blocks [t0, t0 + 15), positives
   max over [t0, t0 + 5), negatives the same on the original series (`plants.Metrics`). Independent variant = the
   plants of `plants.plant` regenerated with the generator of `plants.series_stats` (exact reproduction).
   Persistent variant (new): drawn at each onset t0 and carried through the block:
   - rand / assort / triclose: the swaps applied to the previous planted snapshot are re-applied in order where
     valid (both old edges present, both new edges absent), then fresh swaps under the plant's acceptance rule top
     the count up to round(rho |E_t| / 2) (degrees stay exact);
   - thin: one uniform per pair drawn at t0; a pair is deleted in every block snapshot where its uniform < rho;
   - deact: one vertex set (share rho of the active vertices of snapshot t0) is silent through the block;
   - swap: one relabelling drawn at t0 is applied to every block snapshot.
   Generator: default_rng([seed, mode index, round(1000 rho), t0]). Identity rows on all 14 cells and both variants;
   the permutation-invariant rows of step 14 (except TIRE) on the persistent variant at rho = 0.5 (rand, assort,
   triclose, swap).
C. Synthetic (EP, tolerance 2, burn-in 17, every row of benchmark B plus the identity rows):
   - `lad_sbm`: the SBM generator of the LAD / SCPD authors (SCPD repository, datasets/multi_SBM/SBM_generator.py,
     main()): N = 1000, 151 snapshots, change points at loop steps 15, 30, 60, 75, 90, 105, 135 (snapshot t + 1),
     inter 0.005, intra 0.03, increment 0.01, alpha = 1 (independent snapshots); alternate one-snapshot density
     events and changes of the block count 4 -> 10 -> 2 -> 4. Density is NOT matched. 10 seeds.
   - `lad_sbm_matched`: the same block sequence and change steps, no density events, (p_in, p_out) re-solved at
     every block count so that the expected degree equals that of the four-block start and p_in / p_out = 6.
     10 seeds.
   - `sgnn_merge`: s-GNN merge generator (DyNNet src/dysbm/merge.py defaults): 4 x 100 -> 2 x 200, p = 0.5,
     q = 0.2, T = 200, change uniform in [50, 150); density NOT matched. 20 seeds.
   - `sgnn_merge_matched`: the same with the post-change within-block p re-solved for equal expected degree. 20 seeds.
   - `sgnn_swap`: s-GNN swap generator (DyNNet src/dysbm/swap.py defaults): 4 x 100, p = 0.25, q = 0.2, permute rate
     0.2 (40 vertex pairs exchange blocks), T = 200: an identity-only change at equal density. 20 seeds.
   The graphs are drawn with numpy from the same distributions (not the authors' random streams).

## Calibration checks (a failure is reported as a failure)

C1. Step 12's mask is reproduced (n_mask per dataset) and counts' EP per dataset equals `results/bench.json`.
C2. Regenerated independent plants reproduce step 14's cached series statistics (edge counts, moved share,
    accepted swaps) for every series checked.
C3. The replica of step 14's scoring reproduces the AUC in `results/plants.json` on the independent variant
    (HT09 and MIT, rho = 0.5, rand / assort / triclose) to 1e-9.
C4. Chen-Zhang null: the mean and variance of R over 20,000 random labellings of 50 random 12-point MSTs agree with
    the formula within 2 % (relative).
C5. Persistent swap (isomorphic relabelling): every permutation-invariant row has AUC within 0.03 of 0.5.
C6. NBS on an i.i.d. Bernoulli stream without change has mean within 3 standard errors of 0.
C7. Matched synthetic scenarios: realised mean degree before and after each change within 2 %.
C8. The replica of step 11's per-snapshot features and rows reproduces `data/bench/cache` features and
    `scores_PS.npz` rows on PS to 1e-9.

## Reading rules (fixed now)

R1. Benchmark: a row beats (is below) the counts if the 95 % CI of the pooled contrast row - counts lies above
    (below) 0; otherwise "not distinguishable".
R2. Plants: a row sees a cell if AUC >= 0.65 (the P1 threshold of step 14). A row gains only from the persistence
    break in a structural mode if it sees the independent cell at rho = 0.5 on >= 2 of PS, HT09, PM but sees the
    persistent cell on none of them; the drop AUC(independent) - AUC(persistent) is reported per cell.
R3. Synthetic: a row sees a scenario if the lower 95 % bound of its EP (two-stage bootstrap: seeds, then events)
    exceeds 0.5.
R4. Claims: if an identity-aware row sees a persistent structural plant (R2) on >= 2 of PS, HT09, PM, or sees
    `lad_sbm_matched` or `sgnn_merge_matched` (R3), then "wiring changes are hardly seen" holds only for the
    permutation-invariant detectors and the title / abstract claims must say so.

## Deviations (logged after the runs, with reasons)

D1. Persistent structural plant. The rule written above (re-apply the previous swaps where valid, top up with fresh
    swaps) was tried on MIT before any other result: it kept only part of the persistence (consecutive edge Jaccard
    0.25 against 0.46 in the original). It was replaced, before any other background was scored, by swaps between
    edges with the same presence pattern over the 15-step block (`pattern_swaps`): degrees stay exact at every
    snapshot and each moved edge keeps the pattern of the edge it replaces. A first version demanded that a new pair
    be absent from the whole block; it gave no swap at all on most PM blocks, so the rule became "absent at every
    time of the pattern" and all persistent structural cells were recomputed. Consequence: the persistent plant
    moves fewer edges than the independent plant at the same rho (PS 0.22 vs 0.39, HT09 0.18 vs 0.40, MIT 0.10 vs
    0.38 at rand 0.5; median over blocks), and on PM (dense snapshots, about 200 of 351 pairs) 58 % of the blocks
    admit no swap, median moved share 0: the PM persistent structural cells are uninformative. The reading therefore
    also compares each persistent rho-0.5 cell with the independent cell of the nearest moved share
    (`reading.moved_matched`).
D2. C4 was run as written (Monte Carlo, 20,000 labellings): worst relative error 2.8 %, above the 2 % band, as the
    Monte Carlo error of a variance from 20,000 draws allows. The step replaces it by exact enumeration of all
    labellings (n1 = 2 and 5): worst absolute error 3.6e-15.
D3. The s-GNN swap generator's sequence version (DyNNet src/dysbm/swap.py, generate_sequence_memberships) draws no
    swap with its default sizes (its cluster lists are empty for every block after the first), so its series has no
    change. `sgnn_swap` follows its pair version: 40 pairs between blocks {0, 1} and {2, 3}.
D4. Benchmark keys of MIT, ENRON and AskUbuntu come from the vertex labels of data/cache/<ds>.pkl: the step-1
    matrices of AskUbuntu number each snapshot's vertices afresh, so they carry no identity. Edge counts per snapshot
    are asserted equal to the step-1 matrices.
D5. The first version of this spec also listed two rows of the authors' own design (one detector row and one
    reference contrast). They are not part of this study: the rows were removed from the code and from every result,
    and their lines were removed from this spec (2026-10-27). No other line of the spec changed.
