# Round 3 fixer: analysis spec (written 2026-10-26 before any of these analyses ran)

Concerns: the second-round reports of R1-R5 (fresh blind reviewers). Every analysis below is reported whichever
way it goes. Nothing here changes an existing score series, mask or EP value; existing numbers stay as they are.

## A. Step 23 (benchmark inference), new parts

A1 Pools (R1, R2, R5). `source4` = the four scored datasets with the source paper's construction and events
   (UCI, MAG, SCC, PM); `own_added6` = the other six (MIT, ENRON, PS, HT09, EEG, AskUbuntu); `no_detector_labels`
   = without the datasets whose events are the source detectors' own detections (of the ten: MAG; the flights and
   the Canadian bills are not scored); `10plus_noEEG` = HT09, PM, SCC (FRZ-12 and IG-S were developed on EEG).
   Same blocks as the existing pools (three interval levels, Holm F1 / F2, ranks).
A2 Causal count row `counts_causal` (R1, R2, R5): the count detector with the online scale (each count over the
   median |first difference| of the first 20 steps of its segment), same windows and weights; series written by
   step 22 (section E) to `repro/data/settings/scores_<ID>.npz`. It enters every pool as a row; it is not in F1/F2
   and not in the rank set of the existing 18 rows (so the existing P(rank 1) values do not change); a second rank
   set replaces counts by counts_causal. Contrasts `counts_causal - row` with the same intervals and Holm families.
A3 Small-sample dataset-level inference (R3): for counts - row and counts_causal - row over the D datasets of a
   pool: t_{D-1} interval of the mean of the per-dataset mean differences; exact sign-flip test over datasets
   (all 2^D patterns), p and inverted interval. Reading: the "dataset" statements use the t or sign-flip result
   when it differs from the bootstrap.
A4 Set of rows not excluded from being best (R3; a bootstrap form of Hsu's multiple comparisons with the best):
   for every ordered pair (r, j) the studentized centred bootstrap contrast; c = 95% quantile over draws of the
   maximum over all ordered pairs; row r is excluded if some j has (EP_j - EP_r) - c se_{jr} > 0. Levels: event and
   two-stage; pools all, 10plus, interaction, source4. Rows: the 18 rows of the rank set (constant excluded), plus
   the same with counts_causal in place of counts.
A5 Coverage of the pooled interval (R3). Super-population simulation on the all-ten pool, contrast counts -
   SWORD-exact: dataset effects mu_d = per-dataset mean difference, centred to mean 0, so the population mean is 0;
   each replicate draws D = 10 datasets with replacement from the ten (each keeps its own event count and its
   centred per-event deviations), and events within each drawn dataset; coverage of 0 by (i) the two-stage
   percentile interval (B = 1,000), (ii) the t_9 interval of the per-dataset means, (iii) the event-level interval
   (conditional; expected to undercover the population mean). 2,000 replicates. Reading: report the coverage.
A6 Detection at fixed false-alarm quantiles for every row (R2, R3, R5): evaluate.delay as in part F, alpha 0.01,
   0.02, 0.05, for every row including identity, wiring and counts_causal; per-event hit within tol as 0/1; pooled
   equal weight over the datasets with null steps; paired event-level and two-stage intervals; counts - row and
   counts_causal - row.
A7 Protocol variants for every row with intervals (R1, R2, R3): (a) guard band of 17 steps after each event
   removed from the negatives; (b) one alignment: past-only rows (SWORD, SWORD-z, counts, counts_causal, MMD, the
   six identity rows, the three wiring rows) scored with event windows [e + 1, e + 1 + tau] and the same null
   windows rule (events at e + 1), the rows that use G_t (CUSUM, BOCPD, LAD, SCPD) and TIRE (offline) as is;
   (c) negatives ranked within the event's own segment (PM, EEG; the others have one segment); (d) event-date
   jitter: every event moved by -2, -1, +1, +2 steps for every row; (e) the identity-aware rows on AskUbuntu
   removed (vertex identity there comes from the user ids of the raw data; reported as a sensitivity).
   For (a)-(c): pooled EP, event and two-stage intervals, counts - row, P(counts rank 1); for (d), (e): pooled EP.
A8 Per-case EP of the benchmark EEG (R4): the per-event EP of every row averaged within each CHB-MIT case (the
   segments map to cases through the step-10 metadata).

## B. Step 19 (activity and wiring), new parts

B1 Holm over the rows of the matched test (R1, R3, R5): two-sided bootstrap p of matched EP - 0.5 for every row
   (event-level paired draws, B = 10,000), Holm over the 16 rows of Fig. 1(e) (constant, counts and the proxies
   excluded), in the interaction and the correlation group separately.
B2 Calibration of the matched EP of every row at pseudo-events (R5): 200 cyclic shifts of the event set per dataset
   (windows clear of every real event), the same matching on the count score, the matched EP of every row averaged
   over the pseudo-events. Expected 0.5 for every row if the matching is unbiased. Per-row permutation p = (1 +
   #{pseudo pooled >= observed}) / (1 + 200), two-sided version, from the same shifts (the pooled interaction
   value of each shift). Reading: a row whose pseudo-event mean differs from 0.5 by more than 0.03 makes the
   matched test biased for that row; its observed value is then read against its pseudo-event distribution.
B3 Matching including G_t (R2, R3, R5): (i) `B_Gt`: match on the count score shifted so that the score at t
   includes G_t; (ii) `B_both`: match jointly on the two (past-only and including G_t) with the caliper rule of B.
   Calibration as B (constant, noise, counts, counts_Gt, positive control). Each row is read under the matching
   that fits its alignment (`B_own`: B for past-only rows, B_Gt for CUSUM, BOCPD, LAD, SCPD, TIRE) and under
   B_both. If a calibration fails, the design is reported as failed.
B4 Power of the matched test (R2, R3): SE = (hi - lo) / 3.92 of each row's interval; MDE at 80% power = 2.80 SE;
   power to detect matched EP 0.60.
B5 Power of the per-event rule (R1, R3, R5): on each dataset, at 200 sets of cyclically shifted pseudo-events, a
   level shift of size delta robust SDs of the first differences (delta = 0.5, 1, 2, 4, 8; the robust SD used
   by A) is added to n_+ (activity) or to assortativity (wiring) from the pseudo-event to the end of its segment;
   the per-event rule of A is applied; the share flagged is the power. Aggregate per-dataset test: for each
   dataset and statistic family, the mean over events of the family maximum of the per-event EP of |Delta stat|,
   p against 1,000 cyclic shifts; Holm over datasets.

## C. Step 22 (settings), new section J

J  Identity-aware rows under the setting rules (R1, R3): Hamming, Jaccard, NBS and FRZ on the (w, w_ref) grid
   W x WR of common.py (16 settings) from step 16's key caches; EP at the Table 2 setting, grid median / IQR,
   oracle, leave-one-dataset-out selection by EP, pooled all ten and >= 10 events. Also tolerance tau = 0..10,
   own-span tolerance and guard bands for these rows (read from step 16's score series).

## D. Plants (step 14) and synthetic scenarios (step 16)

D1 SCPD on the plants (R2): SCPD's own signature (rows.scpd_signature, its windows (5, 10), its first-difference
   score) on the planted series at rho 0.5 (rand, assort, triclose, swap) and the activity controls at 0.3,
   seeds 0-9, every background; paired AUC with the block bootstrap of the other rows.
D2 LAD under relabelling (R3): check whether LAD's score is exactly invariant to the isomorphic relabelling; if not,
   find the cause (solver start vector, ties) and report.
D3 Synthetic scenarios (R3): the number of replicates and events per scenario and the resampling unit of the
   interval, read from step 16.

## E. EEG (steps 18, 20, 21)

E1 Within-case ranking with ictal and post-ictal windows removed from the negatives (R4), all three sealed sets.
E2 Line length (Esteller et al. 2001) as a training-free post hoc baseline on the development and the three sealed
   sets: per 2 s window the mean over channels of sum |x_{i+1} - x_i|; score = the two-window change of its log
   (w = 3, w_ref = 15, as l2_bands), plus the level itself; EP with the sealed protocol.
E3 Power under the recomputed-max rule (R4): draw the two-stage resamples of step 21; in each draw the candidate's
   EP is shifted so that its lead over the fixed best equals the development lead; pass if the lead over the
   maximum of the baselines in that draw exceeds the half-width of the observed recomputed-max interval; power =
   share of passing draws (an approximation; labelled as such).
E4 Facts (R4): hours analysed and files per set, Siena version and sampling rate handling (from step 20).
