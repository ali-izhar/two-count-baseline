# Canonical interpretations (binding for both implementations; fixed 2026-10-13, before any real score)

Indices are 0-based within a segment. "Scored step" = a step that has a finite score for that channel.

1. **Window alignment (inclusive).** At step t: test = {t-w+1..t}, ref = {t-w-w_ref+1..t-w}.
   Steps 0..w+w_ref-1 have no score (protocol: first w+w_ref steps). First scored step = w+w_ref.
2. **Moments.** mu_k = (1/N) tr T_k(L - I), L = I - D^{-1/2} A D^{-1/2}, isolated nodes have L_ii = 1.
   N = fixed node count (SW, SWraw) or n_active (S). Self-loops dropped, duplicate edges merged.
3. **Undefined features.** S is NaN when n_active < 2. G is NaN (all 6 values) when n_active < 8.
   Empty snapshot: V = (0, 0); S, G NaN; SW defined (all nodes isolated).
   Window means use nanmean. A feature whose test or ref window has no finite value is left out of the
   mean over features at that t. If all features are left out, the step has no score.
4. **Scale.** scale_j = 1.4826 * MAD of first differences of feature j, differences taken within each
   segment over consecutive finite values, pooled over segments. scale_j == 0 -> feature j dropped (logged).
5. **Thresholds.** Every alarm rule is `score >= thr`.
   - Fixed config: thr = numpy.percentile(scored steps pooled over segments, 90, method='linear').
   - Matched alpha: calibration set = scored steps outside every [tau, tau+H] (H unclipped).
     thr = the smallest value v among calibration scores with mean(cal >= v) <= alpha.
     If no such v exists, thr = +inf (the channel never alarms; logged).
6. **Combined channels (SG, SW2).** Each member's D is divided by its median over its scored steps
   (pooled over segments). Combined score at t = max over the members that have a score at t.
7. **Fixed-config detections.** Scan each segment in time order from its first scored step; alarm at t if
   score >= thr and t - last_alarm >= cooldown (cooldown = delta + 1); last_alarm resets per segment.
8. **F1 matching.** One-sided window [tau, tau+delta]. Process detections in time order; each detection
   matches the earliest unmatched event (same segment) whose window contains it; each event matches at most
   once; unmatched detections are FPs. P = TP/#detections, R = TP/#events, F1 = 2PR/(P+R); 0 if no TP.
9. **Penalized ADD.** H_e = min(H, next_onset - tau - 1, last_step_of_segment - tau) (next_onset = next
   label in the same segment). delay = first t in [tau, tau+H_e] with score >= thr (alpha threshold) minus
   tau; miss -> H_e + 1. recall@alpha = share of events with a hit. Report mean penalized ADD.
10. **Exceedance.** peak_e = max score over scored steps in [tau, tau+delta]; non-tolerance set = scored
    steps outside every [tau', tau'+delta]; exceedance_e = mean(nontol >= peak_e). Event with no scored
    step in its window -> exceedance 1.0. Random expectation per event = 1/(m_e + 1), m_e = number of scored
    steps in its window; report the mean.
11. **Labels.** Missing snapshots are removed first; a label on a removed snapshot moves to the next kept
    snapshot. Then the merge rule (events closer than delta+1 steps merge into the earliest; mixed types
    -> M) is re-applied on final indices; any new merge is logged.
12. **Bootstrap.** Paired resampling of events (with replacement) within a dataset, 2000 replicates,
    numpy default_rng(0), 95% percentile CI, for per-event metrics only (penalized ADD, recall@alpha,
    exceedance) and their SG - V differences. F1 is a point estimate.
13. **Constant channel C.** Score 1.0 at every step that the w+w_ref burn-in leaves scorable.
14. **SWraw.** Paper statistic on full-node-set mu_1..mu_4 (raw, no MAD scaling): mean over
    test x ref pairs of the L1 distance, with the alignment of item 1.
15. **Decision.** Computed in code from PROTOCOL section 5 as amended by DEVIATIONS items 6 and 10
    (PM decisive only if the placebo condition holds; datasets with < 4 events are not counted).
    Legacy MIT / AskUbuntu / Enron: reported, not decisive.
16. **Bootstrap draw.** Per dataset: idx = numpy.random.default_rng(0).integers(0, n_events, size=(2000, n_events)),
    shared by all channels (paired).
17. **Win margins.** ADD win iff ADD_SG <= 0.8 * ADD_V. F1 win iff F1_SG - F1_V >= 0.10 - 1e-12.
    Rule (b) is checked on the decisive datasets.
18. **Placebo (PM).** Placebo labels are never merged and never act as tau; the alpha calibration set is
    defined by the real labels only (placebo steps stay in it).
19. **Combined-channel median 0.** That member is not rescaled; logged.
20. **Label fields** (written by the builders): `observable` (bool, default true), `flags` (list of str, e.g.
    "confound:thanksgiving", "nominal_time"). Primary analysis uses observable labels only. Sensitivity rows:
    OS all labels; OS without Omicron; EEG without dropout-flagged windows; PS with approximate breaks.
    Segment-level `partial` (bool list) and `dropout` (bool list) flags are reported, never used to drop
    steps except in the named sensitivity row.
21. **Decisive set** = {PS, HT09, EEG} plus PM iff its placebo gate passes (DEVIATIONS 14, 17).
    SC, OS, HS and the legacy datasets are reported, not decisive.
22. **Sensitivity selectors.** Omicron = the OS label whose flags contain "confound:us_thanksgiving_2021"
    (never matched by text). EEG dropout row: features and windows unchanged; steps flagged `dropout` get no
    score (removed from detections, calibration and exceedance sets). PM long-splice vs one-window split uses
    the label flag "one_window_transition". The ">= 4 events" rule also applies inside the decisive set.
23. **ADD margin tolerance.** ADD win iff add_v > 0 and ADD_SG <= 0.8 * ADD_V + 1e-12 (exact ties count as
    wins; both 0 is not a win). Same in both implementations.
24. **Rule (b) scope.** Rule (b) is checked on EVERY dataset in the decisive set (PS, HT09, EEG, and PM if
    its gate passes), regardless of event count (PROTOCOL "any dataset"; DEVIATIONS 6 left (b) unchanged).
25. **Degenerate cases.** A channel with no scored step: threshold +inf, no detections, logged.
    Exceedance with an empty non-tolerance set: 0.0, logged. +-inf feature values are treated like NaN.
    Empty decisive set: decision "UNDEFINED" in both implementations.
26. **Merge provenance.** `types_before_merge` keeps the list of all original member types.
27. **OS stable-pair row (DEVIATIONS 8).** Applies to segment 0 only (the 2019-12..2020-04 coverage threat);
    reference = the first 30 kept snapshots of segment 0 (no label inside). Segment 1 is unrestricted in this row.
28. **Label on a removed final snapshot** (no next kept snapshot): the label is dropped and logged.
29. **Feature cache validity** requires equal series sha256, FEATURE_VERSION and code sha256 (features.py,
    datasets.py, gonogo_config.py). The series file must not change during extraction.
30. **Merged-label fields.** flags = sorted union of member flags; observable = True iff any member is
    observable; types_before_merge = list of member types (item 26).
31. **Label pipeline order** (DEVIATIONS 22). For every label set: (1) take the union of the raw label rows of
    the set; (2) remove missing snapshots and move labels (item 11, 28); (3) merge once (item 11, fields per
    item 30); (4) apply the observable filter (keep merged events with >= 1 observable member) except in the
    rows defined on all labels. Label sets: primary = labels.json; PS approximate-breaks row = labels.json
    union approximate labels (observable filter applied); OS all labels and OS without Omicron (item 32): no
    observable filter.
32. **OS without Omicron** = all OS labels except the label selected by item 22; no observable filter.
33. **Rule (b) tolerance.** SW2 passes on a dataset iff F1_SW2 >= F1_V - 0.05 - 1e-12.
34. **PM one-window split** (DEVIATIONS 13). Per-event metrics (penalized ADD and recall at alpha 0.05 and
    0.02, exceedance) on the main PM series, computed per event as in the primary run (same thresholds),
    then averaged separately over the events with and without the flag "one_window_transition". A placebo
    alarm-rate split may also be reported; it is not this row.
35. **Slice runs** (`--max-snapshots`) are engineering checks only: gates, features and label-free scores.
    They map no real labels and compute no metric on real labels. A slice length >= the length of any
    requested series is rejected.
