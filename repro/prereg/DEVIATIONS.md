# Deviations from PROTOCOL.md

All entries below were decided BEFORE any detection score was computed on real data.

1. **HS dropped** (protocol rule): neither Mastrandrea et al. 2015 nor the dataset page gives clock
   times for breaks or lunch. Series is built but has no labels.
2. **PS labels**: only the 4 lunch start/end onsets have exact published times. The breaks are
   published only as "around 10.30am and 3.30pm", staggered by class.
   - Primary analysis: the 4 exact lunch labels only.
   - Secondary analysis (reported, not decisive): add the approximate breaks from
     `labels_breaks_approx.json`.
3. **Replacement dataset for HS: HT09** (SocioPatterns Hypertext 2009 conference contacts), labels
   = session / coffee-break / lunch boundaries from the official HT 2009 conference programme, only if
   exact clock times are published. Same snapshot rules as the school data (5-min bins, one segment
   per day, w=2, w_ref=4, delta=2, H=4). Decision rule (a) then reads "on >= 2 of {PS, HT09, OS}".
   If HT09 also has no exact programme times, it is dropped and rule (a) requires 2 of 2 {PS, OS}.
4. Because PS has only 4 events, the bootstrap CI for PS will be wide; the decision rule is applied
   as written, and the CI width is reported next to it.
5. **HT09 segment = recorded hours** (decided while building HT09, before any score). The HT09 raw file
   holds 7 sparse night-time records (29 Jun 21:45-23:57, 1 Jul 00:43-01:50). The PS/HS rule "segment =
   first to last contact of the local day" would add about 7 h of empty bins to 1 Jul. For HT09, each day's
   records are split at gaps >= 30 min and the largest block is the segment; the 7 rows are dropped. The
   rule uses timestamps only (no labels, no counts) and is a no-op on PS/HS (largest within-day gap 20 s).
   HT09 labels: 22 exact programme times (ht2009.org/program.php, Wayback 2009-06-19), all type M; the
   5-min "Short break" 18:05-18:10 is not labelled. See `data/HT09/README.md`.
6. **Three more datasets added (user request, 2026-10-12, before any real-data score)**:
   - **SC** Stablecoin ERC20 (SNAP / TGB tgbl-coin), daily graphs 2022-04-01..2022-11-01, labels =
     the event file shipped with the dataset. Windows as OS: w=3, w_ref=7, delta=3, H=7.
   - **PM** PAMAP2 IMU correlation networks, graph recipe = the one used by s-GNN
     (Sulem et al., Machine Learning 2023; code dsulem/DyNNet), labels = activity changes.
     Windows/tolerance: the recipe's snapshot unit; w=3, w_ref=7, delta=5, H=10 (snapshot units).
   - **EEG** CHB-MIT scalp EEG, subjects chb01, chb03, chb05 (fixed in advance), seizure files only.
     Graph recipe fixed in advance: channels = nodes, 2 s non-overlapping windows, Pearson
     correlation, edge iff |r| >= 0.5 (absolute threshold, so edge count CAN change; a fixed-density
     rule would hide volume by construction). Labels = annotated seizure onsets.
     w=3, w_ref=15, delta=5, H=15 (2 s units).
   With more datasets, rule (a) "on >= 2 datasets" would get easier. New rule (a): SG beats V
   (same margins) on a STRICT MAJORITY of the labeled datasets that have >= 4 events.
   Rule (b) unchanged. Legacy datasets (MIT, AskUbuntu, Enron) are reported but not decisive.
7. **Per-event ADD horizon is truncated at the next event** (decided before any score). HT09 events are
   3 bins apart while H=4, so horizons overlap. Per event, the horizon is H_e = min(H, next_onset - tau - 1)
   within the same segment; a miss scores H_e + 1. The alpha-calibration region (steps outside all
   [tau, tau+H] windows) is unchanged. Applies to every dataset (no effect where events are far apart).
8. **OpenSky choices** (decided before any score):
   - Onset dates use the UTC day on which the measure took effect, because snapshots are UTC days.
     US Schengen ban = 2020-03-14 (03:59 UTC), so it merges with the 03-17 EU closure (merge rule).
     EU-Russia closures = 2022-02-27 (most national closures), so they merge with 2022-02-24.
     A merged label whose parts have different types gets type M.
   - Missing day 2021-11-23 (no source data): the day is REMOVED from the sequence (not stored as an
     empty graph); later indices and labels shift by one. Partial days next to the gap stay in and are
     flagged in the results. This applies to any missing snapshot in any dataset.
   - Known threat, reported not corrected: the share of flights with both airports known rises from
     57% (Dec 2019) to 67% (Apr 2020), so part of any edge-count change is coverage. A sensitivity row
     with edges restricted to airport pairs present in >= 50% of the first 30 days is reported.
9. **EEG duplicate channel** (decided before any score): `P7-T7` is `T7-P7` with the opposite sign
   (r = -1.0 in every window), so edge (T7-P7, P7-T7) is present in all windows. The recipe's intent
   was to remove duplicate channels; a sign-flipped derivation is a duplicate. `P7-T7` is removed
   (21 nodes). Rule for any dataset: drop a channel whose |r| with an earlier channel is >= 0.999 over
   the whole file.
10. **PM splice confound** (decided before any score): every PAMAP2 label falls exactly at a join of two
    recording parts (s-GNN drops windows with label 0). A detector can fire on the join, not the activity.
    Placebo control: inside single-activity bouts longer than 2*(w+w_ref+H) snapshots, drop a block of
    snapshots whose length matches the median dropped "other" block, creating joins with NO activity
    change (one placebo per eligible bout, positions fixed by a seeded rule before scoring). For each
    channel, the alarm rate at placebo joins (any exceedance in [tau, tau+delta] at the alpha=0.05
    threshold) is reported. PM counts as DECISIVE only if, for both SG and V, the placebo alarm rate is
    <= 2*alpha*(delta+1); otherwise PM is secondary (reported as the s-GNN comparison only).
11. **SC labels and node set**: events whose date an external source contradicts are removed
    (FTX/Binance events dated Aug-Sep 2022 that happened in Nov 2022; "USDC v2.0" from 2020). Every
    other shipped event is kept, including off-chain news. Node set = all 6,715,034 addresses of the
    full period (the scale test; isolated-node dilution is part of what is tested). SC full build is
    deferred until RAM is free; only the 2022-05-01..05-20 slice exists now.
12. **Builder follow-ups to items 10-11** (before any score): SC also drops "Binance CEO mentions selling
    500million in FTX holdings" (dated 2022-06-11; true date 2022-11-06), under the item-11 rule; SC = 19
    labels (V 4, S 7, M 8). PM placebo eligibility is measured after the block removal (>= 20 snapshots
    on each side of the join): 67 placebos. The placebo alarm rate uses the alpha=0.05 threshold
    calibrated on series_placebo.pkl itself (same calibration rule, real activity labels as tau).
13. **PM placebo block length (audit finding H1, before any score)**: the median dropped block L_s is taken
    only over gaps that contain at least one label-0 ("other") sample. One-window mixed-label transitions
    (22 of 98 joins under this rule; a 23rd one-window gap, subject 107, holds 7 label-0 rows and counts as a long splice) are continuous recording and are excluded from L_s. They are still real labels.
    The placebo result is also reported separately for the long-splice labels and the one-window labels.
    PM correlation uses min_periods=30 (pairwise-complete r with fewer valid rows -> no edge).
14. **SC label corrections and status (label audit, before any score)**:
    - Rule: an SC onset is the earliest externally verified date on which the event became public
      (on-chain governance votes are public at the vote date; price moves at the date of the move).
    - Corrections per LABEL_AUDIT.md: UST passes BUSD -> 2022-04-18; FTX LUNC/USTC suspension ->
      2022-09-12 (merges into the 2022-09-09 label); MakerDAO $500M -> 2022-10-05 (vote); the "USTC +35%"
      member -> its own V label on 2022-10-10; the 2022-10-27 event -> 2022-10-28. Merge rule re-applied.
    - SC is made NON-DECISIVE: 16 of 38 shipped events are off-chain news with no stated on-chain action,
      several are reports or price descriptions, and two head events have no dated source. SC is reported as
      the scale test and as a secondary result. Decisive set = {PS, HT09, OS, EEG} plus PM if its placebo
      gate passes; rule (a) = strict majority of that set.
    - PS 12:00 lunch onset is nominal (two canteen turns, no published turn times); kept, noted.
15. **Contact-data segment edges (build.py audit, before any score)**:
    - PS is clipped to the published school hours 08:30-16:30 local (PROTOCOL "school hours only").
    - All contact datasets (PS, HS, HT09): a segment holds only COMPLETE 5-min bins inside the recorded
      span: start = first 5-min boundary >= first contact time, end = last 5-min boundary <= last contact
      time + 20 s. Partial leading/trailing bins are dropped. Label indices are recomputed from times.
    - The PS time origin (local midnight, CEST) is inferred from the published collection hours (+-5 min);
      not stated by the dataset page. Recorded as a known uncertainty.
    - Builders must fail if a label falls inside the w+w_ref burn-in.
16. **EEG flat channels and dropout (build_eeg.py audit C1/M1, before any score)**:
    - Bug fix, not a recipe change: a channel that is exactly constant within a window (np.ptp == 0 on the
      raw values) has undefined r -> no edge. Before the fix, float residue gave r = +-1 between flat
      channels (chb03_01 window 407: 105 of 108 edges were artifacts).
    - Windows in which >= 1 channel holds one constant value for >= 256 samples (recording dropout,
      e.g. chb03_01 windows 404-410) are flagged in build_meta.json; the recipe is unchanged, and a
      sensitivity row that excludes flagged windows from scoring is reported.
17. **OS observability and confounds (OpenSky audit H1/H2/M1, before any score)**:
    - Observability rule (uses raw traffic of the entities the event names, never detection features): a
      label is observable iff the airports/routes the event affects carry >= 1% of the dataset's edges in
      the 7 days before onset. Wuhan lockdown (ZHHH: 0 flights in the data; OpenSky has almost no coverage
      in China) and US-China entry ban (1-6 of ~15,000 edges/day) are NOT observable and are excluded from
      the primary OS analysis; a sensitivity row with all 5 labels is reported.
    - Omicron (2021-11-26) is kept but flagged: US Thanksgiving (11-25/26) is an unlabeled volume shock
      (US-linked flights -38%, EU flat). Sensitivity row without this label.
    - OS then has 3 primary labels (< 4), so OS is NOT decisive; it is reported. Decisive set =
      {PS, HT09, EEG} plus PM if its placebo gate passes; rule (a) = strict majority of that set.
    - Days with a source outage inside the day (an all-zero hour block in lastseen counts, e.g. 2020-01-18)
      get a `partial` flag, and they and the day before them are flagged in results.
18. **Boundary partial snapshots (SC audit H1, before any score), all daily datasets (OS, SC)**: the first
    and the last day of each segment are removed if the raw data covers less than 23 h of that UTC day
    (SC: 2022-11-01 ends 04:53 UTC; 2022-04-01 starts 11:10 UTC). They are marked `missing` and handled
    as in item 8. SC is built from ONE vocabulary of all addresses in the full period (slice ids = full ids).
19. **Gates and legacy windows (main-scorer report, before any score)**:
    - Large-snapshot gate (n_active > 8000, dense eigvalsh impossible): exact moments must agree with a
      Hutchinson estimate (R = 2000 Rademacher probes, |diff| <= 4 standard errors) and G must agree between
      ARPACK eigsh and an independent solver (LOBPCG or eigsh on a shift-inverted/different-ncv run) to
      1e-8, on 5 snapshots per large dataset.
    - Planted-change gate: V must fire on the volume change and not on the rewire; G must fire on the
      moderate rewire; S and G must fire on the strong rewire (S is known to be weak on degree-preserving
      sparse partitions, matching theory).
    - Legacy datasets (not decisive), windows fixed now from the paper's tuned SWORD settings:
      MIT w=7, w_ref=7, delta=5, H=10; AskUbuntu w=2, w_ref=5, delta=2, H=4; Enron w=2, w_ref=2, delta=4,
      H=8. Legacy labels get type U. These rows are reported only as a comparison with the paper.
20. **OS observability: the written rule is binding** (before any score). The measured edge shares are
    Wuhan 0.000%, US-China 0.017%, Schengen+EU 9.690%, Omicron 0.063%, Ukraine+Russia 0.788%. Under the 1%
    rule only Schengen+EU is observable, so the OS primary analysis has 1 label; the item-17 judgement that
    kept Omicron and Ukraine+Russia is withdrawn. Sensitivity row "OS all labels" reports all 5. OS stays
    non-decisive. SC: the 2022-10-10 USTC label merges with the two 2022-10-13 events under the merge rule
    (type M); SC uses the full build data/SC/series.pkl.
21. **Binding counts after the builder fixes** (before any score): PM placebos = 66 (item 12's 67 is
    superseded by the item-13 L_s rule); PM real labels 98 (76 long splices, 22 one-window); OS primary
    (observable) labels = 1 (item 20); SC labels = 18; PS 4 + 4 approximate; HT09 22; EEG 19. The runner must
    assert these counts before scoring and fail on a mismatch.
22. **Cross-check and re-audit decisions (2026-10-13, before any decisive score)**:
    - "OS without Omicron" (INTERPRETATIONS 20) = ALL OS labels minus the Omicron label (4 raw labels, no
      observable filter). The earlier reading (observable minus Omicron) was identical to the primary row.
    - Label order (INTERPRETATIONS 31): merge first, then filter on `observable`.
    - PM one-window split (item 13): per-event metrics on the main PM series, split by the label flag
      (INTERPRETATIONS 34).
    - Rule (b) has a float tolerance of 1e-12 (INTERPRETATIONS 33); "exactly 0.05 worse" passes.
23. **Slice-run exposure (disclosed, 2026-10-13)**: smoke runs with `--max-snapshots 30` (main, 01:23-01:25Z)
    and 15 (indep, 01:02-01:03Z) scored the REAL labels that fall inside the slices. Exposure by event index:
    PS, HT09, EEG, PM, OS (main and stable pairs) and the PM placebo set: 0 events, 0 placebo rates (no metric
    has content). SC (not decisive): main 2 events (segment 0, indices 10 and 16, type V), indep 1 event
    (index 10). No decision, label, window or parameter was changed after these runs. The files were moved to
    the scratchpad quarantine and are not results. From now on slice runs use no real labels (INTERPRETATIONS 35).
24. **Feature numerics after the second audit (2026-10-13, before any decisive score; FEATURE_VERSION v5)**:
    exact traces use pairwise summation (`np.sum(x*x)`, was BLAS `ddot`; SC S mu_4 error 2.7e-10 -> 4e-16
    relative to `math.fsum`). G for n_active > 500 is computed per connected component (dense eigvalsh for
    components <= 500 nodes, ARPACK/LOBPCG for larger ones; LOBPCG residual > 1e-8 is an error), because ARPACK
    can drop copies of an eigenvalue repeated across components. The G gate oracle is the SVD of I + S_tau
    (shares no LAPACK path with production). Every decisive dataset (n_active <= 500) keeps the dense path,
    bit-identical; on all 509 real sparse-path snapshots the old and new G agree to <= 9e-15.
25. **FEATURE_VERSION v6 (third audit, 2026-10-13, before any decisive score)**: item 24 is too narrow; ARPACK
    can also drop copies of an eigenvalue repeated INSIDE one connected component. Every sparse G computation
    now ends with a multiplicity guard (deflation: found eigenvalues shifted below the spectrum, search again
    until the new top value is not above the 7th value + 1e-12; residual <= 1e-8; at most 20 rounds, else an
    error). The large-snapshot gate reference is the guarded routine on the whole S_tau (ARPACK or block-12
    LOBPCG with the same deflation check). Dense path (n_active <= 500, all decisive datasets): bit-identical.
    OS (317 sparse snapshots): max error vs dense 4.4e-15; SC (7 days) v5 vs v6 <= 1.3e-15. A gate failure on
    any dataset still aborts the run (PROTOCOL 6); a failure of a non-decisive dataset after the gates is
    recorded in decision.json (`failed_secondary`) and does not block the decision.
