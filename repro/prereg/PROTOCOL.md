# SWORD go/no-go protocol (pre-registered 2026-10-12, before any data was built)

Question: does a NON-VOLUME channel (structure / global spectrum) beat a VOLUME
detector on datasets with externally dated structural events?
GO -> method paper (SWORD-2). NO-GO -> audit + theory paper.

Nothing below may change after scores are seen. Any deviation must be logged in
`DEVIATIONS.md` with the reason, and the original-protocol result must still be reported.

## 1. Datasets and labels (labels from external sources only; never from the series)

All snapshots are undirected, unweighted, simple graphs (no self-loops, no multi-edges).
Node set per dataset = every node that appears anywhere in the dataset (fixed n).

| ID | Data | Snapshot | Labels (event onsets) | Source of labels |
|---|---|---|---|---|
| PS | SocioPatterns Primary School (2 days) | 5-min bins, school hours only | break start/end, lunch start/end, each day | Stehle et al. PLoS ONE 2011 (timetable) |
| HS | SocioPatterns High School 2013 (5 days) | 5-min bins, school hours only | breaks and lunch start/end per published schedule | Mastrandrea et al. PLoS ONE 2015 (or the dataset page); if no exact schedule is published, HS is DROPPED (not labeled by eye) |
| OS | OpenSky flight list (Zenodo 7923702) | daily airport graph, edge = >=1 flight between two airports that UTC day | 2020-01-23 Wuhan lockdown; 2020-02-02 US entry ban on China (announced 01-31); 2020-03-13 US ban on Schengen travel (announced 03-11); 2020-03-17 EU external border closure; 2021-11-26 Omicron bans (southern Africa); 2022-02-24 Ukraine airspace closure; 2022-02-27/28 EU-Russia airspace closures | Official/press dates, each cited in labels file |

OS windows: 2019-12-01..2020-04-30 and 2021-11-01..2022-04-15 (two separate sequences,
scored together as one dataset). Events closer than delta+1 steps are merged to the earliest.
Every label also carries an a-priori type: V (volume shock), S (structural), or M (mixed),
assigned from the event description, not from the data.

School data: each day is a separate segment. Scores in the first w+w_ref bins of each day are
set to "no score" (burn-in), for every method.

## 2. Channels (features per snapshot)

All moments are EXACT closed forms (no Hutchinson). S = D^{-1/2} A D^{-1/2}.
- V   (volume): log(1+active nodes), log(1+edges).
- SW  (paper SWORD input): exact Chebyshev moments mu_2, mu_3, mu_4 of the normalized
       Laplacian on the FULL fixed node set (isolated nodes included). mu_1 = 0 (no self-loops).
- S   (structure): exact mu_2, mu_3, mu_4 on the ACTIVE subgraph (normalized by n_active).
- G   (global): eigenvalues 2..7 (descending) of S_tau = (D+tau I)^{-1/2} A (D+tau I)^{-1/2}
       on the active subgraph, tau = mean active degree. Sorted order statistics.
- SG  = S and G channels combined (no volume).
- SW2 (SWORD-2) = V, S, G combined.
- C   (constant score) - sanity row.
- SWraw: the paper's own statistic (mean-pairwise L1 of raw full-graph moments mu_1..mu_4, k=4) - reference row.

## 3. One detection statistic for every channel

Per feature j: scale_j = 1.4826 * MAD of first differences of feature j over the whole
series (label-free; same rule for every channel). z_t = f_t / scale.
Two-window statistic: D_t = mean_j | mean(z over test window) - mean(z over ref window) |,
test = last w steps, ref = the w_ref steps before them.
Combined channels (SG, SW2): each member channel's D is divided by its median over the series,
then the combined score is the max over members.

Windows (fixed per time scale, identical for all channels):
- Schools: w=2, w_ref=4, tolerance delta=2 bins, ADD horizon H=4 bins.
- OpenSky: w=3, w_ref=7, delta=3 days, ADD horizon H=7 days.
Matching is one-sided: detection in [tau, tau+delta] (as in the paper).

## 4. Metrics

1. Fixed-config F1: threshold = 90th percentile of the method's scores; cooldown = delta+1.
2. Penalized ADD at matched false-alarm rate alpha in {0.05 (primary), 0.02}: threshold set so that
   a fraction alpha of scored steps outside all [tau, tau+H] windows exceed it. Per event, delay =
   first exceedance in [tau, tau+H] minus tau, else H+1. Report mean, plus recall@alpha.
3. Exceedance (threshold-free): per event, peak score in [tau, tau+delta]; exceedance = share
   of scored steps outside all tolerance windows with score >= that peak. Mean over events
   (lower is better). Random-score expectation reported alongside.
4. Uncertainty: 2000-sample bootstrap over events for every metric difference.
Results are broken down by label type (V / S / M).

## 5. Decision rule

GO (method paper) iff BOTH:
 (a) on >= 2 of the datasets (PS, HS, OS; 2 of 2 if HS is dropped), SG beats V by
     penalized ADD (alpha=0.05) >= 20% lower OR fixed-config F1 >= +0.10; and
 (b) SW2 is not worse than V by more than 0.05 F1 on any dataset.
Otherwise NO-GO (audit + theory paper).
Secondary (reported, not decisive): S vs V, G vs V, SW vs V, S/G performance on S-type labels only.

## 6. Correctness gates (must pass before the decisive run)

- Exact moments match dense eigvalsh on >= 50 random snapshots per dataset (max err < 1e-10).
- G eigenvalues match dense eigvalsh on >= 20 snapshots per dataset.
- Snapshot edge counts match an independent recount from raw files.
- An independent second implementation of the scoring + metrics (written from this file only)
  agrees with the main implementation on every reported number.
- code-auditor review of every script; all high/medium findings fixed or justified.
- Planted-change test: inject a known structural change (SBM rewire, fixed degree sequence) and a
  known volume change into a null stretch; G/S must fire on the first and V on the second.
