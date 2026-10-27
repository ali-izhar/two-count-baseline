# Turnover row: spec (written 2026-10-27, before any run)

Question: are the leads of the identity-aware rows ASE and Hamming over the (causal) counts on the >= 10-event pool
matched by vertex turnover alone (which vertices are active), an activity-only quantity that uses vertex identity?

Row "turnover" (methods/turnover.py): a_t = 0/1 vector of the active vertices of snapshot t (degree > 0 without
self-loops) on the fixed vertex set of n vertices (the pair keys of step 16; AskUbuntu: the raw user-id mapping the
identity rows use, n = all user ids). Score at t: || mean(a over the w = 2 snapshots before t) - mean(a over the
w_ref = 5 snapshots before those) ||_1 / n. Same windows as the counts, scores from w + w_ref (the burn-in of the
common mask, 17 steps, applies), every segment restarts. One setting, no tuning. Scale-free (a share of n), so no
separate causal version exists: the score at t uses only snapshots before t.

Evaluation: EP per event on the common mask of step 12 (evaluate.ep); EP per dataset; pooled EP (equal weight) on all
ten and on the >= 10-event pool with event-level and two-stage intervals (inference.pooled_draws, B = 10,000, seed
20261023); paired two-stage differences ASE - turnover and Hamming - turnover on the >= 10-event pool and per dataset
(event-level bootstrap within the dataset) on HT09, SCC (stablecoin), PM (PAMAP2), EEG.
In the inference families of step 23 it is a baseline row like the causal counts: it gets intervals and appears in the
rank sets, but it is not a member of the Holm family of detectors.

Calibration checks: (1) on a series whose snapshots are identical the score is 0; (2) a relabelling of all vertices
by one permutation applied to every snapshot leaves the score unchanged; (3) the score is in [0, 1].

Reading rule (fixed now): if the ASE and the Hamming leads over the causal counts on the >= 10-event pool are not
significant against turnover (the two-stage interval of row - turnover covers 0), the paper says the lead is matched by
vertex turnover alone (activity); otherwise it says the lead is not explained by turnover and its mechanism is unknown.

## Release note and deviation (added 2026-10-27, after the runs)

This file is the spec as written before the first run of the turnover row, released unchanged above this section.
Step 16 computes the row (`methods/turnover.py`), step 23 part T applies the reading rule (`results/inference.json`
`turnover`; with the stablecoin audit labels and without the stablecoin series: `labels.scc_audit.turnover`,
`labels.no_scc.turnover`).
D1. The spec says the row "appears in the rank sets". The code of step 23 keeps the turnover out of the rank sets and
    the best sets (`UNRANKED`), so P(rank 1) and the rows not excluded from being best are computed without it. The
    reading rule does not use ranks, so the reading is unaffected.
