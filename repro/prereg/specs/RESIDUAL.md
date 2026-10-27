# Incremental-information test (fixed 2026-10-24, before any result)

Question: does any detector rank the labelled events above chance once everything that activity explains is
removed from its score?

## Activity basis
At each step t, with the count detector's windows (w = 2, w_ref = 5) AND SWORD's windows (w = 2, w_ref = 10):
a = window-mean change of n_+ (test minus reference), b = the same for |E|, each divided by the median absolute
first difference of its count over the series (the count detector's scale). Basis X_t = [1, a, b, |a|, |b|, a^2,
b^2, ab] for both window pairs (15 columns after one shared intercept). Segments: features restart at segment
starts, as every rule does.

## Residual score
For a detector score D on one dataset (or one subject's recordings for EEG), over the steps that the common mask
scores, fit D_t = X_t beta + e_t by ordinary least squares (in-sample, all scored steps, events included), and
use the residual e_t as the new score. The residual removes every component of D that is a linear function of
the activity basis (which contains the count detector's score as an exact function up to the gam norm, which is
approximated by the squares). Robustness variant (reported, not primary): ranks of D regressed on the basis
(rank-OLS).

## Metric
EP of the residual score with the same event and null windows, burn-in and pooling as Table 2 (equal weight per
dataset; 10,000 resamples of events within datasets; seed of evaluate.py). Sealed EEG sets: the same, per set,
with the two-stage bootstrap (subjects, then events) and the fit done per subject.

## Calibration (must hold, else the test is broken and reported as such)
* constant score: residual EP 0.5;
* count detector: residual EP within 0.05 of 0.5 pooled (its score is close to a function of the basis);
* a score of i.i.d. noise (seeded, 20 draws): mean residual EP within 0.02 of 0.5.

## Rows
Benchmark B: SWORD (exact), TIRE, CUSUM, LAD, MMD, BOCPD, SCPD, counts, constant, on the ten scored datasets;
pooled all ten and >= 10 events. Sealed sets: every row of sealed.json on CHB-MIT new, Siena, Helsinki.

## Reading
A detector carries information beyond activity on a pool if the 95% CI of its residual EP lies above 0.5. All
results are reported, whichever way they go. No other variant is computed for the paper.
