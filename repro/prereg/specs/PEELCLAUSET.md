# Spec: Peel & Clauset (2015) constructions, GHRG, and their mean-degree baseline (lane peelclauset)

Written 2026-10-24 before any score of the new series was computed.

## Questions
Q1. On P&C's own constructions (MIT Reality Mining weekly, Enron 151 users weekly) and with all of their events,
    what is the EP of every benchmark row, and does the count detector still lead?
Q2. What does P&C's own detector (GHRG + GLR, parametric bootstrap, w = 4) score there?
Q3. Does P&C's mean-degree / clustering baseline (univariate Gaussian GLR, bootstrap, w = 4) find only
    "a total of 2 change points" on MIT on our rebuild, and how does this square with "counts suffice"?

## Data (provenance in each _meta.json)
- MIT_PC: public KONECT copy of the Reality Mining Bluetooth contacts (konect.cc/networks/mit; 96 subjects,
  2004-09-14 .. 2005-05-05). P&C: 97 subjects, 35 weeks. Weekly graphs, weeks start Sunday (P&C's own dendrogram
  files are named by Sundays, 2004-09-05 ...), US/Eastern local time; edge iff >= 1 contact in the week; vertex set =
  all subjects of the file. If the third-party Rda copy of realitymining.mat can be parsed within memory, it is used
  only as a cross-check of the vertex count and weeks.
- ENRON_PC: CMU corpus enron_mail_20150507.tar.gz (sha256 recorded), P&C's 151 addresses (addresses.txt from Peel's
  archived homepage) and the intended logic of Peel's enron_parser.py: sender = From address, receivers = the To
  header (with continuation lines, up to Subject), both in the 151; undirected, no self-loops; week k = Monday
  1999-05-10 + 7k (UTC), k = 0 .. 163 (through June 2002). The published parser strips characters of "From: " from
  both ends of the sender (str.strip), which cuts ".com" to ".c" and would yield no link at all; the rebuild uses the
  intended address.
- Events: MIT, the 16 distinct boundary weeks of the 11 periods of P&C Fig. 8 (MIT_meta.json peel_clauset_events);
  ENRON, the 25 events of P&C Fig. 8 at the plotted week (ENRON_meta.json peel_clauset_events, week_from_1999_05_10
  floored); sensitivity: the same 25 at the date of Peel's own timeline file (enrontimeline.txt), week = floor.

## Scoring
- Rows: every row of repro/11_bench_scores.py at its fixed setting (step 11 run on the new IDs), plus GHRG
  (p-value of the window test; secondary: the window's max log likelihood ratio) and P&C's three Gaussian-GLR baselines
  (mean degree, mean local clustering, mean geodesic distance; p-value), all with w = 4.
- Alignment: D_t uses snapshots before t (GHRG / P&C baselines: the window t-4 .. t-1).
- EP: repro/evaluate.ep. Primary tolerance tol = 1 week (D_e, D_{e+1}; D_{e+1} is the first score that sees
  snapshot e). Sensitivity: tol = 2.
- Masks: (a) the step-12 protocol: common mask over the step-12 primary rows (common burn-in = the largest row
  burn-in, 17 steps) plus the new rows; (b) the P&C protocol: burn-in 4 (their window), each row on the steps where
  it is finite (rows with no score there are reported as such; not paired).
- P&C's own detection protocol (sequential, slide to the detected change point, detect if p > 0.95) is replayed from
  the per-window p-values, giving each method's detection list; reported next to P&C's published detections
  (Fig. 8 green bars, extracted from the vector PDF, and the dendrogram weeks of Peel's demo page).

## Calibration checks (reported as pass/fail)
C1. Event weeks of the rebuild match the stored P&C lists (count, distinct weeks).
C2. The vectorised GHRG port reproduces the authors' Python test statistic exactly on >= 2 windows, and its
    bootstrap null distribution matches theirs (two-sample KS p > 0.01 at 200 draws each), using the same consensus
    tree from the authors' compiled fitHRG / consensusHRG.
C3. On null data (a window with the 4 graphs drawn from one fitted HRG), the GHRG p-value is not concentrated above
    0.95 (false detection rate <= 0.10 over >= 20 windows) -- if the budget allows.

## Reading rules
- "Count lead holds on P&C constructions" iff counts has the highest per-dataset EP among the step-12 primary rows on
  both new sets under the primary mask and tolerance; otherwise report who leads and by how much.
- Q3 reproduced iff our replay of the mean-degree and clustering baselines detects <= 4 change points in total on
  MIT_PC (P&C: 2); else "not reproduced", with the number.
- Every result is reported whichever way it goes. Nothing is tuned on the new series.
