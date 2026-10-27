# Analysis specs

Each file was written before the run of the analysis it specifies and is copied here unchanged. The dates in
the files and the file times are the only record of that order: the files were first committed with the
release, after the runs.

| file | specifies | used by |
|---|---|---|
| `STATS.md` | intervals, Holm families, comparison with the best, paired plant metrics | steps 14, 23 |
| `ROUND3.md` | pools, subsets, protocol variants | step 23 |
| `PEELCLAUSET.md` | the Peel-Clauset constructions and GHRG | step 17 |
| `RESIDUAL.md` | the residual design (its calibration failed; no residual EP is reported) | step 22 R |
| `V4.md`, `V5.md`, `V6.md` | label variants, count variants, LAD eigenvalues, event types, clusters, bounds | `V4.md`: steps 14, 16, 23; `V5.md`: steps 17, 19, 22, 23; `V6.md`: steps 16, 23 |

Code comments cite them as "spec v4 V3" (file `V4.md`, section V3). `S/` in a spec is the author's working
directory, which is not released. Rows and datasets that a spec names but the paper does not (an earlier
detector of the author, sealed EEG sets) belong to an earlier study and are not part of this repository.
