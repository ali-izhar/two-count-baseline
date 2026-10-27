"""PM: PAMAP2 correlation networks (s-GNN recipe, Sulem et al., MLJ 2024 / arXiv:2203.15470).

Raw: data/raw/pamap2/subject101.dat .. subject108.dat (PAMAP2_Dataset/Protocol). Recipe (DEVIATIONS.md items 6, 10, 13):
  * 27 nodes = 3 IMUs (hand, chest, ankle) x (accelerometer +-16g xyz, gyroscope xyz, magnetometer xyz).
  * non-overlapping 100-row windows from row 0; keep a window iff its 100 activity labels are equal and != 0.
  * Pearson r per window, pairwise complete with min_periods=30 (DEV 13); undefined r (NaN, flat) -> no edge.
  * edge iff |r| > 0.2; label = first kept snapshot whose activity differs from the previous one (same subject).
  * one segment per subject.
"""
import os

import numpy as np
import pandas as pd

from data.downloaders.snapshots import sha256, to_csr

W, WREF, DELTA, H = 3, 7, 5, 10
ETA = 0.2
COLS = []
for base, part in ((3, "hand"), (20, "chest"), (37, "ankle")):
    COLS += [(base + 1 + k, f"{part}_acc16_{a}") for k, a in enumerate("xyz")]
    COLS += [(base + 7 + k, f"{part}_gyro_{a}") for k, a in enumerate("xyz")]
    COLS += [(base + 10 + k, f"{part}_mag_{a}") for k, a in enumerate("xyz")]
IDX = [c for c, _ in COLS]
REF_COUNTS = {101: 2490, 102: 2618, 103: 1732, 104: 2302, 105: 2710, 106: 2488, 107: 2314, 108: 2607}
SGNN = "Sulem, Kenlay, Cucuringu, Dong, Machine Learning 113 (2024), doi:10.1007/s10994-023-06405-x (arXiv:2203.15470)"


def series(raw_dir):
    mats, seg, events, ev_meta, seg_info = [], [], [], [], []
    n = len(IDX)
    iu = np.triu_indices(n, 1)
    for k, subj in enumerate(range(101, 109)):
        df = pd.read_csv(os.path.join(raw_dir, f"subject{subj}.dat"), sep=" ", header=None, usecols=[0, 1] + IDX)
        lab = df[1].to_numpy()
        X = df[IDX].to_numpy(float)
        # DEV 9 duplicate-channel rule (|r| >= 0.999 over the whole file)
        C = pd.DataFrame(X).corr(min_periods=30).to_numpy()
        dup = [(int(a), int(b)) for a, b in zip(*iu) if np.isfinite(C[a, b]) and abs(C[a, b]) >= 0.999]
        nwin = len(lab) // 100
        prev, kept, starts = None, 0, []
        for q in range(nwin):
            L = lab[q * 100:(q + 1) * 100]
            if L[0] == 0 or not np.all(L == L[0]):
                continue
            Xw = X[q * 100:(q + 1) * 100]
            flat = np.array([np.nanmax(c) == np.nanmin(c) if np.isfinite(c).any() else True for c in Xw.T])
            R = pd.DataFrame(Xw).corr(min_periods=30).to_numpy()
            R[flat, :] = np.nan
            R[:, flat] = np.nan
            a, b = iu
            r = R[a, b]
            e = np.isfinite(r) & (np.abs(r) > ETA)
            mats.append(to_csr(n, a[e], b[e]))
            seg.append(k)
            if prev is not None and L[0] != prev:
                events.append(len(mats) - 1)
                ev_meta.append({"subject": subj, "segment": k, "index": kept, "frm": int(prev), "to": int(L[0]),
                                    "gap_windows_before": int(q - starts[-1] - 1)})
            prev = int(L[0]); starts.append(q); kept += 1
        seg_info.append({"subject": subj, "segment": k, "rows": len(lab), "windows": nwin, "kept": kept,
                             "ref_count_audit": REF_COUNTS[subj], "n_events": sum(1 for m in ev_meta if m["subject"] == subj),
                             "duplicate_channel_pairs": dup})
    edges = np.array([A.nnz // 2 for A in mats])
    meta = {"id": "PM", "n": n, "T": len(mats), "nodes": [nm for _, nm in COLS], "eta": ETA, "segments": seg_info,
                "windows": {"w": W, "w_ref": WREF, "delta": DELTA, "H": H}, "events": ev_meta,
                "edges_mean": float(edges.mean()), "n_empty": int((edges == 0).sum()),
                "source": "https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring"}
    file_meta = {
        "source_url": "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip",
        "raw_sha256": {f"subject10{k}.dat": sha256(os.path.join(raw_dir, f"subject10{k}.dat")) for k in range(1, 9)},
        "built_by": "data/downloaders/pamap2.py",
        "used_by": [{"paper": SGNN, "use": "Sec. 'Correlation networks from physical activity monitoring': 1 s windows, |r|>0.2, "
                      "27 nodes, labels = activity change; s-GNN uses L=20, adjusted F1 (point-adjust, Xu et al. 2018) with "
                      "+-5 tolerance; individual-level (70/30 split, 10 repetitions), random split, LOSO"}]}
    file_meta["sgnn_comparison"] = {
        "source": "arXiv:2203.15470 source (heldout/raw/PM/paper/input_core.tex, Sec. exp_physical, Table act_per_subj)",
        "same": ["8 subjects 101-108 (Protocol files)", "27 nodes = 3 IMUs x (accelerometer, gyroscope, magnetometer) x xyz",
                 "non-overlapping 100-sample (1 s) windows", "A_t = 1{|C_t| > 0.2}",
                 "label = change of activity between two consecutive snapshots",
                 "change points per subject 13,13,10,11,13,13,12,13 (identical)"],
        "differs": [("timestamps per subject: paper 2490,2618,1732,2302,2709,2487,2314,2606; ours 2490,2618,1732,2302,"
                     "2710,2488,2314,2607 (subjects 105,106,108 one more snapshot each; cause not identified)"),
                    "accelerometer: the paper does not say which of the two (+-16 g / +-6 g); ours uses +-16 g",
                    ("window keep rule (all 100 labels equal and non-zero), NaN handling (pairwise, min_periods=30) and "
                     "flat-channel rule are ours; the paper does not state them"),
                    ("evaluation: paper = adjusted F1 (point-adjust, Xu et al. 2018) with +-5 two-sided tolerance, s-GNN "
                     "window L=20, supervised training splits; ours = one-sided tolerance 5, w=3, w_ref=7, one series "
                     "of 8 segments (one per subject), each with its own burn-in"),
                    "the paper's PAMAP2 network hyperparameters (lr, hidden units, Sort-k size, dropout) are not reported"],
        "reported_sgnn_adjusted_f1": {"individual_level": {"1": 0.76, "2": 0.91, "3": 0.60, "4": 0.73, "5": 0.85,
                                                         "6": 0.74, "7": 0.90, "8": 0.72},
                                       "random_split": 0.81, "LOSO": 0.89,
                                       "table": "Table res_exp1 (mean over 10 splits / 8 folds)"}}
    return [{"id": "PM", "mats": mats, "seg": np.asarray(seg, np.int64), "events": events, "tol": DELTA, "meta": meta,
             "extra": {}, "file_meta": file_meta}]
