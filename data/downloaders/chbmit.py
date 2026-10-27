"""EEG: CHB-MIT chb01/03/05 seizure files as correlation networks (repro/prereg DEVIATIONS 6, 9, 16; LABEL_AUDIT).

Raw: data/raw/chbmit_b/ (19 EDF files + chbNN-summary.txt + PhysioNet SHA256SUMS.txt). Recipe: channels = nodes;
drop a channel whose |r| with an earlier channel is >= 0.999 over the whole file (DEV 9); 2 s non-overlapping
windows (512 samples at 256 Hz) from the file start, last partial window dropped; Pearson r; a channel that is
exactly constant in a window has undefined r -> no edge (DEV 16); edge iff |r| >= 0.5. One segment per seizure file;
label index = floor(onset_seconds / 2). Dropout flag: the window overlaps a run of >= 256 equal raw samples in any
channel (flagged, recipe unchanged). build_check compares with repro/prereg/results/EEG.json (label-free statistics).
"""
import itertools
import json
import os
import re

import numpy as np

from data.downloaders.snapshots import sha256, to_csr

PREREG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                      "repro", "prereg", "results", "EEG.json")
FS, WIN, ETA = 256, 512, 0.5


def read_edf(path):
    with open(path, "rb") as f:
        h = f.read(256)
        nrec, dur, ns = int(h[236:244]), float(h[244:252]), int(h[252:256])
        sh = f.read(256 * ns)
        fld = lambda off, w: [sh[off + i * w: off + (i + 1) * w].decode("ascii").strip() for i in range(ns)]
        o = 0
        lab = fld(o, 16); o += 16 * ns
        o += 80 * ns + 8 * ns
        pmin = np.array(fld(o, 8), float); o += 8 * ns
        pmax = np.array(fld(o, 8), float); o += 8 * ns
        dmin = np.array(fld(o, 8), float); o += 8 * ns
        dmax = np.array(fld(o, 8), float); o += 8 * ns
        o += 80 * ns
        nsamp = np.array(fld(o, 8), int)
        assert (nsamp == nsamp[0]).all() and nsamp[0] == FS * dur
        raw = np.frombuffer(f.read(), dtype="<i2")
    raw = raw[: nrec * ns * nsamp[0]].reshape(nrec, ns, nsamp[0]).transpose(1, 0, 2).reshape(ns, -1)
    gain = (pmax - pmin) / (dmax - dmin)
    return lab, raw, gain, pmin - dmin * gain


def onsets(raw_dir, subj):
    with open(f"{raw_dir}/{subj}-summary.txt") as f:
        txt = f.read()
    out = {}
    for blk in txt.split("File Name: ")[1:]:
        name = blk.split()[0][:-4]
        out[name] = [(int(a), int(b)) for a, b in zip(re.findall(r"Seizure(?: \d+)? Start Time: *(\d+) seconds", blk),
                                                       re.findall(r"Seizure(?: \d+)? End Time: *(\d+) seconds", blk))]
    return out


def runs_windows(x, nwin):  # windows that overlap a run of >= 256 equal consecutive raw values in any channel
    out = set()
    for c in range(x.shape[0]):
        idx = np.flatnonzero(np.concatenate([[True], np.diff(x[c]) != 0, [True]]))
        for a, b in itertools.pairwise(idx):
            if b - a >= 256:
                out.update(q for q in range(a // WIN, (b - 1) // WIN + 1) if q < nwin)
    return sorted(out)


def series(raw_dir):
    with open(PREREG) as f:
        REF = json.load(f)
    FILES = [e["file"][:-4] for e in REF["events"]]
    mats, seg, events, ev_meta, seg_info, nodes = [], [], [], [], [], None
    summ = {s: onsets(raw_dir, s) for s in ("chb01", "chb03", "chb05")}
    for k, name in enumerate(FILES):
        lab, raw, gain, off = read_edf(f"{raw_dir}/{name}.edf")
        X = raw * gain[:, None] + off[:, None]
        C = np.corrcoef(X)
        drop = [j for j in range(len(lab)) if any(abs(C[i, j]) >= 0.999 for i in range(j))]
        keep = [j for j in range(len(lab)) if j not in drop]
        names = [lab[j] for j in keep]
        if nodes is None:
            nodes = names
        assert names == nodes, (name, names)
        X, R = X[keep], raw[keep]
        nwin = X.shape[1] // WIN
        iu = np.triu_indices(len(keep), 1)
        dropout = runs_windows(R, nwin)
        for q in range(nwin):
            Xw = X[:, q * WIN:(q + 1) * WIN]
            Rw = R[:, q * WIN:(q + 1) * WIN]
            flat = np.ptp(Rw, axis=1) == 0
            Z = Xw - Xw.mean(1, keepdims=True)
            nr = np.sqrt((Z * Z).sum(1))
            with np.errstate(invalid="ignore", divide="ignore"):
                r = (Z @ Z.T) / np.outer(nr, nr)
            r[flat, :] = np.nan
            r[:, flat] = np.nan
            a, b = iu
            rr = r[a, b]
            e = np.isfinite(rr) & (np.abs(rr) >= ETA)
            mats.append(to_csr(len(keep), a[e], b[e]))
            seg.append(k)
        sz = summ[name[:5]][name]
        assert len(sz) == 1, (name, sz)
        on, offs = sz[0]
        gidx = len(mats) - nwin + on // 2
        events.append(gidx)
        ev_meta.append({"segment": k, "file": f"{name}.edf", "index": on // 2, "global_index": gidx, "onset_sec": on, "offset_sec": offs,
                            "citation": f"https://physionet.org/content/chbmit/1.0.0/{name[:5]}/{name[:5]}-summary.txt"})
        seg_info.append({"segment": k, "file": f"{name}.edf", "n_windows": nwin, "dropped_channels": [lab[j] for j in drop],
                             "dropout_windows": dropout})
    seg = np.asarray(seg)
    edges = np.array([A.nnz // 2 for A in mats])
    active = np.array([(np.diff(A.indptr) > 0).sum() for A in mats], float)
    def mad_scale(f):
        d = np.concatenate([np.diff(f[seg == s]) for s in np.unique(seg)])
        return 1.4826 * np.median(np.abs(d - np.median(d)))
    V = [mad_scale(np.log1p(active)), mad_scale(np.log1p(edges))]
    ref_ev = [(e["segment"], e["index"]) for e in REF["events"]]
    ref_drop = [f["dropout"] for f in REF["segment_flags"]]
    check = {"T": len(mats), "T_ref": REF["n_steps"], "events_identical": [(m["segment"], m["index"]) for m in ev_meta] == ref_ev,
                 "dropout_identical": [s["dropout_windows"] for s in seg_info] == ref_drop,
                 "V_scale": V, "V_scale_ref": REF["score_diag"]["scale"]["V"],
                 "V_scale_maxdiff": float(np.max(np.abs(np.array(V) - REF["score_diag"]["scale"]["V"])))}
    meta = {
        "id": "EEG", "n": len(nodes), "T": len(mats), "nodes": nodes, "eta": ETA, "segments": seg_info,
        "windows": {"w": 3, "w_ref": 15, "delta": 5, "H": 15}, "events": ev_meta,
        "snapshot_rule": "2 s non-overlapping windows (512 samples) per seizure file; Pearson r of the bipolar channels; "
                      "edge iff |r| >= 0.5; constant channel in a window -> no edge; duplicate channels (|r| >= 0.999 over "
                      "the file) dropped (DEV 9)",
        "source_url": "https://physionet.org/content/chbmit/1.0.0/ (files fetched from the mirror "
                   "https://physionet-open.s3.amazonaws.com/chbmit/1.0.0/; sha256 checked against SHA256SUMS.txt)",
        "raw_sha256": {f: sha256(f"{raw_dir}/{f}") for f in sorted(os.listdir(raw_dir)) if f.endswith((".edf", ".txt"))},
        "label_source": "Shoeb, PhD thesis MIT 2009; Goldberger et al., Circulation 101 (2000) e215 (PhysioNet); onsets from "
                     "chbNN-summary.txt (LABEL_AUDIT: all 19 agree with the WFDB .seizures annotations)",
        "used_by": [{"paper": "this project: repro/prereg (PROTOCOL DEV 6, 9, 16)", "use": "decisive set, w=3, w_ref=15, delta=5"},
                 {"paper": "not part of the sealed set Z (CHB-MIT-new excludes chb01, chb03, chb05 and chb21)", "use": "-"}],
        "edges_mean": float(edges.mean()), "n_empty": int((edges == 0).sum()), "build_check": check}
    return [{"id": "EEG", "mats": mats, "seg": seg, "events": events, "tol": 5, "meta": meta, "extra": {}}]
