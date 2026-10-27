"""SocioPatterns contact series PS (primary school) and HT09 (Hypertext 2009), 5-min bins, timetable labels.

Raw: data/raw/sociopatterns/{primaryschool.csv.gz, primaryschool_metadata.txt, ht2009_contact_list.dat.gz}.
Rules (repro/prereg PROTOCOL.md, DEVIATIONS.md items 2, 3, 5, 15):
  * contact row "t i j" = contact active in [t-20, t]; contact time = t - 20 s.
  * vertex set = every ID in the raw file (fixed n), ordered by sorted ID.
  * PS clock: t = seconds from local midnight 2009-10-01 (CEST; inferred, DEV 15). Clipped to 08:30-16:30.
  * HT09 clock: t = seconds since 08:00 CEST 2009-06-29. Per local day, records split at gaps >= 30 min,
    largest block kept (DEV 5).
  * segment = one local day; complete 5-min bins only: start = first 5-min boundary >= first contact
    time, end = last 5-min boundary <= last contact time + 20 s (DEV 15). Empty bins are kept.
  * snapshot = undirected simple graph, edge iff >= 1 contact row in the bin.
  * labels = bin whose start equals the published clock time; burn-in w+w_ref = 6 must not hold a label.
"""
import gzip
import os

import numpy as np

from data.downloaders.snapshots import sha256, to_csr

BIN = 300
W, WREF, DELTA, H = 2, 4, 2, 4


def read(path):
    with gzip.open(path, "rt") as f:
        rows = [ln.split() for ln in f if ln.strip()]
    t = np.array([int(r[0]) for r in rows], np.int64)
    i = np.array([int(r[1]) for r in rows], np.int64)
    j = np.array([int(r[2]) for r in rows], np.int64)
    return t, i, j


def hm(s):
    s = int(s) % 86400
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}"


def build(ID, path, origin, clip, gap_split, labels, labels2=None, label_src=""):
    t, i, j = read(path)
    ids = np.unique(np.concatenate([i, j]))
    pos = {v: k for k, v in enumerate(ids)}
    n = ids.size
    I = np.array([pos[v] for v in i]); J = np.array([pos[v] for v in j])
    loc = t - 20 + origin  # local seconds since midnight of day 0, contact time
    day = loc // 86400
    mats, seg, seg_info, dropped = [], [], [], 0
    times = []
    events, events2 = [], []
    for d in np.unique(day):
        m = np.where(day == d)[0]
        m = m[np.argsort(loc[m], kind="stable")]
        if gap_split:
            g = np.diff(loc[m])
            cuts = np.where(g >= 1800)[0]
            blocks = np.split(m, cuts + 1)
            big = max(blocks, key=len)
            dropped += m.size - big.size
            m = big
        s = int(d) * 86400
        lo, hi = loc[m].min(), loc[m].max() + 20  # hi = last t
        if clip is not None:
            keep = (loc[m] >= s + clip[0]) & (loc[m] < s + clip[1])
            dropped += int((~keep).sum())
            m = m[keep]
            lo, hi = loc[m].min(), min(loc[m].max() + 20, s + clip[1])
        start = -(-lo // BIN) * BIN
        end = (hi // BIN) * BIN
        nb = int((end - start) // BIN)
        k = len(seg_info)
        b = (loc[m] - start) // BIN
        inb = (b >= 0) & (b < nb)
        dropped += int((~inb).sum())
        mm, bb = m[inb], b[inb]
        g0 = len(mats)
        for q in range(nb):
            sel = mm[bb == q]
            mats.append(to_csr(n, I[sel], J[sel]))
            seg.append(k)
            times.append(int(start + q * BIN))
        info = {"segment": k, "local_day": int(d), "first_bin": hm(start), "end": hm(end), "n_bins": nb,
                    "rows_used": int(mm.size)}
        for lab_list, out in ((labels, events), (labels2 or {}, events2)):
            for (hh, mi, desc) in lab_list.get(int(d), []):
                ts = s + hh * 3600 + mi * 60
                idx = (ts - start) // BIN
                assert (ts - start) % BIN == 0 and 0 <= idx < nb, (ID, d, hh, mi)
                assert idx >= W + WREF, f"label in burn-in {ID} {d} {hh}:{mi}"
                out.append({"segment": k, "index": int(idx), "global_index": int(g0 + idx),
                                "time_local": f"{hh:02d}:{mi:02d}", "desc": desc})
        seg_info.append(info)
    # merge rule: events closer than delta+1 steps (same segment) merge into the earliest
    def merge(ev):
        out = []
        for e in sorted(ev, key=lambda e: e["global_index"]):
            if out and out[-1]["segment"] == e["segment"] and e["index"] - out[-1]["index"] < DELTA + 1:
                out[-1].setdefault("merged", []).append(e["desc"]); continue
            out.append(e)
        return out
    events, events2 = merge(events), merge(events2)
    edges = np.array([A.nnz // 2 for A in mats])
    active = np.array([int((np.diff(A.indptr) > 0).sum()) for A in mats])
    meta = {"id": ID, "n": int(n), "T": len(mats), "segments": seg_info, "rows_total": int(t.size), "rows_dropped": dropped,
                "windows": {"w": W, "w_ref": WREF, "delta": DELTA, "H": H}, "bin_seconds": BIN,
                "events": events, "events_secondary": events2, "label_source": label_src,
                "edges_total": int(edges.sum()), "edges_mean": float(edges.mean()),
                "active_mean": float(active.mean()), "n_empty": int((edges == 0).sum())}
    extra = {"bin_start_local": np.array(times, np.int64), "vertex_ids": ids}
    if labels2:
        extra["events_secondary"] = np.array([e["global_index"] for e in events2], np.int64)
    return {"id": ID, "mats": mats, "seg": np.asarray(seg, np.int64), "events": [e["global_index"] for e in events],
                "tol": DELTA, "meta": meta, "extra": extra}


def series(raw_dir):
    """PS and HT09; file_meta = the registration keys of benchmark B (written to <ID>_meta.json only)."""
    PS_SRC = "Stehle et al., PLoS ONE 6(8):e23176 (2011), https://doi.org/10.1371/journal.pone.0023176"
    ps_lunch = {0: [(12, 0, "lunch start (nominal)"), (14, 0, "lunch end")],
                1: [(12, 0, "lunch start (nominal)"), (14, 0, "lunch end")]}
    ps_breaks = {0: [(10, 30, "break ~10:30 (approx)"), (15, 30, "break ~15:30 (approx)")],
                 1: [(10, 30, "break ~10:30 (approx)"), (15, 30, "break ~15:30 (approx)")]}
    for d in (0, 1):
        ps_breaks[d] = sorted(ps_breaks[d] + ps_lunch[d])
    ps = build("PS", os.path.join(raw_dir, "primaryschool.csv.gz"), origin=0, clip=(8 * 3600 + 1800, 16 * 3600 + 1800),
               gap_split=False, labels=ps_lunch, labels2=ps_breaks, label_src=PS_SRC)
    tue = [(10, 30, "coffee break start"), (10, 45, "keynote (plenary)"), (11, 45, "coffee break start"),
           (12, 0, "session start"), (13, 30, "lunch start"), (15, 0, "sessions start"),
           (16, 30, "coffee break start"), (16, 45, "sessions resume")]
    ht = {0: [(11, 45, "coffee break start"), (12, 0, "workshops resume"), (13, 30, "lunch start"),
              (15, 0, "workshops start"), (16, 30, "coffee break start"), (16, 45, "workshops resume")],
          1: tue, 2: tue}
    ht09 = build("HT09", os.path.join(raw_dir, "ht2009_contact_list.dat.gz"), origin=8 * 3600, clip=None,
                 gap_split=True, labels=ht,
                 label_src="HT 2009 programme, http://web.archive.org/web/20090619001905/http://www.ht2009.org:80/program.php")
    ps["file_meta"] = {
        "source_url": "https://sociopatterns.org/assets/data/primaryschool.csv.gz",
        "raw_sha256": {f: sha256(os.path.join(raw_dir, f)) for f in ("primaryschool.csv.gz", "primaryschool_metadata.txt")},
        "built_by": "data/downloaders/sociopatterns.py",
        "used_by": [{"paper": "Stehle et al., PLoS ONE 6(8):e23176 (2011)", "use": "data source and timetable (label source)"},
                 {"paper": "this project: repro/prereg (PROTOCOL.md, DEV 2, 15)", "use": "5-min bins, w=2, w_ref=4, delta=2"}]}
    ht09["file_meta"] = {
        "source_url": "https://sociopatterns.org/assets/data/ht2009_contact_list.dat.gz",
        "raw_sha256": {"ht2009_contact_list.dat.gz": sha256(os.path.join(raw_dir, "ht2009_contact_list.dat.gz"))},
        "built_by": "data/downloaders/sociopatterns.py",
        "used_by": [{"paper": "Isella et al., J. Theor. Biol. 271 (2011) 166-180", "use": "data source"},
                 {"paper": "this project: repro/prereg (DEV 3, 5, 15)", "use": "programme times as labels"}]}
    return [ps, ht09]
