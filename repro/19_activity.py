"""Step 19: the wiring statistics as rows, and EP by the size of the activity change at the event -> results/activity.json.

Activity = (n_+, |E|) on the declared vertex set under the construction's connection rule; for the thresholded
correlation graphs (PAMAP2, EEG) that is the coupling level, so they are reported apart from the interaction networks.
  D  wiring rows (App. definitions) on the benchmark with the step-12 protocol: transitivity (triangles and
     transitivity of the active subgraph), assortativity (Newman r) and fair_pair (the larger of the two scores),
     each on the online scale, window rule w = 2, w_ref = 10; cyclic-shift p (1,000 shifts)
  C  plain EP of every row by activity stratum (and, spec v5 O, counts minus every compared row on ordinary events): the activity level of an event is the larger of the two percentiles
     (among the event-free windows) of the window maximum of |change of n_+| and of |change of |E||, both at the
     windows of the counts (w = 2, w_ref = 5); strata ordinary (< 0.90), large (0.90-0.99), extreme (>= 0.99); the
     share of event-free windows that reach each level when each is ranked like an event, among the event-free
     windows that do not overlap it (also per count); per row, EP on the extreme minus the ordinary events within
     each dataset that has both
Rows: step 11's caches, every s_<row> of repro/data/identity/scores_<ID>.npz and scores_new_<ID>.npz (prefix id_),
and the D rows. Plain pooled EP per network type. Inputs: steps 2, 10, 11, 12, 16.
Also writes repro/data/activity/scores_<ID>.npz (the D rows; input of step 23).

Usage: 19_activity.py [--summary]
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool

import evaluate as ev
import inference as inf
import numpy as np
import scipy.sparse as sp
from common import DATA, RESULTS, Timer, count_vectors, load, load_graphs, peak_rss_mb, sha, src_sha, write_json
from rows import BENCH, PRIMARY, REPO_VIEW, B, matrices, meta, settings

WIRING_ROWS = ["transitivity", "assortativity", "fair_pair"]
CORR = ("PM", "EEG")                                      # thresholded correlation graphs
STATS = ["nplus", "E", "trans", "tri", "xtrans", "assort", "overlap"]
STRATA = ((0.0, 0.90, "ordinary"), (0.90, 0.99, "large"), (0.99, 1.01, "extreme"))
N_SHIFT_D = 1000
CACHE = DATA / "activity"
IDENTITY = DATA / "identity"


# ------------------------------------------------------------------------------------------------ per snapshot
def triangles(B_):
    """Triangles of a simple graph: edges oriented from lower to higher (degree, index), sum (U U) * U."""
    d = np.diff(B_.indptr)
    rank = np.empty(len(d), np.int64)
    rank[np.lexsort((np.arange(len(d)), d))] = np.arange(len(d))
    C = B_.tocoo()
    keep = rank[C.row] < rank[C.col]
    U = sp.csr_matrix((np.ones(keep.sum()), (C.row[keep], C.col[keep])), shape=B_.shape)
    return float((U @ U).multiply(U).sum())


def expected_overlap(d1, m1, d0, m0):
    """sum_{u<v} min(1, d_u d_v / 2m_t) min(1, d'_u d'_v / 2m_{t-1}); a cap binds only inside the hubs H."""
    if m1 == 0 or m0 == 0:
        return 0.0
    x = d1 * d0
    mu = (x.sum() ** 2 - (x ** 2).sum()) / 2 / (4 * m1 * m0)
    H = np.flatnonzero(((d1 * d1.max() > 2 * m1) | (d0 * d0.max() > 2 * m0)) & (x > 0))
    for i in range(0, len(H), 2000):                     # exact correction on the pairs of H (block rows)
        r = H[i:i + 2000]
        p1 = np.outer(d1[r], d1[H]) / (2 * m1)
        p0 = np.outer(d0[r], d0[H]) / (2 * m0)
        dlt = np.minimum(1, p1) * np.minimum(1, p0) - p1 * p0
        dlt[r[:, None] == H[None, :]] = 0.0
        mu += dlt.sum() / 2
    return float(mu)


def snapshot(A, prev):
    """(statistics, state for the next step) of one snapshot; self-loops removed."""
    A = sp.csr_matrix(A, dtype=np.float64)
    A.setdiag(0)
    A.eliminate_zeros()
    A.data[:] = 1.0
    d = np.diff(A.indptr).astype(float)
    act = np.flatnonzero(d > 0)
    Bm = A[act][:, act].tocsr()
    m = Bm.nnz / 2
    da = d[act]
    tri = triangles(Bm) if m else 0.0
    wedges = float((da * (da - 1)).sum()) / 2
    trans = 3 * tri / wedges if wedges > 0 else 0.0
    cc = ((da ** 2).mean() - da.mean()) ** 2 / (da.mean() ** 3 * len(da)) if len(da) else 0.0
    Uc = sp.triu(Bm, 1).tocoo()
    xa, xb = da[Uc.row], da[Uc.col]
    x, y = np.r_[xa, xb], np.r_[xb, xa]
    r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and x.std() > 0 else 0.0
    U = sp.triu(A, 1).tocsr()
    if prev is None:
        ov = np.nan
    else:
        inter = U.multiply(prev[0]).nnz
        ov = float(np.log((inter + 1) / (expected_overlap(d, m, prev[1], prev[2]) + 1)))
    out = {"nplus": len(act), "E": m, "tri": tri, "trans": trans, "xtrans": trans - cc, "assort": r, "overlap": ov}
    return out, (U, d, m)


def input_key(ID: str) -> str:
    if ID in REPO_VIEW:
        return sha(*[(DATA / f"{s}_{REPO_VIEW[ID]}.npz").read_bytes() for s in ("graphs", "moments")])
    return sha((BENCH / f"{ID}.npz").read_bytes())


def stats(ID: str) -> str:
    p = CACHE / f"stats_{ID}_{sha(input_key(ID), STAT_SHA)}.npz"
    if p.exists():
        return f"{ID}: cached"
    t0 = Timer()
    seg = meta(ID)["seg"]
    # AskUbuntu's snapshots hold only their own users, in no common order: no vertex identity, no overlap
    fixed = ID not in REPO_VIEW or len(set(load_graphs(REPO_VIEW[ID])["n"].tolist())) == 1
    rows, prev = [], None
    for t, A in enumerate(matrices(ID)):
        if not fixed or (t > 0 and seg[t] != seg[t - 1]):
            prev = None                                   # no overlap across a segment boundary
        r, prev = snapshot(A, prev)
        rows.append(r)
    np.savez(p, **{k: np.array([r[k] for r in rows], float) for k in STATS}, seconds=t0())
    return f"{ID}: statistics {t0():.0f} s, peak RSS {peak_rss_mb():.0f} MB"


STAT_SHA = src_sha(triangles, expected_overlap, snapshot, matrices, stats)


def load_stats(ID: str) -> dict:
    return dict(np.load(CACHE / f"stats_{ID}_{sha(input_key(ID), STAT_SHA)}.npz"))


# ------------------------------------------------------------------------------------------------ helpers
def delta(x, seg, w: int, wr: int) -> np.ndarray:
    """Signed window change mean x[t-w, t) - mean x[t-w-wr, t-w) per segment; nan where the windows do not fit."""
    x = np.asarray(x, float)
    D = np.full(len(x), np.nan)
    for s0, s1 in ev.seg_ranges(seg):
        c = np.r_[0.0, np.cumsum(x[s0:s1])]
        t = np.arange(w + wr, s1 - s0 + 1)
        t = t[t < s1 - s0]
        D[s0 + t] = (c[t] - c[t - w]) / w - (c[t - w] - c[t - w - wr]) / wr
    return D


def pooled(vals: dict) -> dict | None:
    """Equal weight per dataset, 95% CI from evaluate.boot_pooled; vals {ds: per-event values}."""
    vals = {d: np.asarray(v, float) for d, v in vals.items() if len(v)}
    if not vals:
        return None
    b = ev.boot_pooled({d: v - 0.5 for d, v in vals.items()})
    return {"mean": b["mean"] + 0.5, "lo": b["lo"] + 0.5, "hi": b["hi"] + 0.5, "n_events": int(sum(map(len, vals.values()))),
            "datasets": sorted(vals), "per_ds": {d: [x + 0.5 for x in b["per_ds"][d]] for d in vals}}


def groups(dss) -> dict:
    inter = [d for d in dss if d not in CORR]
    return {"all": list(dss), "interaction": inter, "correlation": [d for d in dss if d in CORR]}


# ------------------------------------------------------------------------------------------------ one dataset
def identity_rows(ID: str, T: int) -> dict:
    out = {}
    for p in (IDENTITY / f"scores_{ID}.npz", IDENTITY / f"scores_new_{ID}.npz"):
        if p.exists():
            z = np.load(p)
            out.update({f"id_{k[2:]}": np.asarray(z[k], float) for k in z.files if k.startswith("s_") and len(z[k]) == T})
    return out


def null_rank_like_event(nl: np.ndarray, null: np.ndarray, seg: np.ndarray, L: int) -> np.ndarray:
    """Mid-rank percentile of each event-free window among the event-free windows that do not overlap it (the rule
    applied to an event): the rank among all windows minus the rank among its overlapping neighbours (itself included)."""
    s = np.sort(nl)
    lo, hi = np.searchsorted(s, nl, "left"), np.searchsorted(s, nl, "right")
    out = np.empty(len(nl))
    for i, u in enumerate(null):
        j = np.nonzero((np.abs(null - u) < L) & (seg[null] == seg[u]))[0]
        near_lo, near_eq = int((nl[j] < nl[i]).sum()), int((nl[j] == nl[i]).sum())
        out[i] = (lo[i] - near_lo + 0.5 * (hi[i] - lo[i] - near_eq)) / (len(nl) - len(j))
    return out


def evaluate_ds(ID: str, burn: dict, bench: dict) -> dict | None:
    mt = meta(ID)
    seg, evs, tol, T = mt["seg"], mt["events"], mt["tol"], mt["T"]
    L = tol + 1
    z = np.load(BENCH / f"scores_{ID}.npz")
    S = {k[2:]: np.asarray(z[k], float) for k in z.files if k.startswith("s_")}
    mask = ev.common_mask(S, burn, PRIMARY, seg)
    ev_ok, null = ev.windows(mask, seg, evs, evs, L)
    if not ev_ok or not len(null):
        return None
    F = load_stats(ID)
    rec = {"T": T, "tol": tol, "n_events": len(ev_ok), "n_null_windows": len(null),
           "group": "correlation" if ID in CORR else "interaction", "events": ev_ok}

    # ---- D: wiring rows ----
    zs = lambda X: ev.online_scale(X, seg)[0]
    S["transitivity"] = ev.window_scores(zs(np.c_[F["tri"], F["trans"]]), seg)
    S["assortativity"] = ev.window_scores(zs(F["assort"]), seg)
    S["fair_pair"] = np.fmax(ev.window_scores(zs(F["trans"]), seg), S["assortativity"])
    np.savez(CACHE / f"scores_{ID}.npz", **{f"s_{r}": S[r] for r in WIRING_ROWS})        # input of step 23
    S.update(identity_rows(ID, T))
    rows = [r for r in S if r != "seg" and np.isfinite(S[r][mask]).all()]
    rec["rows_not_finite_on_mask"] = [r for r in S if r != "seg" and r not in rows]
    EP = {}
    for r in rows:
        v, _ = ev.ep(S[r], mask, seg, evs, tol)
        EP[r] = np.array([v[e] for e in ev_ok])
    b = bench["datasets"][ID]["m2"]["ep_per_event"]
    rec["D0_max_abs_diff"] = max(float(np.max(np.abs(np.round(EP[r], 4) - np.array(b[r])))) for r in PRIMARY if r in b)
    rng = np.random.default_rng([ev.SEED, B.index(ID)])
    nl = ev.shift_null(S, WIRING_ROWS, mask, seg, evs, tol, rng, N_SHIFT_D)
    rec["D"] = {r: {"ep_mean": float(EP[r].mean()), "shift_p": ev.shift_p(nl[r], EP[r].mean())} for r in WIRING_ROWS}
    rec["ep_per_event"] = {r: EP[r].round(4).tolist() for r in rows}

    # ---- C: activity level of each event and its stratum ----
    C = count_vectors(load(REPO_VIEW[ID])["f"]) if ID in REPO_VIEW else ev.mad_scale(np.c_[F["nplus"], F["E"]], seg)
    level, null_level, per_count = np.zeros(len(ev_ok)), np.zeros(len(null)), {}
    for j, name in enumerate(("nplus", "E")):
        c = np.nan_to_num(np.abs(delta(C[:, j], seg, 2, 5)))
        nl = ev._wmax(c, null, L)
        level = np.maximum(level, inf.pct(nl, ev._wmax(c, ev_ok, L)))
        q = null_rank_like_event(nl, null, np.asarray(seg), L)
        per_count[name] = float((q >= 0.90).mean())
        null_level = np.maximum(null_level, q)
    rec["C_level"] = level.round(4).tolist()
    rec["C_stratum"] = [next(nm for lo, hi, nm in STRATA if lo <= x < hi) for x in level]
    rec["C_null_share"] = {"large_or_extreme": float((null_level >= 0.90).mean()), "extreme": float((null_level >= 0.99).mean()),
                           "per_count": per_count}
    return rec


# ------------------------------------------------------------------------------------------------ pooled
def assemble(R: dict) -> dict:
    dss = sorted(R)
    G = groups(dss)
    G["10plus"] = [d for d in dss if R[d]["n_events"] >= 10]
    rows = sorted(set.intersection(*[set(R[d]["ep_per_event"]) for d in dss]))
    out = {"datasets": R, "groups": G, "rows": rows}
    out["plain"] = {g: {r: pooled({d: R[d]["ep_per_event"][r] for d in ds}) for r in rows} for g, ds in G.items() if ds}
    out["C_strata"] = {}
    for g, ds in G.items():
        if not ds or g == "10plus":
            continue
        out["C_strata"][g] = {}
        for _, _, nm in STRATA:
            sel = {d: [i for i, s in enumerate(R[d]["C_stratum"]) if s == nm] for d in ds}
            out["C_strata"][g][nm] = {r: pooled({d: [R[d]["ep_per_event"][r][i] for i in ix]
                                                 for d, ix in sel.items()}) for r in rows}
            out["C_strata"][g][nm]["_n"] = {d: len(ix) for d, ix in sel.items()}
    out["C_ordinary_contrast"] = ordinary_contrast(R, G, rows)
    out["C_null_share"] = {k: float(np.mean([R[d]["C_null_share"][k] for d in dss])) for k in ("large_or_extreme", "extreme")}
    out["C_null_share"]["per_count"] = {k: float(np.mean([R[d]["C_null_share"]["per_count"][k] for d in dss])) for k in ("nplus", "E")}
    out["C_null_share"]["per_ds"] = {d: R[d]["C_null_share"] for d in dss}
    out["C_within"] = within_contrast(R, rows)
    out["D0_max_abs_diff"] = max(R[d]["D0_max_abs_diff"] for d in dss)
    out["D0_pass"] = out["D0_max_abs_diff"] < 1e-9
    out["identity_rows"] = sorted({r for d in dss for r in R[d]["ep_per_event"] if r.startswith("id_")})
    return out


COMPARED = ["cusum", "bocpd", "mmd", "lad", "scpd", "tire", "id_hamming", "id_jaccard", "id_deltacon", "id_chenzhang",
            "id_nbs", "id_ase", "id_spotlight", *WIRING_ROWS]       # the 16 compared rows (no reference rows)


def ordinary_contrast(R: dict, G: dict, rows) -> dict:
    """Ordinary stratum (spec v5 O): counts minus every compared row, equal weight per dataset over the datasets with
    an ordinary event; event-level (evaluate.boot_pooled) and two-stage (inference.pooled_draws) intervals; the best
    row is the one with the highest pooled EP in the stratum (chosen after the fact)."""
    q = lambda d: [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]
    out = {}
    for g in ("all", "interaction", "correlation"):
        sel = {d: [i for i, s in enumerate(R[d]["C_stratum"]) if s == "ordinary"] for d in G[g]}
        sel = {d: ix for d, ix in sel.items() if ix}
        cmp_ = [r for r in COMPARED if r in rows]
        X = {d: np.stack([np.array(R[d]["ep_per_event"][r])[ix] for r in ["counts"] + cmp_], 1) for d, ix in sel.items()}
        pt, dr = inf.pooled_point(X), inf.pooled_draws(X, "two_stage")
        res = {}
        for i, r in enumerate(cmp_, 1):
            evl = ev.boot_pooled({d: X[d][:, 0] - X[d][:, i] for d in X})
            res[r] = {"ep": float(pt[i]), "diff": float(pt[0] - pt[i]), "ci95_event": [evl["lo"], evl["hi"]],
                      "ci95_two_stage": q(dr[:, 0] - dr[:, i]), "p_two_stage": inf.pval(dr[:, 0] - dr[:, i])}
        best = max(cmp_, key=lambda r: res[r]["ep"])
        out[g] = {"datasets": sorted(X), "n_events": int(sum(len(x) for x in X.values())), "counts": float(pt[0]),
                  "best": best, "counts_minus_best": res[best], "counts_minus": res}
    return out


def within_contrast(R: dict, rows) -> dict:
    """Per row: EP on the extreme minus EP on the ordinary events within each dataset that has both, and its mean."""
    out = {}
    for r in rows:
        dd = {}
        for d, x in R.items():
            ep, st = np.array(x["ep_per_event"][r]), np.array(x["C_stratum"])
            if (st == "extreme").any() and (st == "ordinary").any():
                dd[d] = float(ep[st == "extreme"].mean() - ep[st == "ordinary"].mean())
        out[r] = {"mean": float(np.mean(list(dd.values()))), "n_positive": int(sum(v > 0 for v in dd.values())),
                  "per_ds": dd}
    return out


# ------------------------------------------------------------------------------------------------ summary
def f(v, d=2):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


def ci(b, d=3):
    return "n/a" if not b else f"{f(b['mean'], d)} [{f(b['lo'], d)}, {f(b['hi'], d)}]"


def table(head, rows):
    s = "| " + " | ".join(head) + " |\n|" + "|".join("---" for _ in head) + "|\n"
    return s + "".join("| " + " | ".join(str(c) for c in r) + " |\n" for r in rows)


def summary(O: dict) -> str:
    R, G = O["datasets"], O["groups"]
    dss = G["all"]
    md = [f"Groups: {json.dumps(G)}\n", (f"D0 (EP equals bench.json): max diff {O['D0_max_abs_diff']:.1e} -> "
          f"{'PASS' if O['D0_pass'] else 'FAIL'}\n")]
    md.append("## C. Plain EP by activity stratum\n")
    for g, blk in O["C_strata"].items():
        md.append(f"\n{g}: n per stratum " + json.dumps({k: sum(v['_n'].values()) for k, v in blk.items()}) + "\n")
        md.append(table(["row"] + [s for _, _, s in STRATA], [[r] + [ci(blk[s][r]) for _, _, s in STRATA]
                                                              for r in O["rows"]]))
    if "C_null_share" in O:
        md.append(f"Event-free windows at level >= 0.90 (ranked like events), mean over datasets: {O['C_null_share']['large_or_extreme']:.3f}; "
                  + json.dumps({d: round(v["large_or_extreme"], 3) for d, v in O["C_null_share"]["per_ds"].items()}) + "\n")
    if "C_within" in O:
        md.append("Within datasets, EP extreme - ordinary (mean, datasets positive): " + "; ".join(
            f"{r} {v['mean']:+.3f} ({v['n_positive']}/{len(v['per_ds'])})" for r, v in O["C_within"].items()) + "\n")
    md.append("## C. Ordinary stratum: counts - best other row (event-level / two-stage CI)\n")
    for g, b in O.get("C_ordinary_contrast", {}).items():
        c = b["counts_minus_best"]
        md.append(f"* {g} ({b['n_events']} events, {len(b['datasets'])} datasets): counts {b['counts']:.3f}, best "
                  f"{b['best']} {c['ep']:.3f}, diff {c['diff']:+.3f} [{c['ci95_event'][0]:+.3f}, {c['ci95_event'][1]:+.3f}]"
                  f" / [{c['ci95_two_stage'][0]:+.3f}, {c['ci95_two_stage'][1]:+.3f}]")
    md.append("## Plain pooled EP by network type\n")
    gs = [g for g in ("all", "10plus", "interaction", "correlation") if g in O["plain"]]
    md.append(table(["row"] + gs, [[r] + [ci(O["plain"][g][r]) for g in gs] for r in O["rows"]]))
    md.append("## D. Wiring rows, plain EP per dataset (shift p)\n")
    md.append(table(["row"] + dss, [[r] + [f"{f(R[d]['D'][r]['ep_mean'])} (p={f(R[d]['D'][r]['shift_p'], 3)})"
                                           for d in dss] for r in WIRING_ROWS]))
    md.append(f"\nIdentity-lane rows included: {O['identity_rows'] or 'none'}\n")
    return "\n".join(md)


def main() -> None:
    p = RESULTS / "activity.json"
    if "--summary" in sys.argv and p.exists():
        print(summary(json.loads(p.read_text())))
        return
    t0 = Timer()
    CACHE.mkdir(parents=True, exist_ok=True)
    st = settings()
    big = ["SCC", "MAG", "EEG", "PM", "ENRON"]
    with Pool(4, maxtasksperchild=1) as pool:
        for line in pool.imap_unordered(stats, big + [i for i in B if i not in big]):
            print(line, flush=True)
    bench = json.loads((RESULTS / "bench.json").read_text())
    R = {}
    for ID in B:
        r = evaluate_ds(ID, st["burn"], bench)
        if r is not None:
            R[ID] = r
            print(ID, f"{t0():.0f} s", flush=True)
    out = assemble(R)
    out["_runtime_s"] = round(t0(), 1)
    write_json(p, out)
    print(f"step 19: {t0():.0f} s -> {p}")
    print(summary(json.loads(p.read_text())))


if __name__ == "__main__":
    main()
