"""Step 12: every row on every dataset: each source paper's own metric (key m1) and the event percentile (key m2).

m1  LAD sets: Hits@n (exact index match, LAD's compute_accuracy); SCPD sets (SCPD reports no number): Hits@|E|, recall
    of the events among the top k steps (k = the number of SCPD's published detections), overlap with those
    detections; PM: adjusted F1 of the s-GNN protocol (individual split, tolerance 5, 10 repetitions per subject,
    threshold chosen on the train part); MIT / ENRON / AskUbuntu: F1 of the selection series under the fixed alarm
    rules.
m2  EP per event on the steps every primary row scores; cyclic-shift chance p (1,000 shifts); pooled over datasets with
    equal weight, 95% CI by bootstrap of events within each dataset; paired contrasts EP(counts) - EP(row).
Own-code rows (the source papers' own constructions, step 13; G0 in the code) enter m1 and section 8 only, never the
pooled numbers. Inputs: steps 10, 11 (and 13 for the own-code rows). Output: results/bench.json; `--summary` prints
the tables. Pooled rows: rows.PRIMARY, which also define the common mask. Inference over every row, the robustness
pools and the dataset-level intervals: step 23.

Usage: 12_bench_eval.py [--summary]
"""
from __future__ import annotations

import json
import sys

import evaluate as ev
import numpy as np
from common import RESULTS, Timer, alarms, f1_score, theta_of, write_json
from rows import BENCH, PRIMARY, REPO_VIEW, B, meta, settings, window_burn

from methods.interface import apply_cooldown

SOURCE = {"UCI": "LAD", "USLegis": "LAD", "canVote": "LAD", "MAG": "SCPD", "FL": "SCPD", "SCC": "SCPD", "PM": "s-GNN",
          "MIT": "Peel&Clauset / repo Table 1", "ENRON": "Peel&Clauset / repo Table 1", "PS": "repo prereg",
          "HT09": "repo prereg", "EEG": "repo prereg", "AskUbuntu": "repo Table 1"}
G0 = {"UCI": "lad_own", "USLegis": "lad_own", "canVote": "lad_own", "MAG": "scpd_own", "FL": "scpd_own",
      "SCC": "scpd_own"}
NSHIFT = 1000
HITS_N = {"UCI": 10, "USLegis": 2, "canVote": 2}
TOP_K = {"MAG": 9, "FL": 2, "SCC": 8}
PM_TOL, PM_REPS = 5, 10


# ------------------------------------------------------------------------------------------------ inputs
def rows_of(ID: str, T: int) -> dict:
    z = np.load(BENCH / f"scores_{ID}.npz")
    S = {k[2:]: z[k] for k in z.files if k.startswith("s_")}
    p = BENCH / "g0_rows.npz"
    if ID in G0 and p.exists():
        g = np.load(p)
        k = f"{G0[ID]}_{ID}"
        if k in g.files and len(g[k]) == T:
            S[G0[ID]] = np.asarray(g[k], float)
    return S


def meta_json(ID: str) -> dict:
    p = BENCH / f"{ID}_meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


# -------------------------------------------------------------------------- source-paper metric (m1)
def ranked_top(x, ok, n):
    idx = np.flatnonzero(ok & np.isfinite(x))
    return idx[np.argsort(-x[idx], kind="stable")][:n] if idx.size else None


def hits(x, seg, burn, events, n, tol=0, two_sided=False, constant=False):
    """Share of the events with a top-n step in [e, e + tol] (or [e - tol, e + tol]); exact index match at tol 0.

    constant: a random ranking of the scored steps, so the expected Hits@n is min(n, N) / N per scored event."""
    ok = ev.scored_mask(seg, x, burn)
    if not ok.any():
        return None
    if constant:
        N = int(ok.sum())
        inside = [e for e in events if any(ok[max(0, e - (tol if two_sided else 0)):e + tol + 1])]
        return float(min(n, N) / N * len(inside) / len(events)) if tol == 0 else None
    top = ranked_top(x, ok, n)
    return sum(int(any((e - tol if two_sided else e) <= d <= e + tol for d in top)) for e in events) / len(events)


def alarm_rows(S, seg, st, burn):
    """The fixed alarm rules of the selection series (steps 3, 8, 9; SCPD has none)."""
    out = {}
    s, x = st["counts"], S["counts"]
    th = theta_of(x, s[4], s[5])
    out["counts"] = [s0 + a for s0, s1 in ev.seg_ranges(seg) for a in alarms(x[s0:s1], th, s[3], window_burn(s))]
    s = st["constant"]
    out["constant"] = [s0 + t for s0, s1 in ev.seg_ranges(seg) for t in range(window_burn(s), s1 - s0, s[3])]
    thr = {m: (st[m]["threshold"], st[m]["cooldown"]) for m in ("cusum", "bocpd", "mmd", "lad")}
    thr["tire"] = (st["tire"][1], st["tire"][2])
    for m, (th, cd) in thr.items():
        out[m] = [s0 + a for s0, s1 in ev.seg_ranges(seg)
                  for a in apply_cooldown(np.nan_to_num(S[m][s0:s1], nan=-np.inf), th, cd, start=burn[m])]
    return out


def unit(x):
    """[0, 1] rescaling of a statistic (DyNNet evaluate_baseline); unscored steps get the minimum."""
    x = np.asarray(x, float)
    x = np.where(np.isfinite(x), x, np.nanmin(x))
    rg = x.max() - x.min()
    return (x - x.min()) / rg if rg > 0 else np.zeros_like(x)


def m1_pm(S, mt, rows) -> dict:
    """s-GNN protocol on PM for the unsupervised rows; the s-GNN-I row itself comes from step 13."""
    seg, evs = mt["seg"], mt["events"]
    out = {r: np.full((8, PM_REPS), np.nan) for r in rows + ["constant_all_alarm"]}
    for s, (a, b) in enumerate(ev.seg_ranges(seg)):
        T = b - a
        cps = [e - a for e in evs if a <= e < b]
        flags = np.zeros(T, int)
        flags[cps] = 1
        lab = np.cumsum(flags)
        for r in range(PM_REPS):
            tr, _, te = ev.individual_split(T, cps, np.random.default_rng([ev.SEED, s, r]))
            y_te, y_tr = ev.cp_targets(lab[te], PM_TOL), ev.cp_targets(lab[tr], PM_TOL)
            out["constant_all_alarm"][s, r] = ev.adjusted_f1(np.ones(len(te)), y_te, 0.5)
            for m in rows:
                x = S[m][a:b]
                out[m][s, r] = ev.adjusted_f1(unit(x[te]), y_te, ev.best_threshold(unit(x[tr]), y_tr))
    res = {"metric": "adjusted F1 (point-adjust), two-sided tolerance 5, individual split, 10 repetitions per subject",
           "values": {m: {"mean": float(np.nanmean(v)), "per_subject": np.nanmean(v, 1).round(4).tolist(),
                          "sd_over_reps_mean": float(np.nanmean(np.nanstd(v, 1))), "n_runs": int(np.isfinite(v).sum())}
                      for m, v in out.items()}}
    p = RESULTS / "bench_g0.json"
    g = json.loads(p.read_text()).get("sgnn") if p.exists() else None
    if g:
        res["values"]["s-GNN-I"] = {k: g[k] for k in ("mean", "per_subject", "n_runs")}
        res["values"]["s-GNN-I"]["sd_over_reps_mean"] = float(np.mean(np.std(g["per_run"], 1)))
        res["paper_values"] = {"s-GNN-I individual (per subject 101-108)": g["paper_per_subject"]}
        res["values"]["constant_all_alarm"] = res["values"].pop("constant_all_alarm")
    return res


def m1(ID, S, mt, st, burn) -> dict:
    meta = meta_json(ID)
    seg, evs = mt["seg"], mt["events"]
    rows = PRIMARY + [k for k in ("lad_own", "scpd_own") if k in S]
    res = {"source_paper": SOURCE[ID]}
    if SOURCE[ID] == "LAD":
        n = HITS_N[ID]
        res["metric"] = f"Hits@{n} (exact index match; LAD compute_accuracy)"
        res["paper_values"] = meta.get("paper_results", {}).get("values", {})
        res["values"] = {m: hits(S[m], seg, burn[m], evs, n, constant=(m == "constant")) for m in rows}
        if "events_secondary" in mt["sec"]:
            res["values_secondary_events"] = {m: hits(S[m], seg, burn[m], mt["sec"]["events_secondary"], n,
                                                      constant=(m == "constant")) for m in rows}
        res["n_scored"] = {m: int(ev.scored_mask(seg, S[m], burn[m]).sum()) for m in rows}
    elif SOURCE[ID] == "SCPD":
        k, tol = TOP_K[ID], mt["tol"]
        res["metric"] = (f"Hits@{len(evs)} (exact index); event recall within tol={tol} among the top {k} steps "
                         f"(k = the number of SCPD's published detections), one- and two-sided")
        res["paper_values"] = meta.get("paper_results", {})
        res["values"] = {m: {"hits_at_E": hits(S[m], seg, burn[m], evs, len(evs), constant=(m == "constant")),
                             f"recall_top{k}_one_sided": None if m == "constant" else
                             hits(S[m], seg, burn[m], evs, k, tol),
                             f"recall_top{k}_two_sided": None if m == "constant" else
                             hits(S[m], seg, burn[m], evs, k, tol, True)} for m in rows}
        scpd_pub = meta.get("paper_results", {}).get("SCPD", {})
        pub = {"MAG": [y - 1837 for y in scpd_pub.get("detections_years", [])],
               "FL": scpd_pub.get("detections_week_index", [])}.get(ID)
        if pub:
            res["published_scpd_detections_index"] = pub
            top = {m: ranked_top(S[m], ev.scored_mask(seg, S[m], burn[m]), k) for m in rows}
            res["overlap_with_published_scpd_list"] = {
                m: None if m == "constant" or top[m] is None else len(set(top[m].tolist()) & set(pub))
                for m in rows}
    elif ID in REPO_VIEW:
        ds = REPO_VIEW[ID]
        res["metric"] = f"Table 1 F1 (one-sided tol {mt['tol']}), fixed alarm rules"
        res["values"] = {m: {"f1": f1_score(a, evs, mt["tol"])[0], "n_alarms": len(a)}
                         for m, a in alarm_rows(S, seg, st, burn).items()}
        bl = json.loads((RESULTS / "baselines.json").read_text())
        res["repo_reported_fixed_f1"] = {m: bl[m]["fixed"]["per_dataset"][ds] for m in ("cusum", "bocpd", "mmd", "lad")}
        tire = json.loads((RESULTS / "tire.json").read_text())
        res["repo_reported_fixed_f1"]["tire"] = tire["fixed"]["per_dataset"][ds]
        res["note"] = ("SCPD has no fixed alarm threshold, so it has no F1 here; "
                       "Peel & Clauset report no numeric metric on these series")
    elif ID == "PM":
        res.update(m1_pm(S, mt, ["counts", "cusum", "bocpd", "mmd", "lad", "scpd", "tire"]))
    else:
        res["metric"] = "no source paper reports a metric on this series (M2 only)"
    return res


# ----------------------------------------------------------------------------- event percentile (m2)
def pooled_block(dss, ep_vals, nulls, per):
    blk = {"ep": {}, "contrast_counts_minus": {}, "beats_constant_per_ds": {}, "shift_p_pooled": {}}
    for m in PRIMARY:
        if all(m in ep_vals[d] for d in dss):
            b = ev.boot_pooled({d: ep_vals[d][m] - 0.5 for d in dss})
            blk["ep"][m] = {"mean": b["mean"] + 0.5, "lo": b["lo"] + 0.5, "hi": b["hi"] + 0.5,
                            "per_ds": {d: [v + 0.5 for v in b["per_ds"][d]] for d in dss}}
            obs = np.mean([ep_vals[d][m].mean() for d in dss])
            blk["shift_p_pooled"][m] = ev.shift_p(np.mean([nulls[d][m] for d in dss], axis=0), obs)
    for m in PRIMARY[1:]:
        if all(m in ep_vals[d] for d in dss):
            blk["contrast_counts_minus"][m] = ev.boot_pooled({d: ep_vals[d]["counts"] - ep_vals[d][m] for d in dss})
    for d in dss:
        blk["beats_constant_per_ds"][d] = {m: blk["ep"][m]["per_ds"][d][1] > 0.5 for m in blk["ep"] if m != "constant"}
    return blk


def evaluate_all() -> dict:
    st = settings()
    burn = {**st["burn"], "lad_own": 0, "scpd_own": 0}
    per, ep_vals, nulls = {}, {}, {}
    for ID in B:
        mt = meta(ID)
        S = rows_of(ID, mt["T"])
        seg, evs, tol = mt["seg"], mt["events"], mt["tol"]
        mask = ev.common_mask(S, burn, PRIMARY, seg)
        rows = PRIMARY + [k for k in ("lad_own", "scpd_own") if k in S]
        ok_rows = [m for m in rows if np.isfinite(S[m][mask]).all()]
        ev_ok, null = ev.windows(mask, seg, evs, evs, tol + 1)
        r = {"T": mt["T"], "n_segments": len(ev.seg_ranges(seg)), "n_events": len(evs), "tol": tol,
             "source_paper": SOURCE[ID], "n_mask": int(mask.sum()), "n_events_evaluable": len(ev_ok),
             "n_null_windows": len(null), "under_10_events": len(ev_ok) < 10, "m1": m1(ID, S, mt, st, burn)}
        per[ID] = r
        if not ev_ok or not len(null):
            r["m2"] = {"status": "no evaluable event: every event lies before the common burn-in "
                                 f"({max(burn[m] for m in PRIMARY)} steps) or no null window"}
            continue
        E = {}
        for m in ok_rows:
            v, _ = ev.ep(S[m], mask, seg, evs, tol)
            E[m] = np.array([v[e] for e in ev_ok])
        ep_vals[ID] = E
        rng = np.random.default_rng([ev.SEED, B.index(ID)])
        nulls[ID] = ev.shift_null(S, ok_rows, mask, seg, evs, tol, rng, NSHIFT)
        r["m2"] = {"events": [int(e) for e in ev_ok], "ep_mean": {m: float(E[m].mean()) for m in ok_rows},
                   "ep_per_event": {m: E[m].round(4).tolist() for m in ok_rows},
                   "shift_p": {m: ev.shift_p(nulls[ID][m], E[m].mean()) for m in ok_rows}}
        print(ID, {m: round(float(E[m].mean()), 3) for m in ok_rows}, flush=True)
    dss = sorted(ep_vals)
    big = [d for d in dss if not per[d]["under_10_events"]]
    out = {"datasets": per, "pooled_datasets": dss, "pooled_datasets_10plus": big, "B_boot": ev.NBOOT,
           "n_shift": NSHIFT, "settings": {**{k: v for k, v in st.items() if k != "key"},
                                           "common_burn": max(burn[m] for m in PRIMARY)},
           "pooled": pooled_block(dss, ep_vals, nulls, per),
           "pooled_10plus": pooled_block(big, ep_vals, nulls, per) if big else None,
           "own_rows": {d: {m: {"ep_mean": float(ep_vals[d][m].mean()),
                                "minus_counts": ev.boot_pooled({d: ep_vals[d][m] - ep_vals[d]["counts"]})["per_ds"][d]}
                            for m in ("lad_own", "scpd_own") if m in ep_vals.get(d, {})} for d in dss}}
    return out


# ------------------------------------------------------------------------------------------------ summary
NAME = {"counts": "counts", "constant": "constant", "cusum": "CUSUM", "bocpd": "BOCPD", "mmd": "MMD",
        "lad": "LAD (repo)", "scpd": "SCPD (port, stored graphs)", "tire": "TIRE (repo)",
        "lad_own": "LAD, authors' code + construction (G0)", "scpd_own": "SCPD, own construction (G0)"}


def f(v, d=2):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, float) and not np.isfinite(v):
        return "n/a"
    return f"{v:.{d}f}"


def table(head, rows):
    s = "| " + " | ".join(head) + " |\n|" + "|".join("---" for _ in head) + "|\n"
    return s + "".join("| " + " | ".join(str(c) for c in r) + " |\n" for r in rows)


def summary(R) -> str:
    P = R["datasets"]
    rows_all = PRIMARY
    md = ["## 1. Datasets\n", table(
        ["ID", "source paper", "T", "segments", "events", "evaluable events (M2)", "< 10 events", "tol",
         "null windows"],
        [[d, P[d]["source_paper"], P[d]["T"], P[d]["n_segments"], P[d]["n_events"], P[d]["n_events_evaluable"],
          f(P[d]["under_10_events"]), P[d]["tol"], P[d]["n_null_windows"]] for d in B]),
        f"\nCommon burn-in for M2: {R['settings']['common_burn']} steps per segment.\n", "## 2. M1, LAD datasets\n"]
    lad_ds = ["UCI", "USLegis", "canVote"]
    rr = [[NAME[m]] + [f(P[d]["m1"]["values"].get(m)) + (f" (scored {P[d]['m1']['n_scored'].get(m)})"
                                                         if m in P[d]["m1"]["n_scored"] else "") for d in lad_ds]
          + [f(P["canVote"]["m1"].get("values_secondary_events", {}).get(m))] for m in rows_all + ["lad_own"]]
    rr += [[f"paper: {k}"] + [f(P[d]["m1"]["paper_values"].get(k)) for d in lad_ds] + ["n/a"]
           for k in ("LAD", "EdgeMonitoring", "Activity vector", "TENSORSPLAT")]
    md.append(table(["row", "UCI Hits@10", "USLegis Hits@2", "canVote Hits@2 (events 2013, 2015)",
                     "canVote Hits@2 (secondary 2011, 2015)"], rr))
    md.append("## 3. M1, SCPD datasets (SCPD reports no number on these)\n")
    sc_ds = ["MAG", "FL", "SCC"]
    rr = []
    for m in rows_all + ["scpd_own"]:
        line = [NAME[m]]
        for d in sc_ds:
            v = P[d]["m1"]["values"].get(m)
            line += ["-"] * 3 if v is None else [f(x) for x in v.values()]
        rr.append(line + [f(P[d]["m1"].get("overlap_with_published_scpd_list", {}).get(m)) for d in ("MAG", "FL")])
    md.append(table(["row"] + [f"{d} {c}" for d in sc_ds for c in ("Hits@|E|", "recall@top-k 1-sided",
                                                                  "recall@top-k 2-sided")]
                    + ["MAG overlap with SCPD's 9 published years", "FL overlap with SCPD's 2 published weeks"], rr))
    pm = P["PM"]["m1"]
    md.append("## 4. M1, PM (s-GNN protocol: adjusted F1, +-5, individual split, 10 repetitions per subject)\n")
    rr = [[NAME.get(m, m), f(v["mean"]), " ".join(f(x) for x in v["per_subject"]), v["n_runs"]]
          for m, v in pm["values"].items()]
    if pm.get("paper_values"):
        p = pm["paper_values"]["s-GNN-I individual (per subject 101-108)"]
        rr.append(["paper: s-GNN-I individual", f(float(np.mean(p))), " ".join(f(x) for x in p), "-"])
    md.append(table(["row", "mean adjusted F1", "per subject 101..108", "runs"], rr))
    md.append("## 5. M1, repo Table 1 series (F1, one-sided tolerance, fixed alarm rules)\n")
    rp = ["MIT", "ENRON", "AskUbuntu"]
    rep = P["MIT"]["m1"]["repo_reported_fixed_f1"]
    md.append(table(["row", "MIT", "ENRON", "AskUbuntu", "repo baselines.json fixed F1 (MIT / ENRON / AskUbuntu)"],
                    [[NAME.get(m, m)] + [f"{f(P[d]['m1']['values'][m]['f1'])} ({P[d]['m1']['values'][m]['n_alarms']} "
                                         "alarms)" for d in rp]
                     + [" / ".join(f(P[d]["m1"]["repo_reported_fixed_f1"].get(m)) for d in rp) if m in rep else "-"]
                     for m in P["MIT"]["m1"]["values"]]))
    md.append("## 6. M2, event percentile EP per dataset (mean over evaluable events; constant = 0.50)\n")
    md.append(table(["row"] + [f"{d} (n={P[d]['n_events_evaluable']})" for d in B],
                    [[NAME[m]] + [f(P[d]["m2"]["ep_mean"].get(m)) + f" (p={f(P[d]['m2']['shift_p'].get(m), 3)})"
                                  if "ep_mean" in P[d]["m2"] else "n/a" for d in B] for m in rows_all]))
    for key, title in (("pooled", "all datasets with evaluable events"),
                       ("pooled_10plus", "datasets with >= 10 evaluable events")):
        blk = R[key]
        if not blk:
            continue
        used = R["pooled_datasets" if key == "pooled" else "pooled_datasets_10plus"]
        md.append(f"## 7{'a' if key == 'pooled' else 'b'}. Pooled EP, equal weight per dataset ({title}: "
                  f"{', '.join(used)})\n")
        md.append(table(["row", "pooled EP", "95% CI", "pooled shift p"],
                        [[NAME[m], f(v["mean"], 3), f"[{f(v['lo'], 3)}, {f(v['hi'], 3)}]",
                          f(blk["shift_p_pooled"][m], 4)] for m, v in blk["ep"].items()]))
        md.append(f"\nPaired contrasts EP(counts) - EP(row), {R['B_boot']} bootstrap draws:\n")
        md.append(table(["vs", "pooled diff", "95% CI"] + used,
                        [[NAME[m], f(b["mean"], 3), f"[{f(b['lo'], 3)}, {f(b['hi'], 3)}]"]
                         + [f"{f(b['per_ds'][d][0])} [{f(b['per_ds'][d][1])}, {f(b['per_ds'][d][2])}]" for d in used]
                         for m, b in blk["contrast_counts_minus"].items()]))
        md.append("\nRows whose per-dataset EP CI lies above 0.50 (beats the constant score):\n")
        md.append(table(["dataset", "rows with CI lower bound > 0.50"],
                        [[d, ", ".join(NAME[m] for m, v in blk["beats_constant_per_ds"][d].items() if v) or "none"]
                         for d in used]))
    rr = [[d, NAME[m], f(v["ep_mean"], 3),
           f"{f(v['minus_counts'][0])} [{f(v['minus_counts'][1])}, {f(v['minus_counts'][2])}]"]
          for d, mm in R["own_rows"].items() for m, v in mm.items()]
    if rr:
        md.append("\n## 8. Source papers' own constructions (G0 rows, descriptive, not pooled)\n")
        md.append(table(["dataset", "row", "EP", "EP(row) - EP(counts) [95% CI]"], rr))
    return "\n".join(md)


def main() -> None:
    p = RESULTS / "bench.json"
    if "--summary" in sys.argv and p.exists():
        print(summary(json.loads(p.read_text())))
        return
    t0 = Timer()
    out = evaluate_all()
    write_json(p, out)
    print(f"step 12: {t0():.0f} s -> {p}")
    if "--summary" in sys.argv:
        print(summary(json.loads(p.read_text())))


if __name__ == "__main__":
    main()
