"""Step 23: inference over every benchmark row -> results/inference.json (spec: written before the run).

Rows: the pooled rows of step 12 (rows.PRIMARY), the identity-aware rows and the vertex turnover of step 16
(repro/data/identity/scores_<ID>.npz, scores_new_<ID>.npz), the wiring rows of step 19 (repro/data/activity/
scores_<ID>.npz) and the causal count row of step 22 (repro/data/settings/scores_<ID>.npz: the counts with the online
scale, and four count reference rows). The causal counts and the turnover are reference rows: they get intervals
but are not members of the Holm families; the causal counts also get ranks (in place of the counts); the turnover and
the count reference rows are outside the rank sets. EP per event on step 12's common mask (evaluate.ep, unchanged). Check S0: the core rows reproduce
bench.json, the identity and wiring rows identity.json and activity.json.
Parts (inference.py; every bootstrap paired: one resample serves every row; B = 10,000, seed 20261023)
  A  pooled EP of every row in every pool, equal and event weight, intervals at three levels: event (events within
     datasets: conditional on the datasets), two-stage (datasets, then events), dataset (datasets only); for the
     dataset level also the t_{D-1} interval and the exact sign-flip test over datasets
  B  counts (and the causal counts) minus every row: bootstrap p, the minimum detectable lead at 80% power (MDE80,
     from the two-stage SD of the contrast), Holm within family F1 (the permutation-invariant detectors) and F2 (F1
     + the identity-aware detectors + the wiring statistics), Bonferroni-simultaneous intervals; rank of every row in
     every draw, P(rank 1); the set of rows not excluded from being best (inference.best_set); the smallest
     attainable exact sign-flip p over the datasets and its Holm-adjusted value
  T  vertex turnover (spec: prereg/TURNOVER.md, written before the run): ASE and Hamming minus the
     turnover and minus the causal counts on the >= 10-event pool (two-stage) and on HT09, SCC, PM, EEG (events)
  C  per dataset: counts minus every row, event-level percentile interval and the exact sign-flip test and interval
  D  coverage at mean 0 of the percentile bootstrap and of the sign-flip interval for 2 to 20 events; coverage of the
     pooled intervals in a super-population of datasets (inference.coverage_pooled)
  E  Peel-Clauset constructions (step 17 data): the ten plus MIT_PC and ENRON_PC, and with them replacing MIT, ENRON
  F  detection within tol at fixed null quantiles (evaluate.delay; alpha 0.01, 0.02, 0.05; H = 3 tol) of every row,
     pooled with paired intervals
  G  protocol variants of every row: guard band of 17 steps, guard band of each row's w + w_ref (spec v5 G), one alignment (past-only rows scored from e + 1),
     negatives within the event's segment, event dates jittered by -2..+2 steps
  H  per-case EP on the benchmark EEG (CHB-MIT cases 1, 3, 5)
  L  labels (spec v4 L2): SCC with the audit-corrected events (prereg/scc_audit_events.json) in
     the pools of all ten and >= 10 events, and the turnover check with them and without SCC
  V  count reference rows (spec v4 V3; series of step 22 E): the counts at w = 2, w_ref = 10, the causal edge count,
     the counts with an expanding past scale; the >= 10-event gaps of ASE, Hamming, NBS, TIRE, turnover over each
     causal variant (two-stage, t over datasets); the best causal variant by its pooled EP on all ten
  G2 tolerance curve (spec v4 V1): one tau = 0..10 on every dataset, every row, all ten and >= 10 events
  LAD LAD with its code's eigenvalue counts, 6 and 499 (spec v5 M; series of step 22 M) beside LAD with 5
  P  per-event association of every row with the counts (spec v5 A): Spearman rho, share of events above the counts
  N  datasets (spec v4 V4): T, segments, tol, scored events, null windows, vertex-set size; t_{D-1} interval of
     every row's EP over datasets (dataset_ep_t); coverage of the pooled intervals at K = 4 (spec v4 V5)
  K  event types (spec v6 T): every scored event tagged from its description (TAG_A), EP of every row on the
     activity-by-design events and on the others, the source four and the held-out seven without them; the large or
     extreme events (step 19) of each type; clusters
     (spec v6 C): the pools all ten and >= 10 events with a three-stage bootstrap (PM subjects, EEG cases; K2 also
     the days of PS and HT09)
  U  simultaneous one-sided 95% upper bounds of row - counts and row - causal counts over F2 (spec v6 U)
Part A also gives Holm over the dataset-level t p (t_holm) beside the sign-flip Holm (spec v6 P); the count reference
rows include the edge count alone with the whole-series scale (counts_edges, spec v6 E).
Pools: all ten; >= 10 events; >= 10 events without EEG; held-out 7 (no F1-selection set: MIT, ENRON, AskUbuntu);
without the source detectors' own detections as labels (MAG); the four source-paper datasets with their own
construction and events (UCI, MAG, SCC, PM) and the other six; interaction (8) and correlation (PM, EEG) networks;
without UCI; without SCC (all ten and >= 10 events); each dataset left out.

Usage: 23_inference.py [--summary]
"""
from __future__ import annotations

import json
import sys

import evaluate as ev
import inference as inf
import numpy as np
from common import DATA, RESULTS, ROOT, Timer, load, write_json
from rows import BENCH, PRIMARY, REPO_VIEW, B, meta, settings

IDENTITY = ["hamming", "jaccard", "deltacon", "chenzhang", "nbs", "ase", "spotlight"]
WIRING = ["transitivity", "assortativity", "fair_pair"]
F1 = ["cusum", "bocpd", "mmd", "lad", "scpd", "tire"]
F2 = F1 + IDENTITY + WIRING
FAMILIES = (("F1", F1), ("F2", F2))
CAUSAL, TURN = "counts_causal", "turnover"
COUNT_VARIANTS = ["counts_w2_10", "counts_causal_edges", "counts_causal_expanding", "counts_edges"]   # step 22 E
LAD_K = ["lad_k6", "lad_k499"]                        # step 22 M; spec v5 M: LAD with its code's eigenvalue counts
CAUSAL_VARIANTS = [CAUSAL, "counts_causal_edges", "counts_causal_expanding"]
REFS = ("counts", CAUSAL)
UNRANKED = ("constant", TURN, *COUNT_VARIANTS, *LAD_K)   # reference rows outside the rank and best sets
ROWS = PRIMARY + IDENTITY + WIRING + [CAUSAL, TURN] + COUNT_VARIANTS + LAD_K
# w + w_ref of each row (spec v5 G): the steps after a change in which its score still sees it; CUSUM and BOCPD have
# no window (their burn-in)
SPAN = {"counts": 7, "constant": 0, "cusum": 10, "bocpd": 15, "mmd": 6, "lad": 17, "scpd": 15, "tire": 10,
        **{r: 12 for r in IDENTITY + WIRING}, "spotlight": 11, CAUSAL: 7, TURN: 7, "counts_w2_10": 12,
        "counts_causal_edges": 7, "counts_causal_expanding": 7, "counts_edges": 7, **{r: 17 for r in LAD_K}}
SELECTION, POSTHOC, CORR = ("MIT", "ENRON", "AskUbuntu"), ("MAG",), ("PM", "EEG")
SOURCE = ("UCI", "MAG", "SCC", "PM")                  # the source paper's construction and events
TAU0 = ("UCI",)                                       # tol 0: a past-only score cannot see G_e inside [e, e]
G_T = ("cusum", "bocpd", "lad", "scpd", "tire", *LAD_K)   # rows that see G_t (TIRE: offline, both sides)
PC = {"MIT_PC": "MIT", "ENRON_PC": "ENRON"}
LEVELS, WEIGHTS = ("event", "two_stage", "dataset"), ("equal", "event")


# ------------------------------------------------------------------------------------------------ per-event EP
def series(ID: str) -> dict:
    S = {k[2:]: v for k, v in np.load(BENCH / f"scores_{ID}.npz").items() if k.startswith("s_")}
    for p in (DATA / "identity" / f"scores_{ID}.npz", DATA / "identity" / f"scores_new_{ID}.npz",
              DATA / "activity" / f"scores_{ID}.npz", DATA / "settings" / f"scores_{ID}.npz",
              DATA / "settings" / f"lad_{ID}.npz"):
        if p.exists():
            S.update({k[2:]: np.asarray(v, float) for k, v in np.load(p).items() if k.startswith("s_")})
    return S


def per_event(ID: str, mask_rows, burn, events=None) -> tuple[dict, dict, dict]:
    """({row: EP per evaluable event}, {row: series}, context) on step 12's common mask (events: the stored ones,
    or a label variant)."""
    mt = meta(ID)
    if events is not None:
        mt["events"] = list(events)
    S = series(ID)
    mask = ev.common_mask(S, burn, mask_rows, mt["seg"])
    ev_ok, _ = ev.windows(mask, mt["seg"], mt["events"], mt["events"], mt["tol"] + 1)
    out = {}
    for r, x in S.items():
        if r != "seg" and np.isfinite(x[mask]).all():
            v, _ = ev.ep(x, mask, mt["seg"], mt["events"], mt["tol"])
            out[r] = np.array([v[e] for e in ev_ok])
    return out, S, {**mt, "mask": mask, "events": ev_ok}


def check_s0(E, bench, ident, act) -> dict:
    core = PRIMARY
    dev = {"core_per_event_4dp": 0.0, "core_pooled": 0.0, "identity_4dp": 0.0, "wiring_4dp": 0.0}
    for d in bench["pooled_datasets"]:
        ref = bench["datasets"][d]["m2"]["ep_per_event"]
        dev["core_per_event_4dp"] = max(dev["core_per_event_4dp"], max(
            float(np.max(np.abs(np.round(E[d][m], 4) - ref[m]))) for m in core))
        ri = ident["datasets"][d]["ep_per_event"]
        dev["identity_4dp"] = max(dev["identity_4dp"], max(
            float(np.max(np.abs(np.round(E[d][m], 4) - ri[m]))) for m in ri if m != "counts"))
        ra = act["datasets"][d]["ep_per_event"]
        dev["wiring_4dp"] = max(dev["wiring_4dp"], max(
            float(np.max(np.abs(np.round(E[d][m], 4) - ra[m]))) for m in WIRING if m in ra))
    dev["core_pooled"] = max(abs(float(np.mean([E[d][m].mean() for d in bench["pooled_datasets"]]))
                                 - bench["pooled"]["ep"][m]["mean"]) for m in core)
    dev["pass"] = dev["core_pooled"] < 1e-12 and max(v for k, v in dev.items() if k.endswith("4dp")) < 1e-12
    return dev


# ------------------------------------------------------------------------------------------------ A, B: pools
def _contrasts(dr, pt, rows, ref, families) -> dict:
    """ref minus every other row: diff, CI, bootstrap p, Holm and Bonferroni-simultaneous intervals per family."""
    ic = rows.index(ref)
    q = {}
    for i, r in enumerate(rows):
        if i != ic:
            d = dr[:, ic] - dr[:, i]
            q[r] = {"diff": float(pt[ic] - pt[i]), "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
                    "p": inf.pval(d), "mde80": inf.mde80(float(np.std(d)))}
    for fam, members in families:
        mem = [r for r in members if r in rows and r != ref]
        a = 0.05 / len(mem)
        for r, p in zip(mem, inf.holm([q[r]["p"] for r in mem])):
            d = dr[:, ic] - dr[:, rows.index(r)]
            q[r][f"p_holm_{fam}"] = p
            q[r][f"ci_simultaneous_{fam}"] = [float(np.quantile(d, a / 2)), float(np.quantile(d, 1 - a / 2))]
    return q


def _ranks(dr, pt, rows, idx) -> dict:
    rk = inf.ranks(dr[:, idx])
    return {rows[i]: {"point": int(np.sum(pt[idx] > pt[i]) + 1), "median": float(np.median(rk[:, j])),
                      "ci95": [int(np.quantile(rk[:, j], 0.025)), int(np.quantile(rk[:, j], 0.975))],
                      "p_rank1": float(np.mean(rk[:, j] == 1))} for j, i in enumerate(idx)}


def pool_block(E: dict, dss, rows, levels=LEVELS, weights=WEIGHTS, families=FAMILIES, best=False,
               clusters=None) -> dict:
    X = {d: np.stack([E[d][r] for r in rows], 1) for d in dss}
    refs = [r for r in REFS if r in rows]
    sets = {ref: [i for i, r in enumerate(rows) if r not in UNRANKED and (r not in REFS or r == ref)]
            for ref in refs}                            # rank sets: the detectors and one of the two count rows
    out = {"datasets": list(dss), "n_events": int(sum(len(X[d]) for d in dss))}
    for w in weights:
        pt = inf.pooled_point(X, w)
        blk = {"ep": {r: float(pt[i]) for i, r in enumerate(rows)}}
        for lv in levels:
            dr = inf.pooled_draws(X, lv, w, clusters=clusters)
            q = {"ep_ci95": {r: [float(np.quantile(dr[:, i], 0.025)), float(np.quantile(dr[:, i], 0.975))]
                             for i, r in enumerate(rows)}}
            for ref in refs:
                key = "counts_minus" if ref == "counts" else f"{ref}_minus"
                q[key] = _contrasts(dr, pt, rows, ref, families)
            for name, idx in sets.items():
                q["rank" if name == "counts" else f"rank_{name}"] = _ranks(dr, pt, rows, idx)
                if best:
                    bs = inf.best_set(dr[:, idx], pt[idx])
                    q["best_set" if name == "counts" else f"best_set_{name}"] = {
                        "c": bs["c"], "not_excluded": [rows[idx[i]] for i in bs["not_excluded"]]}
            blk[lv] = q
        if w == "equal" and len(dss) > 2:               # small-sample dataset-level inference
            mu = {r: np.array([E[d][r].mean() for d in dss]) for r in rows}
            blk["dataset_ep_t"] = {r: inf.t_interval(mu[r]) for r in rows}
            blk["dataset_small_n"] = {ref: {r: {"t": inf.t_interval(mu[ref] - mu[r]),
                                                "signflip": inf.signflip(mu[ref] - mu[r])}
                                            for r in rows if r != ref} for ref in refs}
        out[w] = blk
    return out


def dataset_tests(blk, dss) -> None:
    """Holm over the dataset-level sign-flip p and over the t p (families F1, F2), and the sign-flip power limit."""
    for ref in REFS:
        sm = blk["equal"].get("dataset_small_n", {}).get(ref)
        if sm:
            sfx = "" if ref == "counts" else f"_{ref}"
            for test in ("signflip", "t"):
                blk["equal"][f"{test}_holm{sfx}"] = {
                    fam: dict(zip(mem, inf.holm([sm[r][test]["p"] for r in mem]))) for fam, mem in FAMILIES}
    D = len(dss)                                        # the power limit of the exact sign-flip test over datasets
    blk["equal"]["signflip_limit"] = {"n_datasets": D, "min_p": min(1.0, 2 / 2 ** D),
                                      **{f"min_p_holm_{f}": min(1.0, len(m) * 2 / 2 ** D) for f, m in FAMILIES}}


def pools(E, bench) -> dict:
    alln = bench["pooled_datasets"]
    P = {"all": alln, "10plus": bench["pooled_datasets_10plus"],
         "10plus_noEEG": [d for d in bench["pooled_datasets_10plus"] if d != "EEG"],
         "heldout7": [d for d in alln if d not in SELECTION], "no_posthoc": [d for d in alln if d not in POSTHOC],
         "source4": [d for d in alln if d in SOURCE], "own_added6": [d for d in alln if d not in SOURCE],
         "interaction": [d for d in alln if d not in CORR], "correlation": [d for d in alln if d in CORR],
         "all_noUCI": [d for d in alln if d not in TAU0], "interaction_noUCI": [d for d in alln if d not in CORR + TAU0],
         "all_noSCC": [d for d in alln if d != "SCC"],
         "10plus_noSCC": [d for d in bench["pooled_datasets_10plus"] if d != "SCC"]}
    rows = ROWS
    out = {}
    for k, dss in P.items():
        out[k] = pool_block(E, dss, rows, best=k in ("all", "10plus", "interaction", "source4", "all_noUCI", "all_noSCC",
                                                     "10plus_noSCC"))
        dataset_tests(out[k], dss)
        print(f"pool {k}: counts {out[k]['equal']['ep']['counts']:.3f}", flush=True)
    out["drop_one"] = {d: pool_block(E, [x for x in alln if x != d], rows, levels=("event", "two_stage"),
                                     weights=("equal",)) for d in alln}
    return out


# ------------------------------------------------------------------------------------------------ T: turnover
def turnover_check(E, big) -> dict:
    """ASE and Hamming minus the turnover and minus the causal counts: the >= 10-event pool `big` (two-stage, equal
    weight) and each of its datasets (events within the dataset). Reading rule: prereg/TURNOVER.md."""
    pairs = [(a, b) for a in ("ase", "hamming") for b in (TURN, CAUSAL)] + [(CAUSAL, TURN)]
    X = {d: np.stack([E[d][a] - E[d][b] for a, b in pairs], 1) for d in big}
    pt, dr = inf.pooled_point(X), inf.pooled_draws(X, "two_stage")
    q = lambda d: [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]
    out = {"pool_10plus": {"datasets": big, **{f"{a} - {b}": {"diff": float(pt[i]), "ci95": q(dr[:, i]),
                                                              "p": inf.pval(dr[:, i])}
                                               for i, (a, b) in enumerate(pairs)}}}
    for d in big:
        out[d] = {}
        for a, b in pairs:
            x = E[d][a] - E[d][b]
            dd = ev.boot_draws(x)
            out[d][f"{a} - {b}"] = {"diff": float(x.mean()), "ci95": q(dd), "n_events": len(x)}
    lead = {a: out["pool_10plus"][f"{a} - {TURN}"]["ci95"] for a in ("ase", "hamming")}
    out["reading"] = ("matched by vertex turnover alone (activity)" if all(c[0] <= 0 <= c[1] for c in lead.values())
                      else "not explained by turnover; mechanism unknown")
    out["reading_per_row"] = {a: "covers 0" if c[0] <= 0 <= c[1] else "excludes 0" for a, c in lead.items()}
    return out


# ------------------------------------------------------------------------------------------------ L: labels
def label_variants(E, bench, burn) -> dict:
    """SCC with the audit-corrected labels (repro/prereg/scc_audit_events.json; spec v4 L2) in the pools of
    all ten and >= 10 events, and the turnover check with them and without SCC (pools without SCC: part A)."""
    aud = json.loads((ROOT / "prereg" / "scc_audit_events.json").read_text())["events"]
    Ea = {**E, "SCC": per_event("SCC", PRIMARY, burn, events=aud)[0]}
    big = bench["pooled_datasets_10plus"]
    out = {"scc_audit": {"events": aud, "n_events_scored": len(Ea["SCC"]["counts"]),
                         "ep_scc": {r: float(Ea["SCC"][r].mean()) for r in ROWS},
                         "ep_scc_primary": {r: float(E["SCC"][r].mean()) for r in ROWS},
                         "pools": {k: pool_block(Ea, dss, ROWS, best=True)
                                   for k, dss in (("all", bench["pooled_datasets"]), ("10plus", big))},
                         "turnover": turnover_check(Ea, big)},
           "no_scc": {"turnover": turnover_check(E, [d for d in big if d != "SCC"])}}
    for k, b in out["scc_audit"]["pools"].items():
        print(f"SCC audit labels, pool {k}: counts {b['equal']['ep']['counts']:.3f}", flush=True)
    return out


# ------------------------------------------------------------------------------------------------ V: count variants
def count_variants(E, P) -> dict:
    """The count reference rows (spec v4 V3) from the pools of part A: EP on all ten and >= 10 events, ASE, Hamming,
    NBS and TIRE minus each causal variant on >= 10 events (two-stage; t over the 4 datasets), and the best causal
    variant, chosen by its pooled EP on all ten."""
    def ci(dr, i):
        return [float(np.quantile(dr[:, i], 0.025)), float(np.quantile(dr[:, i], 0.975))]
    out = {k: {r: {"ep": P[k]["equal"]["ep"][r], "two_stage": P[k]["equal"]["two_stage"]["ep_ci95"][r],
                   "event_weighted": P[k]["event"]["ep"][r]}
               for r in ["counts"] + CAUSAL_VARIANTS + [COUNT_VARIANTS[0], "counts_edges"]}
           for k in ("all", "10plus")}
    best = max(CAUSAL_VARIANTS, key=lambda r: P["all"]["equal"]["ep"][r])
    out["best_causal_all_ten"] = best
    dss = P["10plus"]["datasets"]
    gap = {}
    for ref in CAUSAL_VARIANTS:
        pairs = [(a, ref) for a in ("ase", "hamming", "nbs", "tire", TURN)]
        X = {d: np.stack([E[d][a] - E[d][b] for a, b in pairs], 1) for d in dss}
        pt, dr = inf.pooled_point(X), inf.pooled_draws(X, "two_stage")
        gap[ref] = {f"{a} - {b}": {"diff": float(pt[i]), "ci95": ci(dr, i), "p": inf.pval(dr[:, i]),
                                   "t": inf.t_interval([X[d][:, i].mean() for d in dss])}
                    for i, (a, b) in enumerate(pairs)}
    out["gap_10plus"] = gap
    return out


# ------------------------------------------------------------------------------------------------ LAD: eigenvalues
def lad_rows(E, P) -> dict:
    """LAD with 6 and 499 eigenvalues (spec v5 M) beside LAD with 5: EP on all ten and >= 10 events, counts - row and
    row - lad (two-stage), per-dataset EP, and the Holm F2 p of counts - row with the row in place of lad."""
    def ci(d):
        return [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]
    out = {}
    for k in ("all", "10plus"):
        dss = P[k]["datasets"]
        rows = ["counts", "lad", *LAD_K]
        X = {d: np.stack([E[d][r] for r in rows], 1) for d in dss}
        pt, dr = inf.pooled_point(X), inf.pooled_draws(X, "two_stage")
        res = {r: {"ep": float(pt[i]), "two_stage": ci(dr[:, i]), "event_weighted": P[k]["event"]["ep"][r]}
               for i, r in enumerate(rows)}
        for i, r in enumerate(rows[1:], 1):
            res[r]["counts_minus"] = {"diff": float(pt[0] - pt[i]), "ci95": ci(dr[:, 0] - dr[:, i]),
                                      "p": inf.pval(dr[:, 0] - dr[:, i])}
            if r != "lad":
                res[r]["minus_lad"] = {"diff": float(pt[i] - pt[1]), "ci95": ci(dr[:, i] - dr[:, 1]),
                                       "p": inf.pval(dr[:, i] - dr[:, 1])}
                cm = P[k]["equal"]["two_stage"]["counts_minus"]
                fam = [x for x in F2 if x != "lad"] + [r]
                ps = [cm[x]["p"] for x in fam[:-1]] + [res[r]["counts_minus"]["p"]]
                res[r]["p_holm_F2_in_place_of_lad"] = inf.holm(ps)[-1]
        res["per_dataset"] = {d: {r: float(E[d][r].mean()) for r in rows} for d in dss}
        out[k] = res
    return out


# ------------------------------------------------------------------------------------------------ P: association
def association(E, P) -> dict:
    """Per-event association of every row with the counts (spec v5 A; descriptive): Spearman rho over the events of
    a pool and within datasets (>= 5 events), and the share of events where the row's EP exceeds the counts' (ties
    one half; equal weight per dataset with a two-stage interval, and event weight)."""
    from scipy.stats import spearmanr
    out = {}
    for k in ("all", "10plus", "interaction", "correlation"):
        dss = P[k]["datasets"]
        rows = [r for r in ROWS if r != "counts"]
        res = {}
        for r in rows:
            a = np.concatenate([E[d][r] for d in dss])
            c = np.concatenate([E[d]["counts"] for d in dss])
            within = [spearmanr(E[d][r], E[d]["counts"])[0] for d in dss if len(E[d]["counts"]) >= 5
                      and np.ptp(E[d][r]) > 0 and np.ptp(E[d]["counts"]) > 0]
            X = {d: ((E[d][r] > E[d]["counts"]) + 0.5 * (E[d][r] == E[d]["counts"]))[:, None] for d in dss}
            dr = inf.pooled_draws(X, "two_stage")[:, 0]
            res[r] = {"spearman_events": float(spearmanr(a, c)[0]) if np.ptp(a) > 0 else None,
                      "spearman_within": float(np.mean(within)) if within else None, "n_within": len(within),
                      "share_above": float(inf.pooled_point(X)[0]),
                      "share_above_ci95": [float(np.quantile(dr, 0.025)), float(np.quantile(dr, 0.975))],
                      "share_above_event_weighted": float(inf.pooled_point(X, "event")[0]),
                      "share_tied": float(np.mean([np.mean(E[d][r] == E[d]["counts"]) for d in dss]))}
        out[k] = {"datasets": dss, "n_events": P[k]["n_events"], "rows": res}
    return out


# ------------------------------------------------------------------------------------------------ G2: tolerance curve
def tau_curve(bench, burn) -> dict:
    """Every row at one tolerance tau = 0..10 on every dataset (spec v4 V1): pooled EP (equal weight), two-stage
    intervals, counts minus every row with Holm F2, P(counts rank 1); all ten and >= 10 events."""
    out = {}
    for tau in range(11):
        E = {}
        for ID in bench["pooled_datasets"]:
            mt, S = meta(ID), series(ID)
            mask = ev.common_mask(S, burn, PRIMARY, mt["seg"])
            v = {r: inf.ep_variant(S[r], mask, mt["seg"], mt["events"], tau) for r in ROWS}
            keep = sorted(set.intersection(*[set(x) for x in v.values()]))
            E[ID] = {r: np.array([v[r][e] for e in keep]) for r in ROWS}
        res = {}
        for k, pool in (("all", bench["pooled_datasets"]), ("10plus", bench["pooled_datasets_10plus"])):
            dss = [d for d in pool if len(E[d]["counts"])]      # a dataset without null windows at this tau drops out
            b = pool_block(E, dss, ROWS, levels=("two_stage",), weights=("equal",), families=(("F2", F2),))["equal"]
            ts = b["two_stage"]
            res[k] = {"datasets_dropped": [d for d in pool if d not in dss],
                      "n_events": int(sum(len(E[d]["counts"]) for d in dss)), "ep": b["ep"], "ci95": ts["ep_ci95"],
                      "counts_minus": {r: {"diff": v["diff"], "ci95": v["ci95"], "p_holm_F2": v.get("p_holm_F2")}
                                       for r, v in ts["counts_minus"].items()},
                      "p_rank1_counts": ts["rank"]["counts"]["p_rank1"],
                      "best_other": max((r for r in F2), key=lambda r: b["ep"][r])}
        out[tau] = res
        print(f"tau {tau}: counts {res['all']['ep']['counts']:.3f}, best {res['all']['best_other']} "
              f"{res['all']['ep'][res['all']['best_other']]:.3f}", flush=True)
    return out


# ------------------------------------------------------------------------------------------------ N: datasets
def datasets_table(E, bench, burn) -> dict:
    """Per pooled dataset (spec v4 V4): T, segments, tolerance, scored events, null windows on the common mask,
    vertex-set size of the stored series."""
    out = {}
    for ID in bench["pooled_datasets"]:
        mt, S = meta(ID), series(ID)
        mask = ev.common_mask(S, burn, PRIMARY, mt["seg"])
        ev_ok, null = ev.windows(mask, mt["seg"], mt["events"], mt["events"], mt["tol"] + 1)
        if ID in REPO_VIEW:
            n = np.asarray(load(REPO_VIEW[ID])["f"]["n"], int)
            nv = {"min": int(n.min()), "median": float(np.median(n)), "max": int(n.max())}
        else:
            nv = {"min": mt["n"], "median": mt["n"], "max": mt["n"]}
        out[ID] = {"T": mt["T"], "segments": len(np.unique(mt["seg"])), "tol": mt["tol"],
                   "events_labelled": len(mt["events"]), "events_scored": len(ev_ok), "null_windows": len(null),
                   "n_mask": int(mask.sum()), "vertex_set": nv, "check_E": len(E[ID]["counts"]) == len(ev_ok)}
    return out


# ------------------------------------------------------------------------------------------------ C, D
def per_dataset(E, bench) -> dict:
    out = {}
    for d in bench["pooled_datasets"]:
        rows = [r for r in ROWS if r != "counts" and r in E[d]]
        res = {"n_events": len(E[d]["counts"])}
        for r in rows:
            x = E[d]["counts"] - E[d][r]
            dr = ev.boot_draws(x)
            res[r] = {"diff": float(x.mean()), "ci95_percentile": [float(np.quantile(dr, 0.025)),
                                                                   float(np.quantile(dr, 0.975))],
                      "signflip": inf.signflip(x)}
        out[d] = res
    return out


def coverage(E, bench) -> dict:
    pool = np.concatenate([E[d]["counts"] - E[d]["tire"] for d in bench["pooled_datasets_10plus"]])
    big = bench["pooled_datasets_10plus"]
    return {"pool": "counts - TIRE per-event differences of the >= 10-event datasets, centred",
            "n_pool": len(pool), "reps": 4000, "B": 2000,
            "coverage": inf.coverage(pool, (2, 4, 5, 6, 9, 10, 20)),
            "pooled": {"contrast": "counts - TIRE, all ten, equal weight; dataset effects centred",
                       **inf.coverage_pooled({d: E[d]["counts"] - E[d]["tire"] for d in bench["pooled_datasets"]})},
            "pooled_10plus": {"contrast": "counts_causal - ASE, >= 10 events (4 datasets), equal weight; dataset "
                                          "effects centred (spec v4 V5)",
                              **inf.coverage_pooled({d: E[d][CAUSAL] - E[d]["ase"] for d in big})}}


# ------------------------------------------------------------------------------------------------ E: Peel-Clauset
def peel_clauset(E, bench, burn) -> dict:
    core = PRIMARY
    mask_rows = PRIMARY
    new = {}
    for ID in PC:
        if (BENCH / f"scores_{ID}.npz").exists():
            e, _, ctx = per_event(ID, mask_rows, burn)
            if ctx["events"]:
                new[ID] = e
    sets = {"published": list(bench["pooled_datasets"]), "plus_pc": list(bench["pooled_datasets"]) + list(new),
            "pc_replaces_mit_enron": [d for d in bench["pooled_datasets"] if d not in PC.values()] + list(new)}
    allE = {**E, **new}
    out = {"n_events": {k: len(v["counts"]) for k, v in new.items()}}
    for k, dss in sets.items():
        out[k] = pool_block(allE, dss, core, levels=("event", "two_stage"), families=(("F1", F1),))
    return out


# ------------------------------------------------------------------------------------------------ F: delay
ALPHAS = (0.01, 0.02, 0.05)


def bench_delay(bench, burn) -> dict:
    rows = ROWS
    out, hits = {}, {a: {} for a in ALPHAS}
    for ID in B:
        mt = meta(ID)
        S = series(ID)
        mask = ev.common_mask(S, burn, PRIMARY, mt["seg"])
        H = 3 * mt["tol"]
        res = {"tol": mt["tol"], "H": H}
        for alpha in ALPHAS:
            for m in rows:
                if m not in S:
                    continue
                try:
                    d = ev.delay(S[m], mask, mt["events"], H, alpha)
                except IndexError:                     # no scored step outside every [e, e + H]
                    res["no_null_steps"] = True
                    continue
                if not d:
                    continue
                v = np.array([d[e] for e in sorted(d)])
                hit = v <= H
                res.setdefault(f"alpha{alpha}", {})[m] = {
                    "n_events": len(v), "detected_within_H": float(hit.mean()),
                    "detected_within_tol": float((v <= mt["tol"]).mean()),
                    "median_delay_detected": float(np.median(v[hit])) if hit.any() else None}
                if ID in bench["pooled_datasets"]:
                    hits[alpha].setdefault(ID, {})[m] = (v <= mt["tol"]).astype(float)
        out[ID] = res
    pooled = {}
    for a in ALPHAS:
        k = f"alpha{a}"
        dss = [ID for ID in out if k in out[ID]]
        pooled[k] = {m: {"mean_detected_within_tol": float(np.mean([out[ID][k][m]["detected_within_tol"] for ID in dss
                                                                    if m in out[ID][k]])),
                         "n_datasets": sum(m in out[ID][k] for ID in dss)} for m in rows}
    inf_ = {}
    for a in ALPHAS:
        dss = sorted(ID for ID in hits[a] if all(m in hits[a][ID] for m in rows))
        Eh = {ID: hits[a][ID] for ID in dss}
        inf_[f"alpha{a}"] = pool_block(Eh, dss, rows, levels=("event", "two_stage"), weights=("equal",), best=True)
    return {"datasets": out, "pooled_equal_weight": pooled, "inference": inf_,
            "rule": "evaluate.delay: theta = (1 - alpha) quantile of the scored steps outside every [e, e + H], "
                    "H = 3 tol; delay = first step in [e, e + H] above theta; hit = delay <= tol"}


def fa_threshold_boot(bench, burn, alpha: float = 0.01, nb: int = 2000) -> dict:
    """Detection within tol at the (1 - alpha) null quantile, with the threshold redrawn in every resample: two-stage
    (datasets, then the events and the null steps of each drawn dataset, independently); paired over rows."""
    rows = ROWS
    D = {}
    for ID in bench["pooled_datasets"]:
        mt = meta(ID)
        S = series(ID)
        mask = ev.common_mask(S, burn, PRIMARY, mt["seg"])
        T, tol, H = mt["T"], mt["tol"], 3 * mt["tol"]
        busy = np.zeros(T, bool)
        for e in mt["events"]:
            busy[e:min(T, e + H + 1)] = True
        nul = np.flatnonzero(mask & ~busy)
        evs = [e for e in mt["events"] if e + H < T and mask[e:e + H + 1].all()]
        if not len(nul) or not evs or not all(r in S for r in rows):
            continue
        X = np.stack([np.asarray(S[r], float) + np.random.default_rng(0).random(T) * 1e-9 *
                      (np.nanstd(np.asarray(S[r], float)[mask]) + 1e-12) for r in rows], 1)     # as evaluate.delay
        D[ID] = (X[nul], np.stack([X[e:e + tol + 1].max(0) for e in evs]))
    names = sorted(D)

    def hits(Nv, Me, rng=None):
        if rng is not None:
            Nv, Me = Nv[rng.integers(len(Nv), size=len(Nv))], Me[rng.integers(len(Me), size=len(Me))]
        return (Me > np.quantile(Nv, 1 - alpha, axis=0)).mean(0)
    pt = np.mean([hits(*D[d]) for d in names], axis=0)
    rng = np.random.default_rng(ev.SEED)
    dr = np.stack([np.mean([hits(*D[names[j]], rng) for j in rng.integers(len(names), size=len(names))], axis=0)
                   for _ in range(nb)])
    ic = rows.index("counts")
    q = lambda d: [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]
    return {"alpha": alpha, "B": nb, "datasets": names,
            "n_null_steps": {d: len(D[d][0]) for d in names}, "n_events": {d: len(D[d][1]) for d in names},
            "ep": {r: float(pt[i]) for i, r in enumerate(rows)}, "ci95": {r: q(dr[:, i]) for i, r in enumerate(rows)},
            "counts_minus": {r: {"diff": float(pt[ic] - pt[i]), "ci95": q(dr[:, ic] - dr[:, i]),
                                 "p": inf.pval(dr[:, ic] - dr[:, i])} for i, r in enumerate(rows) if i != ic}}


# ------------------------------------------------------------------------------------------------ G: variants
def variants(bench, burn) -> dict:
    """Pooled EP of every row under protocol variants, with event and two-stage intervals (pool all ten)."""
    rows = ROWS
    past = [r for r in rows if r not in G_T]
    spec = {"guard17": {"guard": 17}, "guard_own": {"guard": SPAN}, "aligned": {"aligned": True},
            "within_segment": {"within": True},
            **{f"jitter{j:+d}": {"jitter": j} for j in (-2, -1, 1, 2)},
            "within_case": {"within": True, "case": True}, "tau0": {"tol": 0}}
    out = {"past_only_rows": past, "g_t_rows": [r for r in rows if r in G_T]}
    for name, o in spec.items():
        E = {}
        for ID in bench["pooled_datasets"]:
            mt = meta(ID)
            S = series(ID)
            seg, T, tol = mt["seg"], mt["T"], o.get("tol", mt["tol"])
            grp = eeg_case_of_step(mt) if o.get("case") and ID == "EEG" else None
            mask = ev.common_mask(S, burn, PRIMARY, seg)
            evs = [e + o.get("jitter", 0) for e in mt["events"] if 0 <= e + o.get("jitter", 0) < T]
            if o.get("aligned"):
                v0 = {r: inf.ep_variant(S[r], mask, seg, evs, tol) for r in rows if r in G_T}
                ev1 = [e + 1 for e in evs if e + 1 < T]
                v1 = {r: inf.ep_variant(S[r], mask, seg, ev1, tol) for r in past}
                keep = sorted(set.intersection(*[set(v) for v in v0.values()], *[{e - 1 for e in v} for v in v1.values()]))
                E[ID] = {r: np.array([v0[r][e] if r in v0 else v1[r][e + 1] for e in keep]) for r in rows}
            else:
                gd = o.get("guard", 0)
                v = {r: inf.ep_variant(S[r], mask, seg, evs, tol, guard=gd[r] if isinstance(gd, dict) else gd,
                                       within=o.get("within", False), group=grp) for r in rows}
                keep = sorted(set.intersection(*[set(x) for x in v.values()]))
                E[ID] = {r: np.array([v[r][e] for e in keep]) for r in rows}
        dss = [ID for ID in E if len(E[ID]["counts"])]
        out[name] = pool_block(E, dss, rows, levels=("event", "two_stage"), weights=("equal",), families=(("F1", F1),))
        out[name]["n_events_per_ds"] = {ID: len(E[ID]["counts"]) for ID in dss}
        out[name]["per_ds_ep"] = {ID: {r: float(E[ID][r].mean()) for r in rows} for ID in dss}
        print(f"variant {name}: counts {out[name]['equal']['ep']['counts']:.3f}", flush=True)
    return out


# ------------------------------------------------------------------------------------------------ H: EEG cases
def eeg_case_of_step(mt) -> np.ndarray:
    """Case index (chb01, chb03, chb05) of every step of the benchmark EEG (its segments are files)."""
    files = {s["segment"]: s["file"] for s in json.loads((BENCH / "EEG_meta.json").read_text())["segments"]}
    names = sorted({f.split("_")[0] for f in files.values()})
    return np.array([names.index(files[int(k)].split("_")[0]) for k in mt["seg"]])


def eeg_cases(E, burn) -> dict:
    mt = meta("EEG")
    files = {s["segment"]: s["file"] for s in json.loads((BENCH / "EEG_meta.json").read_text())["segments"]}
    ok = per_event("EEG", PRIMARY, burn)[2]["events"]
    case = np.array([files[int(mt["seg"][e])].split("_")[0] for e in ok])
    return {"n_events": {c: int((case == c).sum()) for c in sorted(set(case))},
            "ep": {r: {c: float(E["EEG"][r][case == c].mean()) for c in sorted(set(case))} for r in E["EEG"]}}


# ------------------------------------------------------------------------------------------------ K: event types, clusters
# Tag rule (spec v6 T, fixed from the descriptions before any EP by tag): A (activity by design) if a description of
# the event names a scheduled break, holiday, meal or coffee break, the start or end of a term, of classes or of a
# scheduled session, or a change in the volume of activity or in who takes part; R: a software release (AskUbuntu;
# primary O, sensitivity A); O: every other event.
TAG_A = (r"break|holiday|vacation|christmas|newyear|new year|thanksgiving|columbus|lunch|coffee|term|classes|session|"
         r"workshop|keynote|\bvolumes?\b|\bvol\b|whales leave")
DESC_FIXED = {"AskUbuntu": "Ubuntu release", "EEG": "seizure onset", "PM": "activity transition of the subject"}


def event_tags(EV) -> dict:
    """{dataset: {event: [tag, descriptions]}} of the scored events (descriptions: <ID>_meta.json; ENRON: its unmerged
    dates, a date joining the merged event at its week or one week later)."""
    import re
    out = {}
    for ID, evs in EV.items():
        if ID in DESC_FIXED:
            desc = {e: [DESC_FIXED[ID]] for e in evs}
        else:
            m = json.loads((BENCH / f"{ID}_meta.json").read_text())
            if ID == "ENRON":
                desc = {e: [u["desc"] for u in m["events_unmerged"] if e - 1 <= u["index_iso_week"] <= e] for e in evs}
            else:
                desc = {}
                for x in m["events"]:
                    desc.setdefault(int(x.get("global_index", x["index"])), []).append(
                        x.get("desc") or x.get("name") or x.get("event"))
        a = re.compile(TAG_A, re.IGNORECASE)
        out[ID] = {int(e): ["R" if ID == "AskUbuntu" else "A" if any(a.search(d) for d in desc[e]) else "O", desc[e]]
                   for e in evs}
    return out


def tag_pools(E, EV, bench) -> dict:
    """EP of every row on the events of one type (spec v6 T): A and O (releases O), A_R and O_R (releases A), the
    source four and the held-out seven without the A events; equal and event weight, two-stage intervals."""
    tags = event_tags(EV)
    alln = bench["pooled_datasets"]

    def sub(keep, dss):
        Es = {}
        for d in dss:
            m = np.array([keep(tags[d][int(e)][0]) for e in EV[d]])
            if m.any():
                Es[d] = {r: v[m] for r, v in E[d].items()}
        return Es
    spec = {"tagA": (lambda t: t == "A", alln), "tagO": (lambda t: t != "A", alln),
            "tagA_R": (lambda t: t != "O", alln), "tagO_R": (lambda t: t == "O", alln),
            "source4_noA": (lambda t: t != "A", [d for d in alln if d in SOURCE]),
            "heldout7_noA": (lambda t: t != "A", [d for d in alln if d not in SELECTION])}
    out = {"rule": TAG_A, "tags": {d: {str(e): v for e, v in t.items()} for d, t in tags.items()},
           "n_events": {k: {d: int(sum(keep(t[0]) for t in tags[d].values())) for d in dss}
                        for k, (keep, dss) in spec.items()}}
    for k, (keep, dss) in spec.items():
        Es = sub(keep, dss)
        b = pool_block(Es, sorted(Es), ROWS, levels=("event", "two_stage"), best=True)
        dataset_tests(b, sorted(Es))
        ep = b["equal"]["ep"]
        best = max(F2, key=ep.get)
        b["best_other"] = {"row": best, "ep": ep[best], "counts_minus": b["equal"]["two_stage"]["counts_minus"][best],
                           "counts_causal_minus": b["equal"]["two_stage"]["counts_causal_minus"][best]}
        out[k] = b
        print(f"tags {k} ({b['n_events']} events, {len(Es)} datasets): counts {ep['counts']:.3f}, best {best} "
              f"{ep[best]:.3f}", flush=True)
    return out


def tag_strata(tags, act) -> dict:
    """Large or extreme events (step 19's activity strata) among the A events and the others (releases O): pooled over
    events and averaged over the datasets that have events of the type."""
    out = {}
    for k, keep in (("tagA", lambda t: t == "A"), ("tagO", lambda t: t != "A")):
        per = {}
        for d, t in tags.items():
            st = dict(zip(act["datasets"][d]["events"], act["datasets"][d]["C_stratum"]))
            big = [st[int(e)] in ("large", "extreme") for e, v in t.items() if keep(v[0])]
            if big:
                per[d] = [int(sum(big)), len(big)]
        out[k] = {"large_or_extreme": int(sum(v[0] for v in per.values())), "n": int(sum(v[1] for v in per.values())),
                  "mean_share_over_datasets": float(np.mean([v[0] / v[1] for v in per.values()])), "per_ds": per}
    return out


def cluster_labels(EV, with_days: bool) -> dict:
    """K1: PM subject (= segment) and EEG case of each scored event; K2: also the day (segment) of PS and HT09."""
    out = {"PM": meta("PM")["seg"][EV["PM"]], "EEG": eeg_case_of_step(meta("EEG"))[EV["EEG"]]}
    if with_days:
        out |= {d: meta(d)["seg"][EV[d]] for d in ("PS", "HT09")}
    return out


def cluster_sensitivity(E, EV, P) -> dict:
    """The pools all ten and >= 10 events with the three-stage bootstrap (spec v6 C), K1 and K2, beside two-stage."""
    out = {}
    for name, days in (("K1", False), ("K2", True)):
        cl = cluster_labels(EV, days)
        out[name] = {"n_clusters": {d: len(np.unique(v)) for d, v in cl.items()}}
        for k in ("all", "10plus"):
            dss = P[k]["datasets"]
            b = pool_block(E, dss, ROWS, levels=("three_stage",), weights=("equal",), clusters=cl)["equal"]
            ts = P[k]["equal"]["two_stage"]
            width = lambda c: c[1] - c[0]
            b["width_ratio_counts_minus"] = {r: width(v["ci95"]) / width(ts["counts_minus"][r]["ci95"])
                                             for r, v in b["three_stage"]["counts_minus"].items()}
            out[name][k] = b
            cm = b["three_stage"]["counts_causal_minus"]["ase"]
            print(f"clusters {name} {k}: counts_causal - ase {cm['diff']:+.3f} [{cm['ci95'][0]:.3f}, {cm['ci95'][1]:.3f}]"
                  f" Holm F2 {cm['p_holm_F2']:.3f}", flush=True)
    return out


# ------------------------------------------------------------------------------------------------ U: upper bounds
def upper_bounds(E, P) -> dict:
    """One-sided 95% upper bounds of row - counts (and row - causal counts) over the 16 rows of F2, simultaneous
    (spec v6 U): max-t and Bonferroni on the two-stage draws, Bonferroni t and sign-flip over the datasets."""
    out = {}
    for k in ("all", "10plus"):
        dss = P[k]["datasets"]
        out[k] = {}
        for ref in REFS:
            rows = [ref] + F2
            X = {d: np.stack([E[d][r] for r in rows], 1) for d in dss}
            dr, pt = inf.pooled_draws(X, "two_stage"), inf.pooled_point(X)
            mu = np.stack([X[d].mean(0) for d in dss])
            u = inf.upper_bounds(dr[:, 1:] - dr[:, :1], pt[1:] - pt[0], mu[:, 1:] - mu[:, :1])
            res = {"diff": dict(zip(F2, (pt[1:] - pt[0]).tolist()))}
            for m in ("max_t", "bonferroni", "t_bonferroni", "signflip_bonferroni"):
                res[m] = dict(zip(F2, u[m]))
                res[f"{m}_max"] = {"row": max(F2, key=res[m].get), "bound": max(u[m])}
            res |= {"max_t_c": u["max_t_c"], "t_quantile": u["t_quantile"],
                    "signflip_bounded": all(u["signflip_bounded"]), "n_datasets": len(dss)}
            out[k][ref] = res
            print(f"upper bounds {k} row - {ref}: max-t {res['max_t_max']}, sign-flip {res['signflip_bonferroni_max']}",
                  flush=True)
    return out


# ------------------------------------------------------------------------------------------------ main
def run() -> dict:
    bench = json.loads((RESULTS / "bench.json").read_text())
    st = settings()
    burn = {**st["burn"], "lad_own": 0, "scpd_own": 0}
    pe = {d: per_event(d, PRIMARY, burn) for d in bench["pooled_datasets"]}
    E, EV = {d: v[0] for d, v in pe.items()}, {d: v[2]["events"] for d, v in pe.items()}
    out = {"spec": "prereg/specs/STATS.md section 1, ROUND3.md part A, prereg/TURNOVER.md, "
                   "specs/V4.md L, V, V5.md, V6.md C, T, U, P, E (written before the runs)",
           "rows": {"core": PRIMARY, "identity": IDENTITY, "wiring": WIRING, "causal": CAUSAL, "turnover": TURN,
                    "F1": F1, "F2": F2},
           "B_boot": ev.NBOOT, "seed": ev.SEED,
           "S0": check_s0(E, bench, json.loads((RESULTS / "identity.json").read_text()),
                          json.loads((RESULTS / "activity.json").read_text()))}
    print("S0", out["S0"], flush=True)
    out["pools"] = pools(E, bench)
    out["count_variants"] = count_variants(E, out["pools"])
    out["lad_rows"] = lad_rows(E, out["pools"])
    out["association"] = association(E, out["pools"])
    out["datasets"] = datasets_table(E, bench, burn)
    out["turnover"] = turnover_check(E, bench["pooled_datasets_10plus"])
    out["labels"] = label_variants(E, bench, burn)
    out["per_dataset"] = per_dataset(E, bench)
    out["coverage"] = coverage(E, bench)
    print("coverage", out["coverage"]["coverage"], flush=True)
    out["peel_clauset"] = peel_clauset(E, bench, burn)
    out["bench_delay"] = bench_delay(bench, burn)
    out["bench_delay"]["threshold_boot"] = {f"alpha{a}": fa_threshold_boot(bench, burn, a) for a in (0.01, 0.05)}
    out["variants"] = variants(bench, burn)
    out["variants"]["tau_curve"] = tau_curve(bench, burn)
    out["eeg_cases"] = eeg_cases(E, burn)
    out["event_types"] = tag_pools(E, EV, bench)
    out["event_types"]["by_stratum"] = tag_strata(out["event_types"]["tags"], json.loads((RESULTS / "activity.json").read_text()))
    out["clusters"] = cluster_sensitivity(E, EV, out["pools"])
    out["upper_bounds"] = upper_bounds(E, out["pools"])
    return out


def summary(o: dict) -> str:
    f3 = lambda x: f"{x:.3f}"
    ci = lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]"
    md = [f"S0 (reproduces steps 12, 16, 19): {json.dumps(o['S0'])}\n"]
    pl = {**o["pools"], **{f"{k}_SCCaudit": v for k, v in o["labels"]["scc_audit"]["pools"].items()}}
    for k in ("all", "10plus", "10plus_noEEG", "heldout7", "no_posthoc", "source4", "own_added6", "interaction",
              "correlation", "all_noUCI", "all_noSCC", "10plus_noSCC", "all_SCCaudit", "10plus_SCCaudit"):
        b = pl[k]
        md.append(f"## A/B. pool {k} ({', '.join(b['datasets'])}; {b['n_events']} events)\n")
        md.append("| row | EP equal | event-level CI | two-stage CI | dataset CI | EP event-weighted | two-stage CI |"
                  " counts - row (two-stage CI, p, Holm F2) | rank median [95%] | P(rank 1) |\n|---|---|---|---|---|"
                  "---|---|---|---|---|")
        e, w = b["equal"], b["event"]
        for r in e["ep"]:
            cm = e["two_stage"]["counts_minus"].get(r)
            cms = (f"{cm['diff']:+.3f} {ci(cm['ci95'])} p {cm['p']:.4f} Holm {cm.get('p_holm_F2', float('nan')):.4f}"
                   if cm else "-")
            rk = e["two_stage"]["rank"].get(r)
            rks = f"{rk['median']:.0f} [{rk['ci95'][0]}, {rk['ci95'][1]}] | {rk['p_rank1']:.3f}" if rk else "- | -"
            md.append(f"| {r} | {f3(e['ep'][r])} | {ci(e['event']['ep_ci95'][r])} | {ci(e['two_stage']['ep_ci95'][r])} "
                      f"| {ci(e['dataset']['ep_ci95'][r])} | {f3(w['ep'][r])} | {ci(w['two_stage']['ep_ci95'][r])} "
                      f"| {cms} | {rks} |")
        for ref, key in (("counts_causal", "counts_causal_minus"),):
            md.append(f"\n{ref} - row (two-stage CI, Holm F2): " + "; ".join(
                f"{r} {v['diff']:+.3f} {ci(v['ci95'])} H2 {v.get('p_holm_F2', float('nan')):.4f}"
                for r, v in e["two_stage"][key].items()))
        if "best_set" in e["two_stage"]:
            md.append("\nRows not excluded from being best (two-stage / event): "
                      + ", ".join(e["two_stage"]["best_set"]["not_excluded"]) + " / "
                      + ", ".join(e["event"]["best_set"]["not_excluded"]) + "; with counts_causal in place of counts: "
                      + ", ".join(e["two_stage"]["best_set_counts_causal"]["not_excluded"]))
        sm = e.get("dataset_small_n", {})
        md.append("\nDataset level, small n (t_{D-1} CI, p; sign-flip p): " + "; ".join(
            f"{ref} - {r} {ci(v['t']['ci95'])} p {v['t']['p']:.3f} sf {v['signflip']['p']:.3f}"
            for ref in sm for r, v in sm[ref].items() if r in ("tire", "hamming", "ase", "nbs", "lad", "bocpd",
                                                                "scpd", "counts", "counts_causal", "turnover")))
        md.append(f"\nSign-flip power limit: {json.dumps(e.get('signflip_limit'))}; Holm over the sign-flip p (F2): "
                  + json.dumps({r: round(v, 4) for r, v in e.get("signflip_holm", {}).get("F2", {}).items()}))
        md.append("MDE80 of counts - row (two-stage SD): " + "; ".join(
            f"{r} {v['mde80']:.3f}" for r, v in e["two_stage"]["counts_minus"].items()))
        md.append("\nEvent-level contrasts counts - row (CI, p, Holm F1 / F2): " + "; ".join(
            f"{r} {v['diff']:+.3f} {ci(v['ci95'])} p {v['p']:.4f} H1 {v.get('p_holm_F1', float('nan')):.4f} "
            f"H2 {v.get('p_holm_F2', float('nan')):.4f}" for r, v in e["event"]["counts_minus"].items()) + "\n")
    for name, t in (("", o["turnover"]), (" (SCC audit labels)", o["labels"]["scc_audit"]["turnover"]),
                    (" (without SCC)", o["labels"]["no_scc"]["turnover"])):
        md.append(f"## T. Vertex turnover{name}: {t['reading']} ({json.dumps(t['reading_per_row'])})\n")
        for k, v in t.items():
            if isinstance(v, dict) and k != "reading_per_row":
                md.append(f"* {k}: " + "; ".join(f"{c} {x['diff']:+.3f} {ci(x['ci95'])}" for c, x in v.items()
                                                 if isinstance(x, dict)))
    a = o["labels"]["scc_audit"]
    md.append(f"\nSCC audit labels: {len(a['events'])} events, {a['n_events_scored']} scored; SCC EP audit / primary: "
              + "; ".join(f"{r} {a['ep_scc'][r]:.3f} / {a['ep_scc_primary'][r]:.3f}" for r in a["ep_scc"]))
    md.append("\n## Drop one dataset (equal weight): counts EP, best other row, P(counts rank 1)\n")
    for d, b in o["pools"]["drop_one"].items():
        oth = {r: v for r, v in b["equal"]["ep"].items() if r not in ("counts", "constant", "counts_causal", "turnover")}
        best = max(oth, key=oth.get)
        cm = b["equal"]["two_stage"]["counts_minus"][best]
        md.append(f"* without {d}: counts {f3(b['equal']['ep']['counts'])}, - {best} {cm['diff']:+.3f} {ci(cm['ci95'])}, "
                  f"P(rank 1) {b['equal']['two_stage']['rank']['counts']['p_rank1']:.3f}")
    md.append("\n## C. Per dataset, counts - row: diff, percentile CI, sign-flip p and CI\n")
    for d, res in o["per_dataset"].items():
        md.append(f"* {d} (n = {res['n_events']}): " + "; ".join(
            f"{r} {v['diff']:+.3f} {ci(v['ci95_percentile'])} sf p {v['signflip']['p']:.3f} "
            f"{ci(v['signflip']['ci95']) if v['signflip']['bounded'] else 'not bounded'}"
            for r, v in res.items() if r in ("tire", "bocpd", "lad", "hamming", "ase", "transitivity", "counts_causal",
                                             "turnover")))
    md.append(f"\n## D. Coverage at mean 0 ({o['coverage']['pool']}): {json.dumps(o['coverage']['coverage'])}\n")
    md.append(f"Coverage of the pooled intervals (super-population of datasets): {json.dumps(o['coverage']['pooled'])}\n")
    pc = o["peel_clauset"]
    md.append(f"## E. Peel-Clauset constructions (events scored {pc['n_events']})\n")
    for k in ("published", "plus_pc", "pc_replaces_mit_enron"):
        e = pc[k]["equal"]
        md.append(f"* {k}: " + "; ".join(f"{r} {f3(e['ep'][r])} {ci(e['event']['ep_ci95'][r])}" for r in e["ep"])
                  + f"; P(counts rank 1, two-stage) {e['two_stage']['rank']['counts']['p_rank1']:.3f}")
    md.append("\n## F. Detection within tol at fixed null quantiles, equal weight (two-stage CI; counts - row)\n")
    for a, b in o["bench_delay"]["inference"].items():
        e = b["equal"]
        md.append(f"* {a} ({', '.join(b['datasets'])}): " + "; ".join(
            f"{m} {e['ep'][m]:.3f} {ci(e['two_stage']['ep_ci95'][m])}" for m in e["ep"]))
        md.append("  counts - row (event CI / two-stage CI): " + "; ".join(
            f"{m} {v['diff']:+.3f} {ci(e['event']['counts_minus'][m]['ci95'])} / {ci(v['ci95'])}"
            for m, v in e["two_stage"]["counts_minus"].items()))
    md.append("\n## G. Protocol variants (all ten; two-stage CI; counts - row two-stage; P(counts rank 1))\n")
    for k, b in o["variants"].items():
        if isinstance(b, dict) and "equal" in b:
            e = b["equal"]
            md.append(f"* {k} (events {b['n_events_per_ds']}): " + "; ".join(
                f"{m} {e['ep'][m]:.3f} {ci(e['two_stage']['ep_ci95'][m])}" for m in e["ep"])
                + f"; P(counts rank 1) {e['two_stage']['rank']['counts']['p_rank1']:.3f}")
            md.append("  counts - row: " + "; ".join(f"{m} {v['diff']:+.3f} {ci(v['ci95'])}"
                                                    for m, v in e["two_stage"]["counts_minus"].items()))
    tc = o["variants"]["tau_curve"]
    md.append("\n## G2. Tolerance curve, one tau on every dataset: pooled EP (equal weight), all ten | >= 10 events\n")
    md.append("| row | " + " | ".join(f"tau {t}" for t in tc) + " |\n|---|" + "---|" * len(tc))
    for r in tc["0"]["all"]["ep"] if "0" in tc else tc[0]["all"]["ep"]:
        md.append(f"| {r} | " + " | ".join(f"{v['all']['ep'][r]:.3f} / {v['10plus']['ep'][r]:.3f}" for v in tc.values())
                  + " |")
    for k in ("all", "10plus"):
        md.append(f"\n{k}: best other row, P(counts rank 1), rows above the counts (two-stage CI of counts - row < 0; "
                  "Holm F2 < 0.05): " + "; ".join(
                      f"tau {t} {v[k]['best_other']} {v[k]['ep'][v[k]['best_other']]:.3f} P1 {v[k]['p_rank1_counts']:.3f} "
                      f"dropped {v[k]['datasets_dropped']} "
                      + ",".join(r for r, c in v[k]["counts_minus"].items() if c["ci95"][1] < 0
                                 and (c["p_holm_F2"] or 1) < 0.05) for t, v in tc.items()))
    cv = o["count_variants"]
    md.append(f"\n## V. Count reference rows (best causal variant on all ten: {cv['best_causal_all_ten']})\n")
    for k in ("all", "10plus"):
        md.append(f"* {k}: " + "; ".join(f"{r} {v['ep']:.3f} {ci(v['two_stage'])} (event weight {v['event_weighted']:.3f})"
                                       for r, v in cv[k].items()))
    for ref, g in cv["gap_10plus"].items():
        md.append(f"* >= 10, row - {ref}: " + "; ".join(f"{c} {v['diff']:+.3f} {ci(v['ci95'])} t {ci(v['t']['ci95'])}"
                                                    for c, v in g.items()))
    md.append("\n## N. Datasets: T, segments, tol, events labelled / scored, null windows, vertex set\n")
    for d, v in o["datasets"].items():
        md.append(f"* {d}: T {v['T']}, segments {v['segments']}, tol {v['tol']}, events {v['events_labelled']} / "
                  f"{v['events_scored']}, null windows {v['null_windows']}, vertex set {v['vertex_set']}")
    t10 = o["pools"]["10plus"]["equal"]["dataset_ep_t"]
    md.append("\n>= 10 events, t_3 interval of every row's EP over the 4 dataset means: " + "; ".join(
        f"{r} {ci(v['ci95'])}" for r, v in t10.items()))
    md.append(f"Coverage at K = 4 ({o['coverage']['pooled_10plus']['contrast']}): "
              + json.dumps({k: v for k, v in o["coverage"]["pooled_10plus"].items() if k != "contrast"}))
    if "lad_rows" in o:
        md.append("\n## LAD. LAD with 5 (Table 2), 6 and 499 eigenvalues (spec v5 M)\n")
        for k, res in o["lad_rows"].items():
            md.append(f"* {k}: " + "; ".join(
                f"{r} {v['ep']:.3f} {ci(v['two_stage'])} (event weight {v['event_weighted']:.3f})"
                + (f", counts - row {v['counts_minus']['diff']:+.3f} {ci(v['counts_minus']['ci95'])}"
                   if "counts_minus" in v else "")
                + (f", row - lad {v['minus_lad']['diff']:+.3f} {ci(v['minus_lad']['ci95'])} p {v['minus_lad']['p']:.3f}"
                   f", Holm F2 in place of lad {v['p_holm_F2_in_place_of_lad']:.3f}" if "minus_lad" in v else "")
                for r, v in res.items() if r != "per_dataset"))
            md.append("  per dataset: " + json.dumps({d: {r: round(x, 3) for r, x in v.items()}
                                                      for d, v in res["per_dataset"].items()}))
    if "association" in o:
        md.append("\n## P. Per-event association with the counts: Spearman rho (events / within datasets), share of "
                  "events above the counts [two-stage CI] (event weight), share tied\n")
        for k, b in o["association"].items():
            md.append(f"* {k} ({b['n_events']} events): " + "; ".join(
                f"{r} rho {f3(v['spearman_events']) if v['spearman_events'] is not None else 'n/a'} / "
                f"{f3(v['spearman_within']) if v['spearman_within'] is not None else 'n/a'} (n {v['n_within']}), "
                f"above {v['share_above']:.3f} {ci(v['share_above_ci95'])} ({v['share_above_event_weighted']:.3f}), "
                f"tied {v['share_tied']:.3f}" for r, v in b["rows"].items()))
    if "event_types" in o:
        et = o["event_types"]
        md.append(f"\n## K. Event types (rule: {et['rule']}); events per pool: {json.dumps(et['n_events'])}\n")
        for k in ("tagA", "tagO", "tagA_R", "tagO_R", "source4_noA", "heldout7_noA"):
            b, e = et[k], et[k]["equal"]
            bo = b["best_other"]
            md.append(f"* {k} ({', '.join(b['datasets'])}; {b['n_events']} events): " + "; ".join(
                f"{r} {e['ep'][r]:.3f} {ci(e['two_stage']['ep_ci95'][r])}" for r in e["ep"]))
            md.append(f"  best other {bo['row']} {bo['ep']:.3f}: counts - it {bo['counts_minus']['diff']:+.3f} "
                      f"{ci(bo['counts_minus']['ci95'])}; causal - it {bo['counts_causal_minus']['diff']:+.3f} "
                      f"{ci(bo['counts_causal_minus']['ci95'])}; P(counts rank 1) "
                      f"{e['two_stage']['rank']['counts']['p_rank1']:.3f}; event weight counts "
                      f"{b['event']['ep']['counts']:.3f}, {bo['row']} {b['event']['ep'][bo['row']]:.3f}")
            md.append("  counts - row (two-stage, Holm F2): " + "; ".join(
                f"{r} {v['diff']:+.3f} {ci(v['ci95'])} H2 {v.get('p_holm_F2', float('nan')):.3f}"
                for r, v in e["two_stage"]["counts_minus"].items() if r in F2))
    if "by_stratum" in o.get("event_types", {}):
        bs = o["event_types"]["by_stratum"]
        md.append("* large or extreme events by type: " + "; ".join(
            f"{k} {v['large_or_extreme']}/{v['n']} (mean over datasets {v['mean_share_over_datasets']:.3f})" for k, v in bs.items()))
    if "clusters" in o:
        md.append("\n## K. Clustered bootstrap (three-stage) beside two-stage\n")
        for name, c in o["clusters"].items():
            md.append(f"* {name} clusters {c['n_clusters']}")
            for k in ("all", "10plus"):
                ts3, ts2 = c[k]["three_stage"], o["pools"][k]["equal"]["two_stage"]
                md.append(f"  {k}: counts {ci(ts3['ep_ci95']['counts'])} (two-stage {ci(ts2['ep_ci95']['counts'])}); "
                          + "; ".join(f"counts - {r} {ci(ts3['counts_minus'][r]['ci95'])} H2 "
                                      f"{ts3['counts_minus'][r].get('p_holm_F2', float('nan')):.3f} (two-stage "
                                      f"{ci(ts2['counts_minus'][r]['ci95'])} H2 "
                                      f"{ts2['counts_minus'][r].get('p_holm_F2', float('nan')):.3f})"
                                      for r in ("hamming", "ase", "nbs", "bocpd", "mmd", "scpd"))
                          + "; " + "; ".join(f"causal - {r} {ci(ts3['counts_causal_minus'][r]['ci95'])} H2 "
                                             f"{ts3['counts_causal_minus'][r].get('p_holm_F2', float('nan')):.3f} "
                                             f"(two-stage {ci(ts2['counts_causal_minus'][r]['ci95'])} H2 "
                                             f"{ts2['counts_causal_minus'][r].get('p_holm_F2', float('nan')):.3f})"
                                             for r in ("hamming", "ase", "nbs")))
    if "upper_bounds" in o:
        md.append("\n## U. Simultaneous one-sided 95% upper bounds of row - reference over F2\n")
        for k, v in o["upper_bounds"].items():
            for ref, u in v.items():
                md.append(f"* {k}, row - {ref} (K = {u['n_datasets']}): " + "; ".join(
                    f"{m} max {u[f'{m}_max']['bound']:+.3f} ({u[f'{m}_max']['row']})"
                    for m in ("max_t", "bonferroni", "t_bonferroni", "signflip_bonferroni"))
                    + f"; sign-flip bounded {u['signflip_bounded']}")
    for k in ("all", "10plus"):
        e = o["pools"][k]["equal"]
        if "t_holm_counts_causal" in e:
            md.append(f"\n{k}: t p / Holm F2 (t) / sign-flip p / Holm F2 (sign-flip), counts_causal - row and counts - "
                      "row: " + "; ".join(
                          f"{ref} - {r} {e['dataset_small_n'][ref][r]['t']['p']:.3f} / "
                          f"{e['t_holm' + sfx]['F2'][r]:.3f} / {e['dataset_small_n'][ref][r]['signflip']['p']:.3f} / "
                          f"{e['signflip_holm' + sfx]['F2'][r]:.3f}"
                          for ref, sfx in (("counts_causal", "_counts_causal"), ("counts", ""))
                          for r in ("ase", "hamming", "nbs")))
    c = o["eeg_cases"]
    md.append(f"\n## H. Benchmark EEG per case (events {c['n_events']})\n")
    md.append("; ".join(f"{r} " + "/".join(f"{v:.3f}" for v in x.values()) for r, x in c["ep"].items()))
    return "\n".join(md)


def main() -> None:
    p = RESULTS / "inference.json"
    if "--summary" in sys.argv and p.exists():
        print(summary(json.loads(p.read_text())))
        return
    t0 = Timer()
    write_json(p, run())
    print(f"step 23: {t0():.0f} s -> {p}")
    print(summary(json.loads(p.read_text())))


if __name__ == "__main__":
    main()
