"""Step 30: Figure 1 (fig:main; panels a, b) and Figure 2 (fig:structure; panels c, d below), read live from the JSON
results.

(a) Labelled events: EP of the count detector (x) against the EP of the best other detector on the same dataset (y;
    best of the permutation-invariant detectors, the node-aligned detectors and the structural statistics, chosen per
    dataset after the fact; colour and marker = that row). Per-dataset EP: bench.json datasets.<ds>.m2.ep_mean,
    identity.json / identity_new.json datasets.<ds>.ep_mean, activity.json datasets.<ds>.D.<row>.ep_mean. Large
    markers: pooled EP of the counts (filled) and of the causal counts (open) against the best pooled other row,
    with two-stage 95% CIs (inference.json pools.all.equal).
(b) EP by the size of the activity change at the event (activity.json C_strata.all.<stratum>.<row>.{mean, lo, hi}):
    every detector and structural statistic in the background, three rows with their 95% CIs.
(c) Planted rewiring at fixed degree, rho = 0.5, on HT09 and PAMAP2 (the backgrounds with ten or more bootstrap
    blocks): the mean shift of each row's planted maximum in standard deviations of its highest scores over event-free windows
    (plants.json effect_table.cells; Hamming, Jaccard, DeltaCon, Chen-Zhang, NBS: identity_plants.json
    cells.<cell>.indep.rows), one marker per plant that moves edges (not HT09's triangle-closing plant) and per
    relabelling (open); faint where the paired 95% interval covers 0.5; not shown where tied in at least half the
    trials. Band: |shift| < 0.1 SD (spec V4 V6).
(d) Block models at matched expected degree and s-GNN's vertex exchange (identity_new.json
    synth.<scenario>.minus_counts.<row>): EP(row) - EP(counts) on the same events with its 95% interval, filled where
    it excludes 0; the structural statistics were not run on these series.
Output: paper/figures/fig_main.pdf (Figure 1) and fig_structure.pdf (Figure 2) in the directory of --out; in Figure 2
(c) and (d) are tagged (a) and (b). --png adds 300-DPI previews next to the PDFs.
The log prints every plotted value with its source key.

Usage: 30_fig_main.py [--out PDF] [--png]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import figstyle as fs
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from figstyle import SMALL, usetex_available
from matplotlib.lines import Line2D
from tueplots import bundles

ROOT = Path(__file__).resolve().parent              # repro/
RESULTS = ROOT / "results"
_ap = argparse.ArgumentParser()
_ap.add_argument("--out", default=str(ROOT.parent / "paper" / "figures" / "fig_main.pdf"))
_ap.add_argument("--png", action="store_true", help="300-DPI PNG previews next to the PDFs")
ARGS = _ap.parse_args()
OUT = Path(ARGS.out)
OUT2 = OUT.with_name("fig_structure.pdf")
OUT.parent.mkdir(parents=True, exist_ok=True)


def _load(name):
    return json.loads((RESULTS / f"{name}.json").read_text())


BENCH, IDENT, INEW, INFER = _load("bench"), _load("identity"), _load("identity_new"), _load("inference")
ACT, PLANTS, IDPL = _load("activity"), _load("plants"), _load("identity_plants")

# Detector map (one colour + marker per row in every panel): counts in the brand blue (the reference), the
# permutation-invariant detectors on the grey ramp, the identity-aware detectors in teal, the structural statistics coral.
TEAL2 = "#6CC4BC"                                   # light teal (third identity shade)
DETECTOR = {  # key -> (display, colour, marker)
    "counts": ("counts", fs.BLUE, "o"),
    "counts_causal": ("causal counts", fs.BLUE, "o"),
    "cusum": ("CUSUM", fs.MIDGREY, "X"),
    "bocpd": ("BOCPD", fs.LIGHTGREY, "^"),
    "mmd": ("MMD", "#A7ADB9", "*"),
    "lad": ("LAD", fs.SLATE, "P"),
    "scpd": ("SCPD", fs.GREY, "p"),
    "tire": ("TIRE", "#4B5563", "v"),
    "hamming": ("Hamming", fs.TEAL, "<"),
    "jaccard": ("Jaccard", TEAL2, ">"),
    "deltacon": ("DeltaCon", fs.DTEAL, "d"),
    "chenzhang": ("Chen–Zhang", TEAL2, "8"),
    "nbs": ("NBS", fs.TEAL, "H"),
    "ase": ("ASE", fs.DTEAL, "h"),
    "spotlight": ("SpotLight-style", fs.TEAL, "D"),
    "transitivity": ("transitivity", fs.CORAL, "o"),
    "assortativity": ("assortativity", fs.CORAL, "s"),
    "fair_pair": ("max. of the two", fs.CORAL, "^"),
}
CORE = ("cusum", "bocpd", "mmd", "lad", "scpd", "tire")
IDENTITY = ("hamming", "jaccard", "deltacon", "chenzhang", "nbs", "ase", "spotlight")
NEW = ("ase", "spotlight")
WIRING = ("transitivity", "assortativity", "fair_pair")
INK, NAVY, BLUE, GREY, FAINT, REF = fs.INK, fs.NAVY, fs.BLUE, fs.GREY, fs.FAINT, fs.REF
HALO = [pe.withStroke(linewidth=2.4, foreground="white")]          # labels stay clear of the lines below
WHISK = {"elinewidth": 0.8, "capsize": 1.6, "capthick": 0.8}      # whiskers take the colour of their marker


def pad(lo, hi, f=0.06):
    return lo - f * (hi - lo), hi + f * (hi - lo)


def family_colour(r: str) -> str:
    return BLUE if r.startswith("counts") else fs.TEAL if r in IDENTITY else fs.CORAL if r in WIRING else INK


# ------------------------------------------------------------------------------------------------ data
# (a) the ten datasets with evaluable events, every row on the same mask and events
DS_NAME = {"UCI": "UCI", "MAG": "MAG", "SCC": "Stablecoin", "PM": "PAMAP2", "MIT": "MIT", "ENRON": "Enron",
           "PS": "School", "HT09": "HT09", "EEG": "EEG", "AskUbuntu": "AskUbuntu"}
A_DS = list(BENCH["pooled_datasets"])
assert set(A_DS) == set(DS_NAME) == set(IDENT["pooled_datasets"]) == set(INEW["pooled_datasets"]), A_DS
POOL = INFER["pools"]["all"]["equal"]
# row order of Table 3: within each family, descending pooled EP (the same order in (c), (d) and every table)
CORE, IDENTITY, WIRING = (tuple(sorted(g, key=lambda r: -POOL["ep"][r])) for g in (CORE, IDENTITY, WIRING))
A_OTHERS = CORE + IDENTITY + WIRING


def ds_ep(r, ds):
    if r in CORE or r == "counts":
        return BENCH["datasets"][ds]["m2"]["ep_mean"][r]
    if r in NEW:
        return INEW["datasets"][ds]["ep_mean"][r]
    if r in WIRING:
        return ACT["datasets"][ds]["D"][r]["ep_mean"]
    return IDENT["datasets"][ds]["ep_mean"][r]


A = {}
for ds in A_DS:
    c = ds_ep("counts", ds)
    for r in A_OTHERS:                                # the step-23 per-dataset contrast uses the same EPs
        assert abs(c - ds_ep(r, ds) - INFER["per_dataset"][ds][r]["diff"]) < 1e-9, (ds, r)
    best = max(A_OTHERS, key=lambda r: (ds_ep(r, ds), r != "fair_pair"))   # a tie with the maximum: its part
    A[ds] = (c, best, ds_ep(best, ds))
A_POOL_BEST = max(A_OTHERS, key=lambda r: POOL["ep"][r])
A_POOL = {r: (POOL["ep"][r], *POOL["two_stage"]["ep_ci95"][r]) for r in ("counts", "counts_causal", A_POOL_BEST)}

# (b) EP by activity stratum
STRATA = ("ordinary", "large", "extreme")
CS = ACT["C_strata"]["all"]
B_ROWS = ("counts",) + CORE + IDENTITY + WIRING
B_KEY = {r: (f"id_{r}" if r in IDENTITY else r) for r in B_ROWS}
B_VAL = {r: [(CS[s][B_KEY[r]]["mean"], CS[s][B_KEY[r]]["lo"], CS[s][B_KEY[r]]["hi"]) for s in STRATA] for r in B_ROWS}
B_N = [CS[s]["counts"]["n_events"] for s in STRATA]
B_NDS = [len(CS[s]["counts"]["datasets"]) for s in STRATA]
B_ID = max(IDENTITY, key=lambda r: POOL["ep"][r])     # the node-aligned row with the highest pooled EP
B_HI = ("counts", B_ID, "transitivity")               # with intervals and direct labels
B_FAM = lambda r: "identity" if r in IDENTITY else "wiring" if r in WIRING else "core"
B_BG = {"core": (fs.MIDGREY, "-"), "identity": (TEAL2, "-"), "wiring": (fs.CORAL, (0, (2.5, 1.5)))}
B_BG_NAME = {"core": "other perm.-inv.", "identity": "other node-aligned", "wiring": "other structural"}

# (c) planted structural changes at rho = 0.5 on HT09 and PAMAP2, the backgrounds with ten or more bootstrap blocks:
# the mean shift of each row's planted maximum, in standard deviations of its highest scores over event-free windows
# (plants.json effect_table.cells; the five identity rows from identity_plants.json cells.<cell>.indep.rows, as in
# Appendix E). Cells: every rewiring that moves edges (not HT09's triangle-closing plant) and the relabelling.
RHO = 0.5
C_BG = (("HT09", "HT09", "o"), ("PM", "PAMAP2", "s"))
C_MODES = ("rand", "assort", "triclose", "swap")
C_ROWS = CORE + IDENTITY + WIRING
C_GROUPS = (len(CORE), len(IDENTITY), len(WIRING))
ID5 = ("hamming", "jaccard", "deltacon", "chenzhang", "nbs")
ET = PLANTS["effect_table"]["cells"]


def _moved(bg, mode):
    B = PLANTS["backgrounds"][bg]
    if mode == "swap":
        return 1.0
    hit = [c for c in B["e1"] if c["mode"] == mode and abs(c["rho"] - RHO) < 1e-12]
    assert len(hit) == 1, (bg, mode)
    return hit[0]["achieved"]["moved_edge_share_median"]


def _c_entry(bg, mode, r):
    key = f"{bg}/{mode}/{RHO}"
    if r in ID5:
        x, src = IDPL["cells"][key]["indep"]["rows"][r], f"identity_plants.json cells.{key}.indep.rows.{r}"
    else:
        x = ET[f"{bg}/{mode}/{RHO:g}"].get(r)
        src = f"plants.json effect_table.cells.{bg}/{mode}/{RHO:g}.{r}"
        if x is None:
            return None                               # TIRE was not run on the relabelling
    if not x["evaluable"]:
        return None                                   # tied in at least half the trials
    lo, hi = x["auc_paired_ci95"]
    return {"v": x["increment_null_sd"], "sig": lo > 0.5 or hi < 0.5, "src": src}


C_CELLS = [(bg, m, mk) for bg, _, mk in C_BG for m in C_MODES if _moved(bg, m) >= 0.01]
C_VAL = {r: [(m, mk, _c_entry(bg, m, r)) for bg, m, mk in C_CELLS] for r in C_ROWS}
NEGLIGIBLE = 0.1                                      # spec V4 V6: |increment| < 0.1 null SD

# (d) synthetic block models at matched expected degree, and s-GNN's vertex exchange: EP(row) - EP(counts) on the
# same events (identity_new.json synth.<scenario>.minus_counts.<row>), filled where the 95% interval excludes 0
D_SCEN = (("lad_sbm_matched", "LAD, matched", "o"), ("sgnn_merge_matched", "s-GNN merge, matched", "s"),
          ("sgnn_swap", "s-GNN exchange", "D"))
SYN = INEW["synth"]
SYN_ID = _load("identity_synth")["scenarios"]
D_VAL = {r: [(s, mk, SYN[s]["minus_counts"][r]) for s, _, mk in D_SCEN] for r in CORE + IDENTITY}
D_COUNTS = {s: SYN_ID[s]["rows"]["counts"] for s, _, _ in D_SCEN}

# ------------------------------------------------------------------------------------------------ style
_usetex = usetex_available()
cfg = bundles.tmlr2023(nrows=1, ncols=2)  # TMLR: \textwidth = 6.5 in (includegraphics width=\textwidth)
cfg.update(fs.rc(_usetex))
cfg["figure.constrained_layout.use"] = False
FIG_W = cfg["figure.figsize"][0]
S1 = 2.05                                              # side of the two panels of Figure 1
ROW, RGAP = 0.140, 0.10                                # dot-plot row height and the gap between row groups
C_W = D_W = 2.25                                      # widths of (c) and (d)
C_H = len(C_ROWS) * ROW + (len(C_GROUPS) - 1) * RGAP
Y1, TOP1 = 0.50, 0.50                                 # Figure 1: room for the x labels below, the legend above
FIG1_H = Y1 + S1 + TOP1
Y2, TOP2 = 0.62, 0.25                                 # Figure 2: x labels and the legend below, the titles above
FIG2_H = Y2 + C_H + TOP2
X_A, X_B = 0.50, 0.50 + S1 + 1.10
X_C = 0.95
X_D = X_C + C_W + 0.30
MINUS = fs.minus(_usetex)
TITLES = {"a": "Labelled events", "b": "EP by activity change", "c": "Rewiring at fixed degree",
          "d": "Block models"}
SHOWN = {"a": "a", "b": "b", "c": "a", "d": "b"}       # Figure 2 tags its panels (a) and (b)
OFFSETS = {  # dataset -> (dx_pt, dy_pt, ha) for the labels of (a); a long offset gets a leader line
    "UCI": (4, 8, "center"), "MAG": (-5, 0, "right"), "SCC": (-4, -16, "center"), "PM": (4, 5, "left"),
    "MIT": (6, 0, "left"), "ENRON": (-3, 7, "right"), "PS": (0, -8, "center"), "HT09": (-5, 0, "right"),
    "EEG": (-5, 0, "right"), "AskUbuntu": (-6, 6, "right"),
}


def inch_axes(fig, x, y, w, h):
    W, H = fig.get_size_inches()
    return fig.add_axes([x / W, y / H, w / W, h / H])


def panel_tag(ax, tag, y=1.04, x=-0.02):
    fs.panel_title(ax, SHOWN[tag], TITLES[tag], _usetex, x=x, y=y)


def colour_ticks(labels, keys):
    for t, r in zip(labels, keys):
        t.set_color(family_colour(r))
        t.set_fontweight("bold" if r == "counts" else "normal")


def row_tops(groups, step, gap):
    g_of_row = [g for g, k in enumerate(groups) for _ in range(k)]
    return [i * step + g_of_row[i] * gap for i in range(len(g_of_row))]


# ------------------------------------------------------------------------------------------------ plot
KW_LEG = {"frameon": False, "borderaxespad": 0.0, "handletextpad": 0.3, "labelcolor": INK}
with plt.rc_context(cfg):
    fig = plt.figure(figsize=(FIG_W, FIG1_H))
    ax_a = inch_axes(fig, X_A, Y1, S1, S1)
    ax_b = inch_axes(fig, X_B, Y1, S1, S1)

    # (a) ---------------------------------------------------------------------------------------------
    ax = ax_a
    vals = [v for ds in A_DS for v in (A[ds][0], A[ds][2])] + [v for t in A_POOL.values() for v in t]
    lo, hi = pad(min(vals + [0.5]), max(vals))
    lo, hi = min(0.36, np.floor(lo * 20) / 20), min(1.0, np.ceil(hi * 20) / 20)   # room for the labels on the left
    ax.plot([lo, hi], [lo, hi], color="#C9CDD5", lw=1.0, zorder=1, solid_capstyle="butt")
    ax.axhline(0.5, color=REF, ls=(0, (3.5, 2.5)), lw=0.7, zorder=1)
    ax.axvline(0.5, color=REF, ls=(0, (3.5, 2.5)), lw=0.7, zorder=1)
    for ds in A_DS:
        x, best, y = A[ds]
        _, col, mk = DETECTOR[best]
        ax.plot(x, y, ls="none", marker=mk, ms=5.4, mfc=col, mec="white", mew=0.5, zorder=4)
        dx, dy, ha = OFFSETS[ds]
        lead = {"arrowprops": {"arrowstyle": "-", "color": GREY, "lw": 0.5, "shrinkA": 1.0, "shrinkB": 3.0}} \
            if abs(dx) + abs(dy) > 15 else {}
        ax.annotate(DS_NAME[ds], (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    fontsize=SMALL, color=INK, path_effects=HALO, **lead)
    # pooled EP: counts (filled) and causal counts (open) against the best pooled other row (y); its interval is the
    # vertical whisker; the two horizontal whiskers are dodged by DODGE (display only) so that both stay visible
    px, plo, phi = A_POOL[A_POOL_BEST]
    col = DETECTOR[A_POOL_BEST][1]
    DODGE = 0.013
    ax.errorbar(A_POOL["counts"][0], px, yerr=[[px - plo], [phi - px]], fmt="none", zorder=3, ecolor=col, **WHISK)
    for ref, filled, dy in (("counts_causal", False, -DODGE), ("counts", True, DODGE)):
        cx, clo, chi = A_POOL[ref]
        ax.errorbar(cx, px + dy, xerr=[[cx - clo], [chi - cx]], fmt="none", zorder=3,
                    ecolor=BLUE if filled else "#9FB0FF", **WHISK)
        ax.plot(cx, px, ls="none", marker="o", ms=6.2, mfc=BLUE if filled else "white", mec=BLUE, mew=1.1,
                zorder=6 if filled else 5)
    for ref, dy, name, va in (("counts", DODGE, "counts", "bottom"), ("counts_causal", -DODGE, "causal counts", "top")):
        ax.annotate(name, (A_POOL[ref][2], px + dy), xytext=(3, 0), textcoords="offset points", ha="left",
                    va=va, fontsize=SMALL, color=NAVY, path_effects=HALO)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ticks = [t for t in np.arange(0.2, 1.01, 0.1) if lo - 1e-9 <= t <= hi + 1e-9]
    ax.set_xticks(ticks, [f"{t:.1f}" for t in ticks])
    ax.set_yticks(ticks, [f"{t:.1f}" for t in ticks])
    ax.set_xlabel("EP, counts")
    ax.set_ylabel("EP, best other method")
    fs.frame(ax, grid=None)
    panel_tag(ax, "a")

    # (b) ---------------------------------------------------------------------------------------------
    ax = ax_b
    xs = np.arange(len(STRATA))
    ax.axhline(0.5, color=REF, ls=(0, (3.5, 2.5)), lw=0.7, zorder=1)
    for r in B_ROWS:                                    # background: every other row, thin
        if r in B_HI:
            continue
        ax.plot(xs, [v[0] for v in B_VAL[r]], color=B_BG[B_FAM(r)][0], lw=0.8, alpha=0.9, zorder=2,
                ls=B_BG[B_FAM(r)][1])
    for r, dxs in zip(B_HI, (-0.07, 0.0, 0.07)):
        col = family_colour(r)
        v = np.array(B_VAL[r])
        x = xs + dxs
        ax.errorbar(x, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="none", ecolor=col, zorder=3,
                    **WHISK)
        ax.plot(x, v[:, 0], color=col, lw=1.5 if r == "counts" else 1.2, zorder=4,
                ls=(0, (2.5, 1.5)) if r in WIRING else "-", marker=DETECTOR[r][2], ms=4.2, mfc=col, mec="white",
                mew=0.5)
    ax.set_xlim(-0.3, len(STRATA) - 0.7)
    ax.set_ylim(0.24, 1.04)
    ax.set_xticks(xs, [f"{s}\n({n})" for s, n in zip(STRATA, B_N)])
    ax.tick_params(axis="x", length=0, pad=3)
    yt = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax.set_yticks(yt, [f"{t:.1f}" for t in yt])
    ax.set_ylabel("EP")
    fs.frame(ax)
    hb = [Line2D([], [], color=family_colour(r), lw=1.2, marker=DETECTOR[r][2], ms=3.6,
                 ls=(0, (2.5, 1.5)) if r in WIRING else "-") for r in B_HI]
    hb += [Line2D([], [], color=c, lw=0.8, ls=ls) for c, ls in B_BG.values()]
    ax.legend(hb, [DETECTOR[r][0] for r in B_HI] + list(B_BG_NAME.values()), loc="lower left", ncol=2,
              handlelength=1.6, borderaxespad=0.2, labelspacing=0.25, columnspacing=0.8, handletextpad=0.4,
              labelcolor=INK)
    panel_tag(ax, "b")

    # legend of (a): the detectors that are best on some dataset
    LEG = sorted({A[ds][1] for ds in A_DS} | {A_POOL_BEST}, key=lambda r: list(DETECTOR).index(r))
    hs = [Line2D([], [], ls="none", marker=DETECTOR[r][2], ms=5.4, mfc=DETECTOR[r][1], mec="white", mew=0.5)
          for r in LEG]
    fig.legend(hs, [DETECTOR[r][0] for r in LEG], ncol=len(LEG), loc="lower center",
               bbox_to_anchor=(0.5, (Y1 + S1 + 0.25) / FIG1_H), columnspacing=1.0, **KW_LEG)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    size1 = fig.get_size_inches()
    plt.close(fig)

    fig = plt.figure(figsize=(FIG_W, FIG2_H))
    ax_c = inch_axes(fig, X_C, Y2, C_W, C_H)
    ax_d = inch_axes(fig, X_D, Y2, D_W, C_H)

    # (c) and (d): one row per detector or structural statistic, grouped by family as in Table 3 ----------------
    y_top = row_tops(C_GROUPS, ROW, RGAP)
    Y_OF = {r: y + ROW / 2 for r, y in zip(C_ROWS, y_top)}

    def rows_axis(ax, labels: bool):
        ax.set_ylim(y_top[-1] + ROW, 0)
        for r in C_ROWS:                              # a faint guide line per row
            ax.axhline(Y_OF[r], color=fs.GRID, lw=0.5, zorder=0)
        if labels:
            ax.set_yticks([Y_OF[r] for r in C_ROWS], [DETECTOR[r][0] for r in C_ROWS])
            colour_ticks(ax.get_yticklabels(), list(C_ROWS))
        else:
            ax.set_yticks([])
        ax.tick_params(axis="y", length=0, pad=2.5)
        for sname in ("top", "right", "left"):
            ax.spines[sname].set_visible(False)

    ax = ax_c
    ax.axvspan(-NEGLIGIBLE, NEGLIGIBLE, color="#E4E9FF", zorder=0, lw=0)
    ax.axvline(0.0, color=REF, lw=0.7, zorder=1)
    for r in C_ROWS:
        col = family_colour(r)
        ents = [(m, mk, e) for m, mk, e in C_VAL[r] if e is not None]
        offs = np.linspace(-0.045, 0.045, len(ents)) if len(ents) > 1 else [0.0]
        for (m, mk, e), dy in zip(ents, offs):
            filled = m != "swap"
            ax.plot(e["v"], Y_OF[r] + dy, ls="none", marker=mk, ms=3.6, mew=0.8, mec=col,
                    mfc=col if filled else "white", alpha=1.0 if e["sig"] else 0.30, zorder=3)
    ax.set_xscale("symlog", linthresh=NEGLIGIBLE, linscale=0.8)
    ax.set_xlim(-0.6, 2.4)
    xt = [-0.3, -0.1, 0, 0.1, 0.3, 1, 2]
    ax.set_xticks(xt, [f"{MINUS}{abs(t):g}" if t < 0 else f"{t:g}" for t in xt])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("shift (SD of highest scores over event-free windows)")
    rows_axis(ax, True)
    panel_tag(ax, "c", y=1.0 + 0.10 / C_H, x=-0.02)

    ax = ax_d
    ax.axvline(0.0, color=REF, lw=0.7, zorder=1)
    for r in CORE + IDENTITY:
        col = family_colour(r)
        for (s_, mk, e), dy in zip(D_VAL[r], (-0.045, 0.0, 0.045)):
            lo_, hi_ = e["ci95"]
            sig = lo_ > 0 or hi_ < 0
            ax.errorbar(e["diff"], Y_OF[r] + dy, xerr=[[e["diff"] - lo_], [hi_ - e["diff"]]], fmt="none",
                        ecolor=col, alpha=1.0 if sig else 0.30, zorder=2, elinewidth=0.6, capsize=0)
            ax.plot(e["diff"], Y_OF[r] + dy, ls="none", marker=mk, ms=3.4, mew=0.8, mec=col,
                    mfc=col if sig else "white", alpha=1.0 if sig else 0.45, zorder=3)
    for r in WIRING:
        ax.text(0.0, Y_OF[r], "not run", ha="center", va="center", fontsize=SMALL, color=FAINT)
    ax.set_xlim(-0.42, 0.75)
    xt = [-0.25, 0, 0.25, 0.5]
    ax.set_xticks(xt, [f"{MINUS}{abs(t):g}" if t < 0 else f"{t:g}" for t in xt])
    ax.set_xlabel("EP minus EP of the counts")
    fs.frame(ax, grid=None)
    rows_axis(ax, False)
    panel_tag(ax, "d", y=1.0 + 0.10 / C_H, x=-0.02)

    mk_ = {"ls": "none", "ms": 4.0, "mew": 0.8, "mec": INK}
    hs2 = [Line2D([], [], marker="o", mfc=INK, **mk_), Line2D([], [], marker="s", mfc=INK, **mk_),
           Line2D([], [], marker="o", mfc="white", **mk_),
           Line2D([], [], marker="s", ls="none", ms=7.5, mfc="#E4E9FF", mec="none"),
           Line2D([], [], marker="o", mfc=INK, **mk_), Line2D([], [], marker="s", mfc=INK, **mk_),
           Line2D([], [], marker="D", mfc=INK, **mk_)]
    lb2 = ["(a) HT09", "(a) PAMAP2", "(a) open: relabelling", "(a) below 0.1 SD",
           "(b) LAD's generator", "(b) s-GNN merge", "(b) s-GNN exchange"]
    fig.legend(hs2, lb2, ncol=4, loc="upper center", bbox_to_anchor=(0.5, (Y2 - 0.40) / FIG2_H),
               columnspacing=1.0, **KW_LEG)
    fig.savefig(OUT2, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    size2 = fig.get_size_inches()
    plt.close(fig)

if ARGS.png and shutil.which("pdftoppm"):
    for o in (OUT, OUT2):
        subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", str(o), str(o.with_suffix(""))], check=False)

# ------------------------------------------------------------------------------------------------ audit log
print(f"usetex={_usetex} figure sizes {tuple(round(float(v), 3) for v in size1)}, "
      f"{tuple(round(float(v), 3) for v in size2)} in -> {OUT}, {OUT2}")
print("(a) per-dataset EP (bench / identity / identity_new / activity datasets.<ds>): counts | best other row")
for ds in A_DS:
    x, best, y = A[ds]
    print(f"    {DS_NAME[ds]:10s} counts {x:.3f} | {best:13s} {y:.3f}")
for r, (m, lo_, hi_) in A_POOL.items():
    print(f"    inference.json pools.all.equal: {r} {m:.3f} [two-stage {lo_:.3f}, {hi_:.3f}]")
print(f"(b) activity.json C_strata.all.<stratum>.<row>: EP [95% CI]; n events {B_N}, datasets {B_NDS}; "
      f"highlighted {B_HI}")
for r in B_ROWS:
    print(f"    {DETECTOR[r][0]:13s} " + "  ".join(f"{m:.3f} [{lo_:.3f}, {hi_:.3f}]" for m, lo_, hi_ in B_VAL[r]))
print(f"(c) shift in event-free SD, rho={RHO}; cells {[(bg, m) for bg, m, _ in C_CELLS]}; '*' paired CI excludes 0.5")
for r in C_ROWS:
    print(f"    {DETECTOR[r][0]:15s} " + " ".join("  -   " if e is None else f"{e['v']:+.3f}{'*' if e['sig'] else ' '}"
                                       for _, _, e in C_VAL[r]))
print("(d) identity_new.json synth.<scenario>.minus_counts.<row>: row - counts [95% CI]; counts EP "
      + json.dumps({s_: round(v["ep"], 3) for s_, v in D_COUNTS.items()}))
for r in CORE + IDENTITY:
    print(f"    {DETECTOR[r][0]:15s} " + "  ".join(f"{s_} {e['diff']:+.3f} [{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}]"
                                         for s_, _, e in D_VAL[r]))
