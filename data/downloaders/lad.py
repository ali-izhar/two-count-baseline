"""LAD real-world series (UCI, USLegis = 'Senate co-sponsorship', canVote) from the LAD authors' repository.

Raw: data/raw/lad/ = github.com/shenyangHuang/LAD @ 58330238 (datasets/*_processed). Snapshot rules copy the LAD
loaders: one snapshot per change of the time key in file order. Stored graph = undirected simple graph (self-loops
dropped). Extra arrays aligned with `indices`: `weights` = LAD directed weight w(u,v) + w(v,u) (LAD keeps the last
weight per directed pair and snapshot, nx.add_edge overwrite); for UCI also `msgs` = messages between the pair.
"""
import datetime as dt
import os
import re

import numpy as np
import scipy.sparse as sp

from data.downloaders.snapshots import check, sha256

REPO = "https://github.com/shenyangHuang/LAD"
COMMIT = "58330238a1e907d450a3354e7f2f786b53d54262"
RAWURL = f"https://raw.githubusercontent.com/shenyangHuang/LAD/{COMMIT}/"


def build(snaps, n):
    """snaps: list of dict {(u,v): w} directed (LAD last-wins), plus optional msg counts. Returns mats, extras."""
    mats, W, M = [], [], []
    for d, cnt in snaps:
        sym, msym = {}, {}
        for (u, v), w in d.items():
            if u == v:
                continue
            k = (min(u, v), max(u, v))
            sym[k] = sym.get(k, 0) + w
        if cnt is not None:
            for (u, v), c in cnt.items():
                if u != v:
                    k = (min(u, v), max(u, v))
                    msym[k] = msym.get(k, 0) + c
        keys = sorted(sym)
        a = np.array([k[0] for k in keys], np.int64)
        b = np.array([k[1] for k in keys], np.int64)
        wv = np.array([sym[k] for k in keys], np.float64)
        mv = np.array([msym.get(k, 0) for k in keys], np.float64)
        r, c = np.concatenate([a, b]), np.concatenate([b, a])
        Aw = sp.csr_matrix((np.concatenate([wv, wv]), (r, c)), shape=(n, n))
        Am = sp.csr_matrix((np.concatenate([mv, mv]) + 0.5, (r, c)), shape=(n, n))  # +0.5 keeps zero counts stored
        Aw.sort_indices(); Am.sort_indices()
        A = Aw.copy(); A.data[:] = 1; A = A.astype(np.int8)
        check(A)
        assert np.array_equal(A.indices, Am.indices) and np.array_equal(A.indptr, Am.indptr)
        mats.append(A); W.append(Aw.data); M.append(Am.data - 0.5)
    ex = {"weights": np.concatenate(W).astype(np.float32)}
    if snaps[0][1] is not None:
        ex["msgs"] = np.concatenate(M).astype(np.float32)
    return mats, ex


def uci(raw):
    f = os.path.join(raw, "datasets/UCI_processed/OCnodeslinks_chars.txt")
    snaps, dates, cur, d, cnt = [], [], None, None, None
    num = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")
    with open(f) as fh:
        for line in fh:
            vals = num.findall(line)
            if len(vals) < 3:
                continue
            ds = re.search(r"\d{4}-\d{2}-\d{2}", line).group(0)
            if ds != cur:
                if cur is not None:
                    snaps.append((d, cnt))
                cur, d, cnt = ds, {}, {}
                dates.append(ds)
            u, v, w = int(vals[-3]), int(vals[-2]), int(vals[-1])
            d[(u, v)] = w
            cnt[(u, v)] = cnt.get((u, v), 0) + 1
    snaps.append((d, cnt))
    n = 1 + max(max(max(k) for k in s[0]) for s in snaps)
    mats, ex = build(snaps, n)
    T = len(mats)
    assert T == 196, T
    d0 = dt.date.fromisoformat(dates[0])
    cal = [(dt.date.fromisoformat(x) - d0).days for x in dates]
    ev = [65, 158]
    meta = {
        "id": "UCI", "n": int(n), "T": T, "directed_source": True,
        "snapshot_rule": ("one snapshot per distinct calendar date (yyyy-mm-dd of the message timestamp) in file order, "
                       "as LAD UCI_loader.load_temporarl_edgelist; calendar days without any message are skipped, so "
                       "index != calendar offset. Source rows include account-creation self-edges (u,u,1); dropped here. "
                       "LAD pads every snapshot to max_nodes=1901 and uses nx.directed_laplacian_matrix on the weighted "
                       "digraph (weight = characters of the last message u->v that day)."),
        "dates": dates, "calendar_offset_days": cal, "n_calendar_gaps": int(sum(1 for i in range(1, T) if cal[i] - cal[i-1] > 1)),
        "windows": {"lad_short": 7, "lad_long": 14, "lad_initial": 14, "lad_difference": True, "lad_num_eigen": 6,
                     "lad_percent_ranked": 0.05},
        "events": [{"index": 65, "date": dates[65], "desc": "end of spring term",
                     "citation": "Panzarasa, Opsahl, Carley 2009 JASIST 60(5):911-932 (as cited by LAD KDD 2020 Sec 5.5.1 "
                              "and MultiLAD Sec 7.4.1; index 0-based, LAD code comment real_events=[65,158])"},
                {"index": 158, "date": dates[158], "desc": "start of fall term",
                     "citation": "Panzarasa, Opsahl, Carley 2009 (as cited by LAD; LAD reports detecting day 157, one day "
                              "before, which Hits@10 counts as a miss)"}],
        "tol": 0, "metric": "Hits@10: fraction of the 2 events among the top round(0.05*196)=10 scores (exact index match)",
    }
    return "UCI", mats, ev, meta, ex, [f]


def legis(raw):
    f = os.path.join(raw, "datasets/USLegis_processed/LegisEdgelist.txt")
    snaps, keys, cur, d = [], [], None, None
    with open(f) as fh:
        for line in fh:
            v = line.strip().split(",")
            if len(v) < 4:
                continue
            t, a, b, w = v[0], int(v[1]), int(v[2]), int(v[3])
            if t != cur:
                if cur is not None:
                    snaps.append((d, None))
                cur, d = t, {}
                keys.append(t)
            d[(b, a)] = w  # LAD: v=values[1], u=values[2], add_edge(u, v) on an undirected nx.Graph
    snaps.append((d, None))
    ids = sorted({x for s in snaps for k in s[0] for x in k})
    pos = {x: i for i, x in enumerate(ids)}
    snaps = [({(pos[u], pos[v]): w for (u, v), w in s.items()}, None) for s, _ in snaps]
    # undirected source: LAD's nx.Graph keeps one weight per pair; build() sums both orientations, so dedupe first
    snaps = [({(min(u, v), max(u, v)): w for (u, v), w in s.items()}, None) for s, _ in snaps]
    mats, ex = build(snaps, len(ids))
    T = len(mats)
    assert T == 12, T
    congress = list(range(97, 109))
    meta = {
        "id": "USLegis", "n": len(ids), "T": T, "directed_source": False, "node_ids": ids,
        "paper_name": "US Senate co-sponsorship network (LAD KDD 2020 Sec 5.5.2; MultiLAD Sec 7.4.2)",
        "snapshot_rule": ("one snapshot per value of the first column of LegisEdgelist.txt (0..11), i.e. one per "
                       "Congress 97..108 (LAD plot labels range(97,109)); the paper text says 'biannually', a Congress "
                       "spans two years. Weight = file column 4 (number of co-sponsored bills per pair, per LAD loader "
                       "comment 'shared publications'). LAD pads to 102 nodes, undirected weighted Laplacian, top-6 "
                       "singular values. Node ids are the file's integer ids, remapped to 0..n-1 over the union."),
        "keys": keys, "congress": congress,
        "windows": {"lad_short": 1, "lad_long": 2, "lad_initial": 2, "lad_difference": True, "lad_num_eigen": 6,
                     "lad_percent_ranked": 0.20},
        "events": [{"index": 3, "congress": 100, "years": "1987-1989", "desc": "highest clustering coefficient (peak collaboration)",
                     "citation": "Wang, Chakrabarti, et al. IJCAI 2017 (EdgeMonitoring), as cited by LAD Sec 5.5.2"},
                {"index": 7, "congress": 104, "years": "1995-1997", "desc": "'Republican Revolution'; lowest clustering coefficient",
                     "citation": "Fowler 2006 Social Networks 28(4):454-465; Wang et al. 2017; as cited by LAD Sec 5.5.2"}],
        "tol": 0, "metric": "Hits@2: fraction of the 2 events among the top round(0.20*12)=2 scores",
        "density_note": ("every snapshot is (almost) a complete graph on its ~100 active senators (e.g. 5050 edges on "
                      "101 nodes); the unweighted topology is near constant, all signal is in `weights`."),
        "upstream": "Fowler cosponsorship data (jhfowler.ucsd.edu/cosponsorship.htm, unreachable 2026-10-23); the file "
                 "came to LAD via Wang et al. 2017 (EdgeMonitoring). Provenance of the integer node ids not verified.",
    }
    return "USLegis", mats, [3, 7], meta, ex, [f]


def canvote(raw):
    f = os.path.join(raw, "datasets/canVote_processed/canVote_edgelist.txt")
    snaps, keys, cur, d = [], [], None, None
    with open(f) as fh:
        for line in fh:
            v = line.strip().split(",")
            if len(v) < 4:
                continue
            t, a, b, w = v[0], v[1], v[2], int(v[3])
            if t != cur:
                if cur is not None:
                    snaps.append((d, None))
                cur, d = t, {}
                keys.append(t)
            d[(b, a)] = w  # LAD: v=values[1], u=values[2], DiGraph add_edge(u, v)
    snaps.append((d, None))
    ids = sorted({x for s in snaps for k in s[0] for x in k})
    pos = {x: i for i, x in enumerate(ids)}
    snaps = [({(pos[u], pos[v]): w for (u, v), w in s.items()}, None) for s, _ in snaps]
    act = [len({x for k in s for x in k}) for s, _ in snaps]
    mats, ex = build(snaps, len(ids))
    T = len(mats)
    assert T == 14 and keys == [str(y) for y in range(2006, 2020)], keys
    meta = {
        "id": "canVote", "n": len(ids), "T": T, "directed_source": True, "node_ids": ids, "active_nodes_per_year": act,
        "paper_name": "Canadian bill voting network (LAD KDD 2020 Sec 5.5.3, App C; MultiLAD Sec 7.4.3, App A.6)",
        "snapshot_rule": ("one snapshot per year 2006..2019 (first column). Directed edge voter->sponsor in the paper; "
                       "LAD loader adds edge values[2]->values[1]; weight = number of yes votes of the voter for bills "
                       "of the sponsor in that year. LAD uses only the MPs active that year (node set varies), pads to the "
                       "largest yearly node count, directed weighted Laplacian, top-338 singular values. Here: union "
                       "node set (n), undirected, self-loops dropped. Some node ids are bare integers (e.g. '18', '20'), "
                       "not MP slugs; their identity is not documented in the repo."),
        "keys": keys,
        "windows": {"lad_short": 2, "lad_long": 4, "lad_initial": 4, "lad_difference": True, "lad_num_eigen": 338,
                     "lad_percent_ranked": 0.154},
        "events": [{"index": 7, "year": 2013, "desc": "Justin Trudeau elected Liberal leader; cross-party cohesion against "
                     "Conservatives begins to decline", "citation": "Marland 2013 (CCA/CPSA annual meeting), as cited by LAD "
                     "Sec 5.5.3 [31]; NOTE: named in the paper as a point LAD detected, not as a priori ground truth"},
                {"index": 9, "year": 2015, "desc": "Oct 19 2015 federal election, Liberal majority; seats 308->338",
                     "citation": "Cross 2016 Can J Pol Sci 49(4):601-620, as cited by LAD Sec 5.5.3 [13]; same caveat"}],
        "events_secondary_note": ("LAD Anomaly_Detection.canVote() carries a commented-out real_events=[5,9] = years "
                               "2011 and 2015 (2011: May 2 2011 election, Conservative majority; this reading is ours, the "
                               "code gives no label). The papers report no Hits number for canVote."),
        "tol": 0, "metric": "none reported; the code ranks the top round(0.154*14)=2 years",
        "activity_note": ("the election years 2011 and 2015 have 414 and 476 active nodes against 306-340 in other "
                       "years (outgoing and incoming MPs both vote that year)."),
    }
    return "canVote", mats, [7, 9], meta, ex, [f], [5, 9]


PAPERS = {
    "lad": "Huang, Hitti, Rabusseau, Rabbany. Laplacian Change Point Detection for Dynamic Graphs. KDD 2020. "
        "arXiv:2007.01229, doi:10.1145/3394486.3403077",
    "multilad": "Huang, Coulombe, Hitti, Rabbany, Rabusseau. Laplacian Change Point Detection for Single and Multi-view "
             "Dynamic Graphs. TKDD 2024, arXiv:2302.01204",
}
RESULTS = {
    "UCI": {"metric": "Hits@10", "source": "LAD KDD 2020 Table 5; identical in MultiLAD TKDD Table 3",
                "values": {"LAD": 0.50, "EdgeMonitoring": 0.0, "Activity vector": 0.50, "TENSORSPLAT": 0.0},
                "notes": "LAD top-10 detections shown in LAD Fig 6: 65, 73, 94, 122, 142, 157, 177, 180, 181, 191 "
                      "(read off figure annotations; text: LAD hits 65, misses 158 with 157)"},
    "USLegis": {"metric": "Hits@2", "source": "LAD KDD 2020 Table 5 ('Senate'); identical in MultiLAD Table 3",
                    "values": {"LAD": 1.00, "EdgeMonitoring": 1.00, "Activity vector": 0.50, "TENSORSPLAT": 0.50},
                    "notes": "Text also: LetoChange (Peel & Clauset) detects both and DeltaCon only the 104th, per Wang "
                          "et al. 2017 (not in the table). Activity vector inside the LAD two-window pipeline detects both."},
    "canVote": {"metric": "none", "source": "LAD KDD 2020 Sec 5.5.3 / Fig 8; MultiLAD Fig 6",
                    "values": {}, "notes": "No table entry. LAD (top-338 singular values) ranks 2013 and 2015 as the two "
                                     "most anomalous years."},
}


def series(raw_dir):
    """One dict per dataset (id, mats, seg, events, tol, meta, extra) for UCI, USLegis, canVote."""
    out = []
    for fn in (uci, legis, canvote):
        r = fn(raw_dir)
        ID, mats, ev, meta, ex, files = r[:6]
        meta["source"] = {"repo": REPO, "commit": COMMIT,
                              "files": [{"url": RAWURL + os.path.relpath(p, raw_dir), "sha256": sha256(p)} for p in files]}
        meta["papers"] = PAPERS
        meta["used_by"] = ("LAD KDD 2020 and MultiLAD TKDD 2024 (same single-view protocol: Z* = max of differenced "
                           "short/long window cosine scores of L2-normalized Laplacian singular values; startup "
                           "period scored 0; Hits@n with n = round(percent_ranked*T))")
        meta["paper_results"] = RESULTS[ID]
        meta["edges_per_snapshot"] = [int(A.nnz // 2) for A in mats]
        extra = dict(ex)
        if len(r) > 6:
            extra["events_secondary"] = np.asarray(r[6], np.int64)
        out.append({"id": ID, "mats": mats, "seg": np.zeros(len(mats), np.int64), "events": ev, "tol": meta["tol"], "meta": meta,
                        "extra": extra})
    return out
