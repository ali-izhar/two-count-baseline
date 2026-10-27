"""MIT and ENRON as benchmark-B members: the repo's own Table 1 graphs (repro/data/graphs_<ds>.npz, step 1).

Self-loops are removed (none present). The meta records the repo events and, for comparison, the events of
Peel & Clauset (AAAI 2015) Fig. 8 read from the vector drawing at week resolution.
"""
import datetime as dt

import numpy as np
import scipy.sparse as sp

from data.downloaders.mit_reality import MIT_EVENTS
from data.downloaders.snapshots import sha256

PEEL = "Peel & Clauset, AAAI 2015 (arXiv:1403.0989)"
ANCHOR = dt.date(2001, 2, 12) - dt.timedelta(days=dt.date(2001, 2, 12).weekday())   # Monday of ENRON index 124


def repo_series(repo, ds):
    z = np.load(f"{repo}/repro/data/graphs_{ds}.npz", allow_pickle=False)
    n = z["n"]
    assert (n == n[0]).all()
    n = int(n[0])
    ip = np.concatenate([[0], np.cumsum(z["indptr_len"])])
    ix = np.concatenate([[0], np.cumsum(z["nnz"])])
    mats = []
    for t in range(len(z["n"])):
        A = sp.csr_matrix((np.ones(ix[t + 1] - ix[t], np.int8), z["indices"][ix[t]:ix[t + 1]],
                           z["indptr"][ip[t]:ip[t + 1]]), shape=(n, n))
        A.setdiag(0)
        A.eliminate_zeros()
        A.sort_indices()
        mats.append(A)
    return mats, z["true_cps"].copy(), sha256(f"{repo}/repro/data/graphs_{ds}.npz")


def monday(d):
    return d - dt.timedelta(days=d.weekday())


# Peel & Clauset Fig. 8 (page 9), read from the vector drawing (bench/BO/peel_fig.py -> peel_fig.json).
# x positions -> weeks from the tick marks (MIT: 13.655 pt/week from 2004-09-06; Enron: 2.95 pt/week from 1999-05-10).
# Labels are paired with lines by colour; colours cycle in chronological order. Resolution: one week.
PEEL_MIT = [  # (label, first week, last week or None)  16 distinct boundaries
    ("Start of Semester (fall)", "2004-09-06", None), ("Sponsor Week", "2004-10-18", "2004-10-25"),
    ("Exam Week", "2004-11-08", "2004-11-15"), ("Thanksgiving", "2004-11-26", None),
    ("Last Week of Classes", "2004-12-06", "2004-12-13"), ("Finals Week", "2004-12-13", "2004-12-20"),
    ("Independent Activities Period", "2005-01-03", "2005-01-31"), ("MLK Day", "2005-01-17", None),
    ("Start of Semester (spring)", "2005-01-31", None), ("Exam Week", "2005-02-28", "2005-03-07"),
    ("Spring Break", "2005-03-21", "2005-03-28")]
PEEL_ENRON_X = [  # (label, x in pt); week = (x - 72.33) / 2.95 after 1999-05-10
    ("Launch of Enron online", 157.88), ("Launch of EBS", 178.53), ("EBS-Blockbusters partnership", 255.24),
    ("Stock all-time high", 269.99), ("Enron attorney discusses Belden's strategies", 287.69),
    ("FERC exonerates Enron", 299.49), ("EBS $53m 'profit'", 317.19), ("Skilling announced to be CEO", 323.09),
    ("Blackouts in CA", 331.94), ("Enron schedules conference call to boost stock", 358.49),
    ("The 'asshole' call", 370.29), ("Schwarzenegger, Lay, Milken meeting", 382.09),
    ("Quarterly conference call", 393.89), ("Skilling meets with analysts and investors in NY", 408.64),
    ("Watkins raises accounting irregularities", 424.24), ("Dynegy agrees to buy Enron", 457.32),
    ("Enron restates its third quarter earnings", 461.75), ("Enron shares plunge below $1", 465.96),
    ("Dynegy deal collapses", 466.38), ("Enron files for bankruptcy", 467.64),
    ("Stephen Cooper takes over as Enron CEO", 491.25), ("Lay cancels Senate committee appearance", 494.20),
    ("Fastow, Kopper, Lay invoke the Fifth", 497.15), ("Skilling and Watkins testify", 497.15),
    ("Arthur Andersen LLP indicted", 508.95)]

def idx_of(d):  # ASSUMES consecutive ISO weeks (no empty week) in the repo series
    return 124 + (monday(d) - ANCHOR).days // 7


def series(repo):
    out = []
    # MIT ---------------------------------------------------------------
    mats, cps, h = repo_series(repo, "mit_reality")
    assert len(cps) == len(MIT_EVENTS)
    repo_mit = [(nm, d, c, int(i)) for (nm, (d, c)), i in zip(MIT_EVENTS.items(), cps)]
    seg = np.zeros(len(mats), np.int64)
    meta = {
        "id": "MIT", "n": mats[0].shape[0], "T": len(mats), "segments": [{"segment": 0, "n_snapshots": len(mats)}],
        "windows": {"w": 7, "w_ref": 7, "delta": 5, "H": 10, "note": "legacy windows of repro/prereg DEV 19; paper tolerance 5"},
        "snapshot_rule": "daily graph; edge iff >=1 Bluetooth proximity row with prob2 >= 0.3 that day; days kept iff "
                      ">5 rows and >1 edge (data/downloaders/mit_reality.py); vertex set = all 63 user ids; self-loops removed (none present)",
        "events": [{"index": i, "date": d, "desc": nm, "citation": c} for nm, d, c, i in repo_mit],
        "event_rule": ("data/downloaders/mit_reality.py: the first day the MIT academic calendar 2008-2009 marks as a holiday or "
                       "vacation (Christmas, New Year: 5 U.S.C. 6103), or the first day after the last day of classes; index = "
                       "the first kept day on or after the date (Christmas -> 2008-12-26: no kept day 22-25 Dec)"),
        "source_url": "http://realitycommons.media.mit.edu/ (Proximity.csv; manual download behind a registration form)",
        "raw_sha256": {"Proximity.csv": "519bd39e64ce76d2975e4054af83b9ec2772604d70763cc415738ac330657191 (data/raw/mit_reality/)",
                        "graphs_npz": h, "cache_pkl": sha256(f"{repo}/data/cache/mit_reality.pkl")},
        "dataset_identity": ("NOT the Reality Mining 2004-05 study used by Peel & Clauset: the Reality Commons 'Social Evolution' "
                          "study (Madan et al., IEEE Pervasive Computing 11(4), 2012; one undergraduate dormitory), file "
                          "Proximity.csv (columns user.id / remote.user.id.if.known / prob2); kept days 2008-01-30 .. 2009-06-24."),
        "label_corrections": ["Columbus Day 2008-10-12 -> 2008-10-13 (index 19 -> 20)",
                              "Thanksgiving 2008-11-26 -> 2008-11-27 (64 -> 65)",
                              "Spring Break 2009-03-15 -> 2009-03-23 (168 -> 176)",
                              "End of Semester 2009-05-15 kept as End of Classes (last day of classes 2009-05-14)"],
        "peel_clauset_events": {
            "paper": PEEL, "dataset": "Reality Mining (Eagle & Pentland 2006), 97 subjects, WEEKLY graphs, 35 weeks (Aug 2004 - May 2005), w=4",
            "n_events_stated": 16, "label_source_stated": "Eagle, PhD thesis MIT 2005 (ref. [11])",
            "events_from_fig8": [{"desc": a, "week_start": b, "week_end": c} for a, b, c in PEEL_MIT],
            "distinct_boundaries": 16,
            "note": "Fig. 8 shows 11 labelled periods; their 16 distinct start/end week boundaries match the 16 events stated "
                 "in the text. Read from the vector figure at week resolution; Thanksgiving line lies at 25-27 Nov 2004.",
            "comparison": "No overlap: different study (2004-05 vs 2008-09), different subjects, weekly vs daily graphs. "
                       "The Peel events cannot be mapped onto this series. Reality Mining itself needs the registration form."},
        "used_by": [{"paper": "this repo, paper/main.tex Table 1", "use": "daily graphs, 6 events, tol 5 one-sided"},
                 ],
    }
    out.append({"id": "MIT", "mats": mats, "seg": seg, "events": list(cps), "tol": 5, "meta": meta, "extra": {}})

    # ENRON -------------------------------------------------------------
    mats, cps, h = repo_series(repo, "enron")
    # Repo events (data/downloaders/enron.py) mapped to ISO weeks; index 124 = ISO week 2001-W07 (Skilling CEO, 12 Feb 2001).
    repo_en = [("Skilling becomes CEO", "2001-02-12"), ("Skilling resigns", "2001-08-14"), ("$618M loss reported", "2001-10-16"),
               ("SEC probe announced", "2001-10-22"), ("Dynegy merger fails", "2001-11-28"), ("Bankruptcy filed", "2001-12-02")]


    rep = [{"desc": a, "date": b, "index_iso_week": idx_of(dt.date.fromisoformat(b))} for a, b in repo_en]
    assert sorted({r["index_iso_week"] for r in rep}) == [124, 150, 159, 160, 165], rep
    peel = []
    for lab, x in PEEL_ENRON_X:
        w = (x - 72.33) / 2.95
        day = dt.date(1999, 5, 10) + dt.timedelta(days=round(7 * w))
        peel.append({"desc": lab, "x_pt": x, "week_from_1999_05_10": round(w, 2), "date_read": day.isoformat(),
                         "week_monday": monday(day).isoformat(), "index_in_this_series": idx_of(day)})
    ev_peel = sorted({p["index_in_this_series"] for p in peel if 0 <= p["index_in_this_series"] < len(mats)})
    seg = np.zeros(len(mats), np.int64)
    meta = {
        "id": "ENRON", "n": mats[0].shape[0], "T": len(mats), "segments": [{"segment": 0, "n_snapshots": len(mats)}],
        "windows": {"w": 2, "w_ref": 2, "delta": 4, "H": 8, "note": "legacy windows of repro/prereg DEV 19; paper tolerance 4"},
        "snapshot_rule": "weekly (ISO week %G-W%V) undirected graph; edge iff >=1 email sender->To/Cc/Bcc recipient that week, "
                      "dates 1998-01-01..2003-01-01; vertex set = all 86,664 addresses of the corpus; weeks with no email are "
                      "absent (data/downloaders/enron.py); self-loops removed (none present)",
        "events": [{"index": int(i), "citation": "repo config.ENRON_CPS_MERGED; dates in data/downloaders/enron.py "
                     "('verified against SEC filings', no per-date source stored)"} for i in cps],
        "events_unmerged": rep,
        "events_secondary_desc": "Peel & Clauset Fig. 8 events mapped to this series (see peel_clauset_events)",
        "source_url": "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz",
        "raw_sha256": {"note": "raw tarball not on disk (data/raw absent); hashes are of the repo intermediates",
                        "graphs_npz": h, "cache_pkl": sha256(f"{repo}/data/cache/enron.pkl")},
        "index_to_date": ("index 124 = ISO week 2001-W07; the repo stores no timestamps. The spacing 124->150->159/160->165 "
                       "matches consecutive ISO weeks for the 6 repo dates, so index t = Monday 2001-02-12 + 7*(t-124) days "
                       "is ASSUMED for all t (not verifiable without the raw corpus)."),
        "peel_clauset_events": {
            "paper": PEEL, "dataset": "151 users (FERC-released cleaned version, Klimt & Yang 2004), weekly, May 1999 - June 2002, w=4,8,16",
            "n_events_stated": 25, "events_from_fig8": peel,
            "note": "read from the vector figure; most lines sit on week ticks, four (Watkins, Dynegy agrees, shares < $1, "
                 "Dynegy collapses) sit between ticks, so their date_read is approximate to +-1 day",
            "comparison": ("Same corpus and overlapping period, different vertex set (86,664 addresses vs 151 users). "
                        "Peel's 25 events include Dynegy deal collapses and the bankruptcy (also repo events); Peel has no "
                        "'Skilling resigns' (14 Aug 2001), no '$618M loss' (16 Oct 2001) and no 'SEC probe' (22 Oct 2001); "
                        "Peel marks 'Skilling announced to be CEO' in the week of 2000-12-25, the repo uses 'Skilling becomes "
                        "CEO' 2001-02-12. " + f"{sum(p['week_monday'] < '2001-02-12' for p in peel)} of Peel's 25 events fall before the repo's first event week.")},
        "used_by": [{"paper": "this repo, paper/main.tex Table 1", "use": "weekly graphs, 4 merged events, tol 4 one-sided"},
                 {"paper": PEEL, "use": "151-user version, 25 events, precision/recall vs delay"}],
    }
    out.append({"id": "ENRON", "mats": mats, "seg": seg, "events": list(cps), "tol": 4, "meta": meta,
                    "extra": {"events_secondary": np.asarray(ev_peel, np.int64)}})
    return out
