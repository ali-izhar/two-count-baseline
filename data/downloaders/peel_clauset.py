"""The two constructions of Peel & Clauset (AAAI 2015, arXiv:1403.0989): MIT_PC and ENRON_PC, weekly, with all of
their events.

MIT_PC (P&C: 97 faculty and graduate students, weekly, 35 weeks, 16 events, w = 4)
  raw   data/raw/konect_mit/download.tsv.mit.tar.bz2, the public KONECT copy of the Reality Mining Bluetooth contacts
        (http://konect.cc/files/download.tsv.mit.tar.bz2; 96 subjects, 2004-09-14 .. 2005-05-05). The original
        realitymining.mat sits behind a registration form.
  rule  week = Sunday .. Saturday in US/Eastern time (P&C's own dendrogram files are named by Sundays), first week
        2004-09-12; edge iff >= 1 contact in the week; vertex set = the 96 subjects of the file.
  events the 16 distinct boundary dates of the 11 periods of P&C Fig. 8 (term calendar, Eagle's thesis), each in
        the week that contains it; "start of fall semester" (2004-09-06) lies before the first week of the copy.
ENRON_PC (P&C: emails among 151 users, weekly snapshots May 1999 - June 2002, 25 events)
  raw   data/raw/enron_cmu/enron_mail_20150507.tar.gz (https://www.cs.cmu.edu/~enron/), read inside the tarball;
        data/raw/peel_code/{addresses.txt, enron_parser.py, enrontimeline.txt} from Peel's archived homepage
        (github.com/piratepeel/piratepeel.github.io, old/code/).
  rule  the logic of Peel's enron_parser.py: every message file of every folder; sender = the From address,
        receivers = the To header with its continuation lines (the parser stops at Subject, so Cc / Bcc are not
        used); a link needs both ends in the 151 addresses; undirected, no self-loops, binary per week; week k =
        [Mon 1999-05-10 + 7k, + 7 days) in UTC, k = 0 .. 163 (through June 2002). Two departures from the published
        script, both needed for it to do what it says: it strips the characters of "From: " from both ends of the
        sender (str.strip), which turns "...@enron.com" into "...@enron.c" so no sender would match; and it adds a
        link to the next written week however many weeks later it falls. Messages before week 0 are dropped.
  events the 25 events of P&C Fig. 8 at the plotted week (read from the vector figure); 25 events fall in 23
        distinct weeks. Secondary: the same 25 at the date in Peel's timeline file (floor of the week).
"""
from __future__ import annotations

import datetime as dt
import email.utils
import tarfile
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from data.downloaders.snapshots import sha256, to_csr

MIT_FIRST = dt.date(2004, 9, 12)
MIT_EVENTS = [  # (date, boundary of the P&C Fig. 8 period)
    ("2004-09-06", "start of fall semester"), ("2004-10-18", "sponsor week begins"), ("2004-10-25", "sponsor week ends"),
    ("2004-11-08", "exam week begins"), ("2004-11-15", "exam week ends"), ("2004-11-26", "Thanksgiving"),
    ("2004-12-06", "last week of classes begins"), ("2004-12-13", "last week of classes ends / finals begin"),
    ("2004-12-20", "finals end"), ("2005-01-03", "IAP begins"), ("2005-01-17", "MLK day"),
    ("2005-01-31", "IAP ends / spring semester begins"), ("2005-02-28", "exam week begins"),
    ("2005-03-07", "exam week ends"), ("2005-03-21", "spring break begins"), ("2005-03-28", "spring break ends")]

ENRON_START = dt.datetime(1999, 5, 10, tzinfo=dt.timezone.utc)
ENRON_T = 164
ENRON_EVENTS = [  # (P&C Fig. 8 label, plotted week (weeks from 1999-05-10), date in Peel's timeline or None)
    ("Launch of Enron online", 29.0, None), ("Launch of EBS", 36.0, "2000-01-20"),
    ("EBS-Blockbuster partnership", 62.0, None), ("Stock all-time high", 67.0, "2000-08-23"),
    ("Enron attorney discusses Belden's strategies", 73.0, "2000-10-03"), ("FERC exonerates Enron", 77.0, "2000-11-01"),
    ("EBS $53m 'profit'", 83.0, None), ("Skilling announced to be CEO", 85.0, "2000-12-13"),
    ("Blackouts in CA", 88.0, "2001-01-17"), ("Enron schedules conference call to boost stock", 97.0, "2001-03-23"),
    ("The 'asshole' call", 101.0, "2001-04-17"), ("Schwarzenegger, Lay, Milken meeting", 105.0, "2001-05-17"),
    ("Quarterly conference call", 109.0, "2001-07-12"),
    ("Skilling meets with analysts and investors in NY", 114.0, "2001-07-24"),
    ("Watkins raises accounting irregularities", 119.29, "2001-08-22"), ("Dynegy agrees to buy Enron", 130.51, "2001-11-09"),
    ("Enron restates its third quarter earnings", 132.01, "2001-11-19"),
    ("Enron shares plunge below $1", 133.43, "2001-11-28"), ("Dynegy deal collapses", 133.58, "2001-11-29"),
    ("Enron files for bankruptcy", 134.0, "2001-12-02"), ("Stephen Cooper takes over as Enron CEO", 142.01, "2002-01-30"),
    ("Lay cancels Senate committee appearance", 143.01, "2002-02-03"),
    ("Fastow, Kopper, Lay invoke the Fifth", 144.01, "2002-02-07"), ("Skilling and Watkins testify", 144.01, "2002-02-07"),
    ("Arthur Andersen LLP indicted", 148.01, "2002-03-14")]
TOL = 1  # weeks; D_{e+1} is the first score that sees snapshot e


def _pack(n, T, week, i, j):
    mats = [to_csr(n, i[week == k], j[week == k]) for k in range(T)]
    edges = np.array([A.nnz // 2 for A in mats])
    active = np.array([int((np.diff(A.indptr) > 0).sum()) for A in mats])
    return mats, {"edges_per_week": edges.tolist(), "active_per_week": active.tolist(),
                  "n_empty": int((edges == 0).sum())}


# ------------------------------------------------------------------------------------------------ MIT
def mit(files) -> dict:
    """files: [download.tsv.mit.tar.bz2]."""
    p = Path(files[0])
    with tarfile.open(p, "r:bz2") as tf:
        txt = tf.extractfile("mit/out.mit").read().decode()
    a = np.array([ln.split() for ln in txt.splitlines() if ln and not ln.startswith("%")], dtype=np.int64)
    ids = np.unique(a[:, :2])
    pos = np.searchsorted(ids, a[:, :2])
    tz = ZoneInfo("America/New_York")
    day = np.array([dt.datetime.fromtimestamp(int(s), tz).date().toordinal() for s in a[:, 3]])
    week = (day - MIT_FIRST.toordinal()) // 7
    T = int(week.max()) + 1
    mats, stats = _pack(len(ids), T, week, pos[:, 0], pos[:, 1])
    ev = []
    for d, desc in MIT_EVENTS:
        k = (dt.date.fromisoformat(d).toordinal() - MIT_FIRST.toordinal()) // 7
        ev.append({"date": d, "desc": desc, "index": int(k), "in_series": bool(0 <= k < T)})
    idx = sorted({e["index"] for e in ev if e["in_series"]})
    meta = {"id": "MIT_PC", "n": len(ids), "T": T,
            "construction": "Peel & Clauset (2015) Sec. VI.A: weekly Bluetooth proximity graphs of the Reality Mining "
                            "subjects; this copy: KONECT 'mit' (96 subjects; P&C: 97), weeks Sunday-Saturday "
                            "US/Eastern from 2004-09-12 (P&C: 35 weeks; the copy starts 2004-09-14)",
            "week_start_dates": [(MIT_FIRST + dt.timedelta(7 * k)).isoformat() for k in range(T)],
            "events": ev, "n_events_stated_by_pc": 16, "n_events_in_series": len(idx),
            "label_source": "Peel & Clauset (2015) Fig. 8 (top), from Eagle (2005, MIT PhD thesis); dates read from "
                            "the vector figure at week resolution",
            "tol": TOL, **stats}
    return {"id": "MIT_PC", "mats": mats, "seg": np.zeros(T, np.int64), "events": idx, "tol": TOL, "meta": meta,
            "extra": {"vertex_ids": ids},
            "file_meta": {"source_url": "http://konect.cc/files/download.tsv.mit.tar.bz2",
                          "raw_sha256": {p.name: sha256(p)}, "built_by": "data/downloaders/peel_clauset.py"}}


# ------------------------------------------------------------------------------------------------ Enron
def _header(f) -> tuple[str | None, str | None, list[str]]:
    """Date, From and the To addresses of one message (header up to Subject, as Peel's parser reads it)."""
    date = frm = None
    to, rec = [], False
    for raw in f:
        line = raw.decode("latin-1").rstrip("\r\n")
        if line.startswith("Date:"):
            date = line[5:].strip()
        if line.startswith("From"):
            frm = line.split(":", 1)[1].strip() if ":" in line else None
        if line.startswith("To"):
            rec = True
            line = line[3:] if line.startswith("To:") else line[2:]
        if line.startswith("Subject") or not line:
            break
        if rec:
            to += [r.strip().strip(",") for r in line.split()]
    return date, frm, to


def enron(files) -> dict:
    """files: [enron_mail_20150507.tar.gz, addresses.txt, enron_parser.py, enrontimeline.txt] (any order)."""
    by = {Path(f).name: Path(f) for f in files}
    tarball = by["enron_mail_20150507.tar.gz"]
    addresses = sorted(a.strip() for a in by["addresses.txt"].read_text().splitlines() if a.strip())
    pos = {a: k for k, a in enumerate(addresses)}
    I, J, W = [], [], []
    n_files = n_before = n_after = n_bad = 0
    with tarfile.open(tarball, "r:gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            n_files += 1
            date, frm, to = _header(tf.extractfile(m))
            if frm not in pos:
                continue
            rs = [pos[r] for r in to if r in pos]
            if not rs:
                continue
            try:
                t = email.utils.parsedate_to_datetime(date).astimezone(dt.timezone.utc)
            except (TypeError, ValueError):
                n_bad += 1
                continue
            k = (t - ENRON_START).days // 7
            if k < 0:
                n_before += 1
                continue
            if k >= ENRON_T:
                n_after += 1
                continue
            I += [pos[frm]] * len(rs)
            J += rs
            W += [k] * len(rs)
    I, J, W = np.array(I), np.array(J), np.array(W)
    mats, stats = _pack(len(addresses), ENRON_T, W, I, J)
    ev = []
    for desc, wk, date in ENRON_EVENTS:
        e = {"desc": desc, "week_plotted": wk, "index": int(np.floor(wk)), "timeline_date": date}
        if date:
            d = dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc)
            e["index_timeline"] = (d - ENRON_START).days // 7
        else:
            e["index_timeline"] = e["index"]   # month only in the timeline: the plotted week
        ev.append(e)
    idx = sorted({e["index"] for e in ev})
    idx2 = sorted({e["index_timeline"] for e in ev})
    meta = {"id": "ENRON_PC", "n": len(addresses), "T": ENRON_T,
            "construction": "Peel & Clauset (2015) Sec. VI.B: emails among their 151 users, weekly from Mon 1999-05-10 "
                            "(their parser's start date) through June 2002",
            "week_start_dates": [(ENRON_START + dt.timedelta(7 * k)).date().isoformat() for k in range(ENRON_T)],
            "events": ev, "n_events_stated_by_pc": 25, "n_event_weeks": len(idx),
            "events_timeline_weeks": idx2,
            "label_source": "Peel & Clauset (2015) Fig. 8 (bottom), events from Peel's timeline file (source named "
                            "there: agsm.edu.au/bobm/teaching/BE/Enron/timeline.html)",
            "messages_scanned": n_files, "links_kept": len(I), "messages_before_week0": n_before,
            "messages_after_june_2002": n_after, "unparsed_dates": n_bad, "tol": TOL, **stats}
    return {"id": "ENRON_PC", "mats": mats, "seg": np.zeros(ENRON_T, np.int64), "events": idx, "tol": TOL,
            "meta": meta, "extra": {"events_timeline": np.array(idx2, np.int64)},
            "file_meta": {"source_url": "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz; addresses and "
                                        "timeline: github.com/piratepeel/piratepeel.github.io old/code/",
                          "raw_sha256": {f: sha256(p) for f, p in sorted(by.items())},
                          "built_by": "data/downloaders/peel_clauset.py"}}
