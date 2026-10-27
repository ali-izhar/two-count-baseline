"""Step 10: the stored graphs and events of benchmark B -> repro/data/bench/<ID>.npz and <ID>_meta.json.

Every snapshot is a binary symmetric graph without self-loops on a fixed vertex set. Layout of <ID>.npz: n, T,
indptr (T, n + 1) (SCC: `rows`, the row of each entry, since (T, n + 1) would need 11.5 GB), indices, offsets
(T + 1; snapshot t owns entries offsets[t]:offsets[t + 1]), segment, events, tol, meta_json, plus per-source extras
(e.g. `weights`, the source paper's own edge weight aligned with `indices`).
Raw files: data/raw/<source>/ (data/README.md). MIT and ENRON are the repo's Table 1 graphs (step 1); MIT_PC and
ENRON_PC are the Peel & Clauset constructions (data/downloaders/peel_clauset.py).
A dataset is rebuilt only when the sha256 of its raw files or the reader code changes.

Usage: 10_bench_data.py [ID ...]        (default: all, small sets first, SCC last)
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent)]           # works under python -I as well

import numpy as np
from common import Timer, peak_rss_mb

from data.downloaders.snapshots import check, from_mats, sha256

REPO = ROOT.parent
RAW = REPO / "data" / "raw"
OUT = ROOT / "data" / "bench"
# reader module, raw directory (or files), datasets it returns
GROUPS = [
    ("lad", RAW / "lad", ("UCI", "USLegis", "canVote")),
    ("sociopatterns", RAW / "sociopatterns", ("PS", "HT09")),
    ("repo_view", [REPO / "repro/data/graphs_mit_reality.npz", REPO / "repro/data/graphs_enron.npz",
                   REPO / "data/cache/mit_reality.pkl", REPO / "data/cache/enron.pkl"], ("MIT", "ENRON")),
    ("tgb.mag", RAW / "mag", ("MAG",)),
    ("tgb.fl", RAW / "fl", ("FL",)),
    ("pamap2", RAW / "pamap2", ("PM",)),
    ("chbmit", RAW / "chbmit_b", ("EEG",)),
    ("tgb.scc", RAW / "scc", ("SCC",)),
    ("peel_clauset.mit", [RAW / "konect_mit/download.tsv.mit.tar.bz2"], ("MIT_PC",)),
    ("peel_clauset.enron", [RAW / "enron_cmu/enron_mail_20150507.tar.gz"]
     + [RAW / "peel_code" / f for f in ("addresses.txt", "enron_parser.py", "enrontimeline.txt")], ("ENRON_PC",)),
]


def raw_files(src) -> list[Path]:
    return sorted(src) if isinstance(src, list) else sorted(p for p in Path(src).rglob("*") if p.is_file())


def file_hashes(files) -> dict:
    """sha256 per file, memoised on (size, mtime) so that the 5 GB of SCC is hashed once."""
    memo_p = OUT / "raw_sha256.json"
    memo = json.loads(memo_p.read_text()) if memo_p.exists() else {}
    out = {}
    for p in files:
        st = p.stat()
        m = memo.get(str(p))
        if not m or m[0] != st.st_size or m[1] != st.st_mtime_ns:
            memo[str(p)] = m = [st.st_size, st.st_mtime_ns, sha256(p)]
        out[str(p.relative_to(REPO))] = m[2]
    memo_p.write_text(json.dumps(memo, indent=0))
    return out


def group_key(module: str, src) -> str:
    code = [Path(importlib.import_module(f"data.downloaders.{module.split('.')[0]}").__file__),
            REPO / "data/downloaders/snapshots.py", Path(__file__)]
    h = hashlib.sha256(json.dumps(file_hashes(raw_files(src)), sort_keys=True).encode())
    for p in code:
        h.update(p.read_bytes())
    return h.hexdigest()


def save(d: dict) -> None:
    if "mats" in d:
        mats = d["mats"]
        for A in mats:
            check(A)
        n, T = mats[0].shape[0], len(mats)
        offsets, rows, cols = from_mats(mats)
    else:
        n, T = d["n"], d["T"]
        offsets, rows, cols = d["coo"]
    extra = dict(d["extra"])
    if d.get("indptr", True):
        ip = np.zeros((T, n + 1), np.int64)
        for t in range(T):
            ip[t, 1:] = np.cumsum(np.bincount(rows[offsets[t]:offsets[t + 1]], minlength=n))
    # member order of the two source layouts (CSR lists; weighted edge lists), so equal arrays give equal files
    out = {"n": np.int64(n), "T": np.int64(T)}
    if "mats" in d:
        out.update(indptr=ip, indices=cols, offsets=offsets)
    else:
        out.update(indices=cols, offsets=offsets, weights=extra.pop("weights"))
    out.update(segment=np.asarray(d["seg"], np.int64), events=np.asarray(d["events"], np.int64),
               tol=np.int64(d["tol"]), meta_json=np.array(json.dumps(d["meta"])))
    if "mats" not in d:
        out.update(indptr=ip) if d.get("indptr", True) else out.update(rows=rows)
    out.update(extra)
    p = OUT / f"{d['id']}.npz"
    np.savez_compressed(p, **out)
    meta = dict(d["meta"])
    for k, v in d.get("file_meta", {}).items():     # registration keys; npz_sha256 follows raw_sha256
        meta[k] = v
        if k == "raw_sha256":
            meta["npz_sha256"] = sha256(p)
    (OUT / f"{d['id']}_meta.json").write_text(json.dumps(meta, indent=1))


def main(ids: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    keys_p = OUT / "keys.json"
    keys = json.loads(keys_p.read_text()) if keys_p.exists() else {}
    for module, src, members in GROUPS:
        if ids and not set(ids) & set(members):
            continue
        key = group_key(module, src)
        if all(keys.get(i) == key and (OUT / f"{i}.npz").exists() for i in members):
            print(f"{','.join(members)}: cached", flush=True)
            continue
        t = Timer()
        mod, _, fn = module.partition(".")
        reader = importlib.import_module(f"data.downloaders.{mod}")
        arg = REPO if mod == "repo_view" else src
        built = [getattr(reader, fn)(arg)] if fn else reader.series(arg)
        for d in built:
            save(d)
            keys[d["id"]] = key
            keys_p.write_text(json.dumps(keys, indent=1))
            print(f"{d['id']}: T {len(d['seg'])}, {len(d['events'])} events, {t():.0f} s, peak RSS "
                  f"{peak_rss_mb() / 1024:.1f} GB, {time.strftime('%H:%M:%S')}", flush=True)
        del built


if __name__ == "__main__":
    main(sys.argv[1:])
