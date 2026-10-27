"""Download and process temporal Enron email network.

Source: CMU Enron Corpus https://www.cs.cmu.edu/~enron/
Timeline: 1999-2002 (main activity period)

Ground-truth events:
- 2001-02-12: Skilling becomes CEO
- 2001-08-14: Skilling resigns (major shock)
- 2001-10-16: $618M loss reported
- 2001-10-22: SEC probe announced
- 2001-11-28: Dynegy merger fails
- 2001-12-02: Bankruptcy filed
"""

import email
import email.utils
import pickle
import tarfile
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import networkx as nx

from config import RAW_DIR

# Ground truth events for CPD evaluation
ENRON_EVENTS = {
    "Skilling_CEO": "2001-02-12",
    "Skilling_Resigns": "2001-08-14",
    "Loss_Reported": "2001-10-16",
    "SEC_Probe": "2001-10-22",
    "Dynegy_Fails": "2001-11-28",
    "Bankruptcy": "2001-12-02",
}

CMU_CORPUS_URL = "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"

# Valid date range for filtering (Enron activity period)
MIN_DATE = datetime(1998, 1, 1)
MAX_DATE = datetime(2003, 1, 1)


def download_enron(
    data_dir: Path | None = None,
    aggregation: str = "weekly",
    use_cached_edges: bool = True,
):
    """Download temporal Enron email network.

    Args:
        data_dir: Directory to save data
        aggregation: Time aggregation ("daily", "weekly", "monthly")
        use_cached_edges: Use cached edge list if available

    Returns:
        Tuple of (graphs, timestamps, true_cps)
    """
    if data_dir is None:
        data_dir = RAW_DIR / "enron"
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    edges_cache = data_dir / "enron_edges.pkl"

    # Load or parse edges
    if use_cached_edges and edges_cache.exists():
        print(f"Loading cached edges from {edges_cache}...")
        with open(edges_cache, 'rb') as f:
            cached = pickle.load(f)
            temporal_edges = cached['edges']
    else:
        temporal_edges = _parse_corpus(data_dir)
        with open(edges_cache, 'wb') as f:
            pickle.dump({'edges': temporal_edges}, f)
        print(f"Cached {len(temporal_edges)} edges to {edges_cache}")

    # Aggregate by time period
    edges_by_time = defaultdict(list)
    for src, dst, dt in temporal_edges:
        if aggregation == "daily":
            key = dt.strftime("%Y-%m-%d")
        elif aggregation == "weekly":
            key = dt.strftime("%G-W%V")
        else:
            key = dt.strftime("%Y-%m")
        edges_by_time[key].append((src, dst))

    print(f"Aggregated into {len(edges_by_time)} {aggregation} periods")

    # Build graphs
    timestamps = sorted(edges_by_time.keys())
    all_nodes = {n for edges in edges_by_time.values() for e in edges for n in e}

    graphs = []
    for ts in timestamps:
        G = nx.Graph()
        G.add_nodes_from(all_nodes)
        G.add_edges_from(edges_by_time[ts])
        graphs.append(G)

    # Map events to indices
    true_cps, events = _map_events(timestamps, aggregation)

    # Save
    save_path = data_dir / f"enron_{aggregation}.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump({
            'graphs': graphs,
            'timestamps': timestamps,
            'true_cps': true_cps,
            'events': events,
            'metadata': {
                'source': 'CMU Enron Corpus',
                'aggregation': aggregation,
                'n_nodes': len(all_nodes),
                'total_edges': len(temporal_edges),
                'ground_truth_events': ENRON_EVENTS,
            }
        }, f)

    print(f"Saved {len(graphs)} graphs to {save_path}")
    print(f"Ground truth: {len(true_cps)} change points")
    return graphs, timestamps, true_cps


def _parse_corpus(data_dir: Path) -> list:
    """Download and parse CMU Enron Corpus to extract temporal edges."""
    tar_path = data_dir / "enron_mail.tar.gz"
    extract_dir = data_dir / "maildir"

    # Download if needed
    if not extract_dir.exists():
        if not tar_path.exists():
            print("Downloading CMU Enron Corpus (~0.44 GB)...")
            urllib.request.urlretrieve(CMU_CORPUS_URL, tar_path)
        print("Extracting (this takes a few minutes)...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(data_dir, filter="data")  # untrusted archive: no links or paths outside data_dir
        tar_path.unlink()

    # Parse emails
    print("Parsing email headers...")
    temporal_edges = []
    email_to_id = {}
    mail_count = 0

    for mail_file in extract_dir.rglob("*"):
        if not mail_file.is_file():
            continue
        try:
            with open(mail_file, 'rb') as f:
                msg = email.message_from_binary_file(f)

            # Parse date
            date_str = msg.get('Date')
            if not date_str:
                continue
            try:
                dt = email.utils.parsedate_to_datetime(date_str)
            except (TypeError, ValueError):
                continue

            # Filter to valid date range
            if not (MIN_DATE <= dt.replace(tzinfo=None) <= MAX_DATE):
                continue

            # Parse sender
            from_header = msg.get('From', '')
            _, from_addr = email.utils.parseaddr(str(from_header))
            if not from_addr:
                continue
            from_addr = from_addr.lower()

            # Parse recipients
            recipients = set()
            for header in ['To', 'Cc', 'Bcc']:
                addr_list = msg.get(header, '')
                if addr_list:
                    for addr in str(addr_list).split(','):
                        _, parsed = email.utils.parseaddr(addr.strip())
                        if parsed and parsed.lower() != from_addr:
                            recipients.add(parsed.lower())

            # Get/create IDs and add edges
            if from_addr not in email_to_id:
                email_to_id[from_addr] = len(email_to_id)
            src_id = email_to_id[from_addr]

            for recip in recipients:
                if recip not in email_to_id:
                    email_to_id[recip] = len(email_to_id)
                temporal_edges.append((src_id, email_to_id[recip], dt))

            mail_count += 1
            if mail_count % 50000 == 0:
                print(f"  {mail_count} emails, {len(temporal_edges)} edges...")

        except Exception:  # noqa: BLE001, S112  (a malformed message is skipped)
            continue

    print(f"Parsed {mail_count} emails -> {len(temporal_edges)} edges")
    return temporal_edges


def _map_events(timestamps: list, aggregation: str) -> tuple:
    """Map ground truth events to timestamp indices."""
    true_cps = []
    events = {}

    for name, date_str in ENRON_EVENTS.items():
        event_dt = datetime.strptime(date_str, "%Y-%m-%d")

        if aggregation == "monthly":
            key = date_str[:7]
        elif aggregation == "weekly":
            key = event_dt.strftime("%G-W%V")
        else:
            key = date_str

        # Find exact or closest match
        if key in timestamps:
            idx = timestamps.index(key)
        else:
            # Find first timestamp >= event
            idx = next((i for i, ts in enumerate(timestamps) if ts >= key), None)

        if idx is not None:
            true_cps.append(idx)
            events[name] = idx
            print(f"  {name}: {date_str} -> index {idx} ({timestamps[idx]})")

    return sorted(set(true_cps)), events


if __name__ == "__main__":
    download_enron()
