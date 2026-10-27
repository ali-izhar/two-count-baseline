"""Download and process AskUbuntu temporal network from SNAP."""

import gzip
import pickle
import shutil
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import networkx as nx

from config import RAW_DIR

DATA_URL = "https://snap.stanford.edu/data/sx-askubuntu.txt.gz"


def download_askubuntu(data_dir: Path | None = None, aggregation: str = "monthly"):
    """Download AskUbuntu interaction network from SNAP.

    Args:
        data_dir: Directory to save data
        aggregation: Time aggregation ("daily", "weekly", "monthly")
    """
    if data_dir is None:
        data_dir = RAW_DIR / "askubuntu"

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    gz_path = data_dir / "sx-askubuntu.txt.gz"
    txt_path = data_dir / "sx-askubuntu.txt"

    # Download if needed
    if not txt_path.exists():
        print(f"Downloading AskUbuntu data from {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, gz_path)

        print("Extracting...")
        with gzip.open(gz_path, 'rb') as f_in, open(txt_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        gz_path.unlink()
        print(f"Saved to {txt_path}")

    # Parse temporal edges
    print("Parsing temporal edges...")
    edges_by_time = defaultdict(list)

    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('%') or not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    src, dst, timestamp = int(parts[0]), int(parts[1]), int(parts[2])
                    dt = datetime.fromtimestamp(timestamp)

                    if aggregation == "daily":
                        key = dt.strftime("%Y-%m-%d")
                    elif aggregation == "weekly":
                        # Use ISO year and ISO week for correct year boundary handling
                        key = dt.strftime("%G-W%V")
                    else:  # monthly
                        key = dt.strftime("%Y-%m")

                    edges_by_time[key].append((src, dst))
                except (ValueError, OSError):
                    continue

    print(f"Loaded {sum(len(v) for v in edges_by_time.values())} edges across {len(edges_by_time)} time periods")

    # Create graph sequence
    timestamps = sorted(edges_by_time.keys())
    graphs = []

    for ts in timestamps:
        G = nx.Graph()
        G.add_edges_from(edges_by_time[ts])
        graphs.append(G)

    # Map Ubuntu releases to indices
    # Dataset covers 2009-01 to 2016-03 (76 months), but has gaps in early months
    # Release dates are typically in April (04) and October (10)
    # Note: 10.04 has no data (dataset jumps from 2010-01 to 2010-05)
    # Note: 16.04 is after dataset end (2016-03)
    ubuntu_releases = {
        "Ubuntu_10.04": "2010-04",  # Lucid Lynx (LTS) - may not match due to data gap
        "Ubuntu_10.10": "2010-10",  # Maverick Meerkat
        "Ubuntu_11.04": "2011-04",  # Natty Narwhal
        "Ubuntu_11.10": "2011-10",  # Oneiric Ocelot
        "Ubuntu_12.04": "2012-04",  # Precise Pangolin (LTS)
        "Ubuntu_12.10": "2012-10",  # Quantal Quetzal
        "Ubuntu_13.04": "2013-04",  # Raring Ringtail
        "Ubuntu_13.10": "2013-10",  # Saucy Salamander
        "Ubuntu_14.04": "2014-04",  # Trusty Tahr (LTS)
        "Ubuntu_14.10": "2014-10",  # Utopic Unicorn
        "Ubuntu_15.04": "2015-04",  # Vivid Vervet
        "Ubuntu_15.10": "2015-10",  # Wily Werewolf
    }

    true_cps = []
    events = {}
    for event, release_date in ubuntu_releases.items():
        # Find matching timestamp for this release
        # Note: idx represents the period DURING which the release happened.
        # For CPD, the behavioral shift may be visible in this month or the next.
        # Evaluation should use a tolerance window (e.g., ±1 month).
        matching = [i for i, ts in enumerate(timestamps) if ts.startswith(release_date[:7])]
        if matching:
            idx = matching[0]
            true_cps.append(idx)
            events[event] = idx

    # Save with metadata
    save_path = data_dir / f"askubuntu_{aggregation}.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump({
            'graphs': graphs,
            'timestamps': timestamps,
            'true_cps': sorted(set(true_cps)),
            'events': events,
            'metadata': {
                'source': 'SNAP (Stanford Network Analysis Project)',
                'url': DATA_URL,
                'aggregation': aggregation,
                'n_graphs': len(graphs),
                'n_events': len(events),
                'date_range': f'{timestamps[0]} to {timestamps[-1]}',
                'description': 'AskUbuntu Q&A interactions. Change points are Ubuntu release dates.',
            },
        }, f)

    print(f"Saved {len(graphs)} graphs to {save_path}")
    print(f"True change points (Ubuntu releases): {len(true_cps)}")


if __name__ == "__main__":
    download_askubuntu()
