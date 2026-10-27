"""Process the MIT Social Evolution proximity data (Reality Commons; Madan et al., IEEE Pervasive Computing 11(4), 2012).

Source: http://realitycommons.media.mit.edu/socialevolution.html
(Manual download required - place Proximity.csv in data/raw/mit_reality/)

Kept days: 2008-01-30 to 2009-06-24 (270 days; a day is kept iff >5 rows with prob2 >= 0.3 and >1 edge).

Events (MIT_EVENTS): the first day that the MIT academic calendar 2008-2009 marks as a holiday or vacation, or, for
the end of term, the first day after the last day of classes; each date carries its source. An event maps to the
first kept day on or after its date.
"""

import pickle
from pathlib import Path

import networkx as nx
import pandas as pd

from config import RAW_DIR

# MIT Registrar, Academic Calendar 2008-2009, as published in Oct 2008 (Wayback capture)
CAL = "http://web.archive.org/web/20081018102435/http://web.mit.edu:80/registrar/www/calendar.html"
USC = "https://www.law.cornell.edu/uscode/text/5/6103"   # 5 U.S.C. 6103, legal public holidays
MIT_EVENTS = {   # name: (date, source)
    "ColumbusDay": ("2008-10-13", f"{CAL}: 'OCTOBER 13 Monday Columbus Day Holiday'"),
    "Thanksgiving": ("2008-11-27", f"{CAL}: 'NOVEMBER 27,28 Thurs-Fri Thanksgiving Vacation'"),
    "Christmas": ("2008-12-25", f"{USC}: 'Christmas Day, December 25'"),
    "NewYear": ("2009-01-01", f"{USC}: 'New Year's Day, January 1'"),
    "SpringBreak": ("2009-03-23", f"{CAL}: 'MARCH 23-27 Mon-Fri Spring Vacation'"),
    "EndOfClasses": ("2009-05-15", f"{CAL}: 'MAY 14 Thursday Last day of classes' (event: the next day)"),
}


def event_indices(timestamps: list[str]) -> dict[str, int]:
    """Index of the first kept day on or after each event date."""
    return {name: next(i for i, d in enumerate(timestamps) if d >= date) for name, (date, _) in MIT_EVENTS.items()}


def process_mit_reality(
    data_dir: Path | None = None,
    probability_threshold: float = 0.3,
):
    """Process MIT Reality Mining CSV into graph sequence.

    Args:
        data_dir: Directory containing Proximity.csv
        probability_threshold: Minimum probability for edge inclusion

    Returns:
        Tuple of (graphs, timestamps, true_cps)
    """
    if data_dir is None:
        data_dir = RAW_DIR / "mit_reality"
    data_dir = Path(data_dir)

    csv_path = data_dir / "Proximity.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"MIT Reality data requires manual download from:\n"
            f"  http://realitycommons.media.mit.edu/realitymining.html\n"
            f"Place Proximity.csv in: {data_dir}"
        )

    print(f"Processing {csv_path}...")
    df = pd.read_csv(csv_path, header=0)

    # Clean column names
    df.columns = df.columns.str.strip().str.strip('"').str.replace("\ufeff", "", regex=False)
    df = df.rename(columns={
        "user.id": "user_id",
        "remote.user.id.if.known": "remote_user_id",
        "time": "timestamp",
        "prob2": "probability"
    })

    # Parse columns
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(str).str.strip().str.strip('"'), errors="coerce")
    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")
    df["remote_user_id"] = pd.to_numeric(df["remote_user_id"], errors="coerce")
    df["probability"] = pd.to_numeric(df["probability"], errors="coerce")
    df["date"] = df["timestamp"].dt.date
    df = df.dropna(subset=["timestamp", "probability"])

    print(f"Loaded {len(df)} proximity records")

    # Build daily graphs
    all_users = set(df["user_id"].unique()) | set(df["remote_user_id"].unique())
    all_users = {u for u in all_users if pd.notna(u)}
    daily_graphs = {}

    for date in sorted(df["date"].unique()):
        day_df = df[(df["date"] == date) & (df["probability"] >= probability_threshold)]
        if len(day_df) > 5:
            G = nx.Graph()
            G.add_nodes_from(all_users)
            edges = day_df[["user_id", "remote_user_id"]].dropna().values
            G.add_edges_from(edges)
            if G.number_of_edges() > 1:
                daily_graphs[date.isoformat()] = G

    timestamps = sorted(daily_graphs.keys())
    graphs = [daily_graphs[d] for d in timestamps]

    print(f"Created {len(graphs)} daily graphs")
    print(f"Date range: {timestamps[0]} to {timestamps[-1]}")
    print(f"Nodes: {len(all_users)}")

    # Map events to indices
    events = event_indices(timestamps)
    true_cps = list(events.values())
    for name, idx in events.items():
        print(f"  {name}: {MIT_EVENTS[name][0]} -> index {idx} ({timestamps[idx]})")

    true_cps = sorted(set(true_cps))

    # Save
    save_path = data_dir / "mit_reality.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump({
            'graphs': graphs,
            'timestamps': timestamps,
            'true_cps': true_cps,
            'events': events,
            'metadata': {
                'source': 'MIT Reality Mining',
                'n_nodes': len(all_users),
                'total_days': len(graphs),
                'probability_threshold': probability_threshold,
                'ground_truth_events': {k: {'date': d, 'source': c} for k, (d, c) in MIT_EVENTS.items()},
            }
        }, f)

    print(f"Saved to {save_path}")
    print(f"Ground truth: {len(true_cps)} change points")
    return graphs, timestamps, true_cps


if __name__ == "__main__":
    process_mit_reality()
