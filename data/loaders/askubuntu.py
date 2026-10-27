"""AskUbuntu temporal network loader.

Source: SNAP https://snap.stanford.edu/data/sx-askubuntu.html

Events: the Ubuntu release months that data/downloaders/askubuntu.py maps to snapshot indices (true_cps).
"""

import pickle
from pathlib import Path

from config import RAW_DIR
from data.loaders.base import GraphSequence

DATA_DIR = RAW_DIR / "askubuntu"


# Ubuntu release dates (potential change points)
UBUNTU_RELEASES = {
    "Ubuntu_10.04": "2010-04-29",
    "Ubuntu_10.10": "2010-10-10",
    "Ubuntu_11.04": "2011-04-28",
    "Ubuntu_11.10": "2011-10-13",
    "Ubuntu_12.04": "2012-04-26",
    "Ubuntu_12.10": "2012-10-18",
    "Ubuntu_13.04": "2013-04-25",
    "Ubuntu_13.10": "2013-10-17",
    "Ubuntu_14.04": "2014-04-17",
}


def load_askubuntu(
    data_dir: Path | None = None,
    aggregation: str = "monthly",  # "daily", "weekly", "monthly"
    download: bool = True,
) -> GraphSequence:
    """Load AskUbuntu interaction network.

    Args:
        data_dir: Directory containing AskUbuntu data
        aggregation: Time aggregation level
        download: If True, download if not present

    Returns:
        GraphSequence with interaction graphs
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR

    # Check for preprocessed data
    pkl_path = data_dir / f"askubuntu_{aggregation}.pkl"
    if pkl_path.exists():
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        return GraphSequence(
            graphs=data['graphs'],
            timestamps=data.get('timestamps', []),
            true_cps=data.get('true_cps', []),
            events=data.get('events', UBUNTU_RELEASES),
            metadata={'source': 'SNAP', 'name': 'AskUbuntu', 'aggregation': aggregation},
        )

    if download:
        from data.downloaders.askubuntu import download_askubuntu
        download_askubuntu(data_dir, aggregation)
        return load_askubuntu(data_dir, aggregation, download=False)

    raise FileNotFoundError(
        f"AskUbuntu data not found in {data_dir}. "
        "Run: python -m data.downloaders.askubuntu"
    )
