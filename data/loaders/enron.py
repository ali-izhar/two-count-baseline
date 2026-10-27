"""Enron email network loader (the series `data/downloaders/enron.py` builds).

Source: CMU Enron Corpus https://www.cs.cmu.edu/~enron/
Events: the six dates of ENRON_EVENTS in the downloader; repro/ uses the four weekly events of
config.ENRON_CPS_MERGED.
"""

import pickle
from pathlib import Path

from config import RAW_DIR
from data.loaders.base import GraphSequence

DATA_DIR = RAW_DIR / "enron"


def load_enron(
    data_dir: Path | None = None,
    aggregation: str = "weekly",
    download: bool = True,
) -> GraphSequence:
    """Load Enron email network.

    Args:
        data_dir: Directory containing enron data
        aggregation: Time aggregation ("daily", "weekly", "monthly")
        download: If True, download data if not present

    Returns:
        GraphSequence with email communication graphs
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR

    # Check for preprocessed data
    pkl_path = data_dir / f"enron_{aggregation}.pkl"
    if pkl_path.exists():
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        return GraphSequence(
            graphs=data['graphs'],
            timestamps=data.get('timestamps', []),
            true_cps=data.get('true_cps', []),
            events=data.get('events', {}),
            metadata=data.get('metadata', {'source': 'CMU', 'name': 'Enron'}),
        )

    if download:
        from data.downloaders.enron import download_enron
        download_enron(data_dir, aggregation=aggregation)
        return load_enron(data_dir, aggregation=aggregation, download=False)

    raise FileNotFoundError(
        f"Enron data not found in {data_dir}. "
        "Run: python -m data.downloaders.enron"
    )
