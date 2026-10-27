"""MIT Social Evolution proximity loader (the series `data/downloaders/mit_reality.py` builds).

Source: http://realitycommons.media.mit.edu/socialevolution.html (manual download of Proximity.csv).
Events: six dates from the MIT academic calendar 2008-09 and the federal holidays (MIT_EVENTS of the downloader).
"""

import pickle
from pathlib import Path

from config import RAW_DIR
from data.loaders.base import GraphSequence

DATA_DIR = RAW_DIR / "mit_reality"


def load_mit_reality(
    data_dir: Path | None = None,
    process: bool = True,
    download: bool = False,  # Not supported - manual download required
) -> GraphSequence:
    """Load the MIT Social Evolution proximity network.

    Args:
        data_dir: Directory containing the MIT data
        process: If True, process CSV if .pkl not found
        download: Ignored (the data need a manual download)

    Returns:
        GraphSequence with daily proximity graphs

    Note:
        Unlike the other loaders, this one needs a manual download of Proximity.csv from
        http://realitycommons.media.mit.edu/socialevolution.html
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR

    pkl_path = data_dir / "mit_reality.pkl"
    if pkl_path.exists():
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        return GraphSequence(
            graphs=data['graphs'],
            timestamps=data['timestamps'],
            true_cps=data['true_cps'],
            events=data['events'],
            metadata=data.get('metadata', {'source': 'MIT', 'name': 'Reality Mining'}),
        )

    if process:
        from data.downloaders.mit_reality import process_mit_reality
        process_mit_reality(data_dir)
        return load_mit_reality(data_dir, process=False)

    raise FileNotFoundError(
        f"MIT Reality data not found in {data_dir}.\n"
        "Download Proximity.csv from http://realitycommons.media.mit.edu/realitymining.html\n"
        "Then run: python -m data.downloaders.mit_reality"
    )
