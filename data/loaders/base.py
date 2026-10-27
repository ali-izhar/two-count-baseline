"""Base classes for dataset handling."""

from dataclasses import dataclass, field
from typing import Any

import networkx as nx


@dataclass
class GraphSequence:
    """Container for a sequence of graphs with metadata.

    Attributes:
        graphs: List of NetworkX graphs
        timestamps: List of timestamps/dates (strings or datetime)
        true_cps: List of known change point indices
        events: Dict mapping event names to indices
        metadata: Additional dataset information
    """
    graphs: list[nx.Graph]
    timestamps: list[str] = field(default_factory=list)
    true_cps: list[int] = field(default_factory=list)
    events: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
