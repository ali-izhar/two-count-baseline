# A two-count baseline for change detection on dynamic networks

Change detectors for dynamic networks are judged by how well they score dated events in a few real networks. This
project measures what these events change and compares published detectors with a score of two counts, the number
of active vertices and the number of edges.

We run 13 published detectors, 3 structural statistics, the two counts and a constant score on 10 labelled datasets,
rank each event among the event-free windows of its series, and test what the detectors see when the structure changes
at fixed degrees. Paper: [`paper/main.pdf`](paper/main.pdf). Every number in it comes from [`repro/`](repro/README.md).

## Methods

| Method | Code | Source |
|---|---|---|
| Two counts (baseline) | `methods/counts.py` | Kendrick, Musial and Gabrys, Computer Science Review 2018 |
| Vertex turnover | `methods/turnover.py` | this work |
| LAD | `methods/lad.py` | Huang, Hitti, Rabusseau and Rabbany, KDD 2020 |
| SCPD | `methods/scpd.py` | Huang, Danovitch, Rabusseau and Rabbany, PAKDD 2023 |
| TIRE | `methods/tire.py` | De Ryck, De Vos and Bertrand, IEEE Trans. Signal Process. 2021 |
| CUSUM | `methods/cusum.py` | Page, Biometrika 1954 |
| BOCPD | `methods/bocpd.py` | Adams and MacKay, arXiv 2007 |
| MMD | `methods/mmd.py` | Gretton et al., JMLR 2012 |
| s-GNN (failed reproduction, not pooled) | `methods/sgnn.py` | Sulem, Kenlay, Cucuringu and Dong, Machine Learning 2024 |
| Hamming, Jaccard | `methods/hamming.py` | edge-set distances |
| DeltaCon | `methods/deltacon.py` | Koutra, Vogelstein and Faloutsos, SDM 2013 |
| Chen–Zhang | `methods/chenzhang.py` | Chen and Zhang, Annals of Statistics 2015 |
| NBS | `methods/nbs.py` | Wang, Yu and Rinaldo, Annals of Statistics 2021 |
| ASE | `methods/ase.py` | Sussman et al., JASA 2012; Madrid Padilla, Yu and Priebe, JMLR 2022 |
| SpotLight-style sketch | `methods/spotlight.py` | Eswaran, Faloutsos, Guha and Mishra, KDD 2018 |
| Transitivity, degree assortativity, their maximum | `repro/plants.py`, `repro/19_activity.py` | assortativity: Newman, Physical Review Letters 2002 |
| GHRG (Peel and Clauset's series only) | `methods/ghrg.py` | Peel and Clauset, AAAI 2015 |
| NetSimile (appendix) | `methods/netsimile.py` | Berlingerio, Koutra, Eliassi-Rad and Faloutsos, arXiv 2012 |
| Conditional score test (appendix) | `methods/scoretest.py` | Rasch 1960; Mantel and Haenszel, JNCI 1959 |

The node-aligned detectors (Hamming to SpotLight-style) are two-window versions of the published statistics. The datasets
and their sources are listed in [`data/README.md`](data/README.md).

## Citation

```bibtex
@misc{ali2026twocount,
  title  = {A Two-Count Baseline for Change Detection on Dynamic Networks},
  author = {Ali, Izhar},
  year   = {2026},
  url    = {https://github.com/ali-izhar/two-count-baseline}
}
```

MIT license.
