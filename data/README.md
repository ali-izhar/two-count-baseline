# Data

The raw data are not in the repository. Download each file into `data/raw/<dir>/` and check it against the sha256
below. `downloaders/` read the raw files into snapshots; `repro/10_bench_data.py` builds the datasets of the paper.

## Selection series

The fixed settings of the detectors are chosen on these three series.

| dataset | snapshots | vertices | events | source |
|---|---|---|---|---|
| MIT Social Evolution (2008-09) | 270 daily proximity graphs | 63 people | 6 (MIT academic calendar 2008-09 and federal holidays; one source per date in `downloaders/mit_reality.py`) | [Reality Commons, Social Evolution](http://realitycommons.media.mit.edu/socialevolution.html), `Proximity.csv` (manual download; sha256 519bd39e64ce76d2975e4054af83b9ec2772604d70763cc415738ac330657191) |
| AskUbuntu | 76 monthly answer graphs | each month's active users only, up to 8,102 | 11 (Ubuntu releases) | [SNAP sx-askubuntu](https://snap.stanford.edu/data/sx-askubuntu.html), `sx-askubuntu.txt.gz` (no sha256 is recorded; the downloader does not check the file) |
| Enron | 200 weekly email graphs | 86,664 addresses | 4 (corporate events; two one week apart merged) | [CMU Enron corpus](https://www.cs.cmu.edu/~enron/), `enron_mail_20150507.tar.gz` (the tarball of ENRON_PC below; the downloader deletes it after extraction) |

AskUbuntu's months hold only that month's active users and follow the local time zone of the machine that builds them.

## The 13 datasets

MIT and Enron are built from the selection series; the other datasets come from these files.

| IDs | dir | source | files (sha256) |
|---|---|---|---|
| UCI, USLegis, canVote | `lad/` | LAD authors' repository, `git clone https://github.com/shenyangHuang/LAD` at commit `58330238a1e907d450a3354e7f2f786b53d54262` (the whole repository) | `datasets/UCI_processed/OCnodeslinks_chars.txt` 72ca06f1dea84472cd025f1248f179a015b92f145097fb2cae3d687a844d3c39, `datasets/USLegis_processed/LegisEdgelist.txt` 849d044d42c09c1f289c3b9776fb499e882eeab21e8b1d57c9e595c7f9806e37, `datasets/canVote_processed/canVote_edgelist.txt` be1448cc280d17fd248e2b1ad03dff7e37e40aeb20fe0adb80a3e9ec5b84cc08 |
| MAG | `mag/` | `https://object-arbutus.alliancecan.ca/swift/v1/14c95234f6cd4a21a47deafe20cce2a7/tgb/history_scpd.zip` (SCPD; unzip in place) | `history_scpd.zip` d21cac5079e3c4919fc02ed50d97a3b0f73739dc50cb807477b73227fcc36794, `coauth-MAG-History-full-edgelist.txt` 9676cbfdd62711ffe6e4c893646f58c7fae3415b51c8dee57ae5ddf580f32be2 |
| FL | `fl/` | `.../tgb/flight_scpd.zip` (same host; unzip in place) | `flight_scpd.zip` fd1972f7746a07ebd9911a14dc29d708bdcef7f7cbe7f1a697f7ef2e0f3bdc30, `edges_all.txt` 12afc79362a1568114586a6ef2f443879b88db143e301a3898c2c940b462c095, `node_order.txt` d62410911186e8ed7982b93454d6ba57d402947be7d98e8d1a16d761c7d6df4e, `node_country.txt` 515d3bb12f7ea11d97796ad87b4b2ae02d9f948711c1965c4835e79f8aea498e, `node_continent.txt` 5c05372270d2216eaa500e4adeab28c6e446baa733073ad6c5143eb3623ede3e |
| SCC | `scc/` | `.../tgb/stablecoin_scpd.zip` (same host, 1.2 GB; unzip in place, 3.9 GB) | `stablecoin_scpd.zip` 47c52fc49f4c7cd4ad8f272a373c91072c7215e77dcaa10d4e65962bbb6798d5, `coin_edgelistv3.csv` 750ae15613e1f440144a7acc548d07e0aecd30dc5a33cfe4b716157bc7014276 |
| PM | `pamap2/` | `https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip`, `PAMAP2_Dataset/Protocol/subject101.dat` .. `subject108.dat` | 101 80b3fffd089c236ae88098df3b906b8d3c67a858023fe4f75045d78612147266, 102 468f3c6ba3e6e961f383783665265afbcddbbd33a3a84bba0386b7d0d4e3c632, 103 5be64631d085e3efc700728395516da3512ff9de17dab30d253508441a98d3f0, 104 7eb272213cbfe11d9e466a2163b6c9c2ed8d2542fc6f08283fd1bf222cac8994, 105 7837cfede57fe45ec20d553c6c2b0c498ab324daae23bfd202c0ffea9f71a7f6, 106 0da1dda3fa56d75d6413fc9198949d3ebc2102d77574cb5fe0739674687609f2, 107 a02195e9599eb249f79103912d27eb4f62d8a0f59e4763b0dd8250fee5f3f020, 108 81ce6b21b9c1e01dc02099c4f9387699547b08b28b387d09a56e0cb0a8af22fc |
| PS, HT09 | `sociopatterns/` | `https://sociopatterns.org/assets/data/primaryschool.csv.gz`, `.../primaryschool_metadata.txt`, `.../ht2009_contact_list.dat.gz` | 5c93d9f5a61ad44b3c90fb5146d42345ae86dd44d77bc5da2461989d3d547fc9, 92844d1206ba38825d6ca801563cfc5c0f1c993f3efd012012c21aceaeff4329, 43014e65bb8f6aa7d75d36a8ab61279b617bee88197e04a330c4386305ae1738 |
| EEG | `chbmit_b/` | PhysioNet CHB-MIT 1.0.0 (`https://physionet-open.s3.amazonaws.com/chbmit/1.0.0/`, faster than physionet.org): `chb01/chb01_{03,04,15,16,18,21,26}.edf`, `chb03/chb03_{01,02,03,04,34,35,36}.edf`, `chb05/chb05_{06,13,16,17,22}.edf`, the three `chbNN-summary.txt` and `SHA256SUMS.txt` | every file is checked against PhysioNet's `SHA256SUMS.txt` (f1698efeb9fef887e55cedf6eda13386d9a888db663aec48b2aa194b7b753d2b) |

## Peel and Clauset's series

| IDs | dir | source | files (sha256) |
|---|---|---|---|
| MIT_PC | `konect_mit/` | KONECT copy of the Reality Mining Bluetooth contacts, `http://konect.cc/files/download.tsv.mit.tar.bz2` (96 subjects; the original `realitymining.mat` needs a registration) | `download.tsv.mit.tar.bz2` 5c3986275ffa54351a3caf84a122c857f9160114de935a3e81c153ad450547ed |
| ENRON_PC | `enron_cmu/` | CMU Enron corpus, `https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz` (read inside the tarball) | `enron_mail_20150507.tar.gz` b3da1b3fe0369ec3140bb4fbce94702c33b7da810ec15d718b3fadf5cd748ca7 |
| ENRON_PC, GHRG | `peel_code/` | Peel's archived homepage, `https://github.com/piratepeel/piratepeel.github.io` at commit `08af476ff943a76f6fcec2116c6c612e851502f7`, folder `old/code/` | `addresses.txt` a9b8ea306f6e1f97d93f1fc00a637267bcee93992b1d1d7ceb6b2df8b86bd74f, `enron_parser.py` 3d950a1ce9f84f31007dd0fe5bf796746e4989d594e417d2550b3228181e6fac (read, not run), `enrontimeline.txt` d9fb08cb9ebbcd59ab64d58024f503d9cf7689e53c9ea21daf69273c80108081, `cpdetectr1.3.tar.gz` 7ff247b94b13c62053775a45af4855e5efc0078171be91ef436247e142043765 (the authors' GHRG code: step 17 compiles its fitHRG and consensusHRG unchanged), `1403.0989v2.pdf` 061ec781eb58225a259d51481ad52436f93149699aec621114942b1f9df0a965 (the paper; step 17 reads Fig. 8 from it) |

## External code

Each is run from outside its folder. `repro/13_bench_g0.py` runs LAD's code with three small shims and s-GNN's code
through `methods/sgnn.py` (also used by `repro/22_settings.py --sgnn`). SCPD is run as the port `methods/scpd.py` of
`https://github.com/shenyangHuang/SCPD` at commit `187831f`, which needs no download.

* `lad/`: LAD, `https://github.com/shenyangHuang/LAD` at commit `58330238`.
* `sgnn/DyNNet/`: s-GNN, `https://github.com/dsulem/DyNNet` at commit `2cd2035`.
