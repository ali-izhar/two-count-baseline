"""MAG, FL and SCC: the three real datasets of SCPD (Huang et al., PAKDD 2023, arXiv 2305.08750), TGB-hosted zips.

Raw: data/raw/{mag,fl,scc}/ (the zips and their extracted edge lists). Snapshots follow the SCPD code
(github.com/shenyangHuang/SCPD, datasets/*_loader.py, subroutines/dos.py, subroutines/Anomaly_Detection.py). Every
stored matrix is symmetric, binary, without self-loops, on a FIXED vertex set; SCPD's own edge weight is kept in
the extra key `weights` (aligned with `indices`).
"""
import os

import numpy as np
import pandas as pd

from data.downloaders.snapshots import pack, sha256

SCPD_ARXIV = 'Huang, Danovitch, Rabusseau, Rabbany. Fast and Attributed Change Detection on Dynamic Graphs with Density of States. PAKDD 2023, arXiv:2305.08750'
TGB_HOST = 'https://object-arbutus.alliancecan.ca/swift/v1/14c95234f6cd4a21a47deafe20cce2a7/tgb'


# ----------------------------------------------------------------------------------------------- MAG
def mag(raw_dir):
    src = os.path.join(raw_dir, 'coauth-MAG-History-full-edgelist.txt')
    df = pd.read_csv(src, header=None, names=['year', 'u', 'v', 'w'], dtype=np.int64)
    assert np.all(np.diff(df.year.to_numpy()) >= 0), 'file not sorted by year'
    raw_rows = len(df)
    df = df[df.year >= 1837].reset_index(drop=True)          # geology_loader: years < 1837 skipped
    years = np.arange(1837, int(df.year.max()) + 1); T = len(years)
    assert set(df.year.unique()) == set(years), 'missing years'
    n_self = int((df.u == df.v).sum())
    vid = np.unique(np.concatenate([df.u.to_numpy(), df.v.to_numpy()]))
    a = np.searchsorted(vid, np.minimum(df.u, df.v)); b = np.searchsorted(vid, np.maximum(df.u, df.v))
    snap = (df.year.to_numpy() - 1837).astype(np.int64)
    k = pd.DataFrame({'s': snap, 'a': a, 'b': b, 'w': df.w.to_numpy(), 'o': np.arange(len(df))})
    k = k[k.a != k.b]
    dedup = k.groupby(['s', 'a', 'b'], sort=True).agg(w=('w', 'last'), m=('o', 'size')).reset_index()   # nx.Graph: last weight wins
    offsets, r, c, w = pack(T, dedup.s.to_numpy(), dedup.a.to_numpy(), dedup.b.to_numpy(), dedup.w.to_numpy())
    # SCPD Sec. 6.1 / code: MAG_geology(historydosN20.pkl, 0, 180): spectrums[0:-1] -> years 1837..2016 scored
    ev = [  # onset year -> snapshot index; provenance: SCPD Sec. 6.1, Fig. 4 and plot_dblp_spectro (bars (24,28),(102,108),(110,111),(113,116), line 97)
        {'name': 'American Civil War', 'onset_year': 1861, 'end_year': 1865, 'dates': '1861-04-12 to 1865 (Lee surrenders 1865-04-09)', 'scpd_bar': [24, 28]},
        {'name': "Adolf Hitler's rise to power ('becomes president' in Fig. 4)", 'onset_year': 1934, 'end_year': 1934, 'dates': '1934-08-02 Hitler head of state (Fuehrer) after Hindenburg death; chancellor since 1933-01-30', 'scpd_bar': [97, 97]},
        {'name': 'Second World War', 'onset_year': 1939, 'end_year': 1945, 'dates': '1939-09-01 to 1945-09-02', 'scpd_bar': [102, 108]},
        {'name': 'First Kashmir War', 'onset_year': 1947, 'end_year': 1948, 'dates': '1947-10-22 to 1949-01-01 ceasefire (SCPD: 1947-1948)', 'scpd_bar': [110, 111]},
        {'name': 'Korean War', 'onset_year': 1950, 'end_year': 1953, 'dates': '1950-06-25 to 1953-07-27 armistice', 'scpd_bar': [113, 116]},
    ]
    for e in ev:
        e['index'] = e['onset_year'] - 1837
        e['citation'] = 'Event and years as stated in SCPD Sec. 6.1 (arXiv 2305.08750). Calendar dates: general historical record (e.g. Encyclopaedia Britannica); not re-fetched in this build.'
    meta = {
        'id': 'MAG', 'name': 'MAG-History co-authorship network (SCPD version)',
        'source_url': f'{TGB_HOST}/history_scpd.zip',
        'source_url_in_scpd_readme': 'https://object-arbutus.cloud.computecanada.ca/tgb/history_scpd.zip (HTTP 404 on 2026-10-23; same file served from the alliancecan host above)',
        'sha256': {'history_scpd.zip': sha256(os.path.join(raw_dir, 'history_scpd.zip')), 'coauth-MAG-History-full-edgelist.txt': sha256(src)},
        'origin': 'Benson et al. 2018 simplicial data (coauth-MAG-History), MAG papers tagged History; SCPD projects each paper (simplex) to a clique. File rows: year,u,v,w.',
        'snapshot_rule': 'one snapshot per calendar year 1837..2017 (SCPD geology_loader.load_temporarl_edgelist: rows with year < 1837 dropped; edges of that year only, not cumulative). '
                      'Undirected; self-loops dropped here (SCPD keeps them in nx.Graph); vertex set = every author in 1837..2017, relabelled 0..n-1 in increasing raw id (vertex_ids). '
                      'SCPD weight = last w of the pair in that year (nx add_edge overwrite), stored in `weights`; SCPD uses the weighted normalized Laplacian on the ACTIVE authors only.',
        'counts': {'raw_rows': raw_rows, 'rows_kept': len(df), 'self_loop_rows_dropped': n_self, 'duplicate_pair_rows_merged': int(len(k) - len(dedup)), 'n': len(vid), 'T': T,
                    'undirected_edges_total': len(dedup), 'mean_edges_per_snapshot': float(len(dedup) / T)},
        'timestamps': [str(y) for y in years],
        'scpd_protocol': {'score': 'SCPD: 50-bin DOS (Nmoment=20, nZ=100, Jackson) per snapshot, L2-normalised; context = principal left singular vector of last w signatures; Z = 1 - |cos|; '
                                 'difference Z_t - Z_{t-1}; max over windows; top percent_ranked snapshots are the anomalies',
                           'windows': [5, 10], 'percent_ranked': 0.05, 'scored_snapshots': 'indices 0..179 (years 1837..2016; spectrums[0:-1])', 'n_anomalies': 9,
                           'initial_window_zeroed': 'first max(windows)+1 = 11 snapshots get score 0'},
        'windows': {'scpd_short': 5, 'scpd_long': 10, 'delta': 2, 'note': 'delta (tolerance, years) is a BS choice; SCPD defines no tolerance'},
        'events': ev,
        'event_provenance': 'POST HOC: SCPD names these events after looking at its detections (Sec. 6.1: "many of the anomalies correspond to important historical events"). They are not pre-registered ground truth.',
        'used_by': [{'paper': SCPD_ARXIV, 'protocol': 'qualitative: top-5% SCPD scores compared with history (Fig. 4); SPOTLIGHT (Fig. 9) and EdgeMonitoring (Fig. 10) shown in App. B.4. No Hits@n or other metric on this dataset.'}],
        'paper_results': {
            'metric': 'none (qualitative). SCPD reports Hits@7 only on synthetic SBM/BA graphs (Table 4); on MAG it lists the top-5% snapshots.',
            'SCPD': {'detections_years': [1864, 1900, 1934, 1938, 1944, 1945, 1947, 1950, 1958], 'source': 'Fig. 4; reproduced exactly by BS/scpd_repro.py from historydosN20.pkl shipped in the SCPD repo',
                      'matched_by_paper_text': ['1864 Civil War', '1934 Hitler', '1938/1944/1945 WWII', '1947 Kashmir', '1950 Korean War'], 'unmatched': [1900, 1958]},
            'SPOTLIGHT_rrcf': {'detections_years': [1837, 1843, 1844, 1853, 1858, 1860, 1965, 1969, 1989], 'source': 'Fig. 9 (labels read from the figure)', 'paper_text': 'misses WWII as a top anomaly'},
            'EdgeMonitoring': {'detections': None, 'source': 'Fig. 10 shows a score over window index 0..220 with thresholds; no list is given', 'paper_text': "output is noisy, many points share high scores"},
            'LAD': 'not run on MAG (all eigenvalues infeasible)', 'runtime_s': {'SCPD_DOS': 30}},
    }
    return {'id': 'MAG', 'n': len(vid), 'T': T, 'coo': (offsets, r, c), 'seg': np.zeros(T, np.int64), 'events': [e['index'] for e in ev],
                'tol': 2, 'meta': meta, 'extra': {'weights': w, 'vertex_ids': vid.astype(np.int64), 'year': years.astype(np.int64)}}


# ----------------------------------------------------------------------------------------------- FL
def fl(raw_dir):
    d = raw_dir
    df = pd.read_csv(os.path.join(d, 'edges_all.txt'), header=None, names=['d', 'u', 'v', 'c'])
    order = pd.read_csv(os.path.join(d, 'node_order.txt'), header=None, names=['a', 'i'])
    country = pd.read_csv(os.path.join(d, 'node_country.txt'), header=None, names=['a', 'cc'], keep_default_na=False)
    cont = pd.read_csv(os.path.join(d, 'node_continent.txt'), header=None, names=['a', 'ct'], keep_default_na=False)
    oid = dict(zip(order.a, order.i - 1)); n = len(order)
    assert sorted(oid.values()) == list(range(n))
    df['i'] = df.u.map(oid); df['j'] = df.v.map(oid); assert df.i.notna().all() and df.j.notna().all()
    df['day'] = (pd.to_datetime(df.d) - pd.Timestamp('2020-01-01')).dt.days
    assert not df.duplicated(['d', 'u', 'v']).any()
    T = 30                                          # SCPD: 30 weeks 2020-01-01..2020-07-28; days 210..212 (partial week) dropped
    df['wk'] = df.day // 7
    n_self = int((df.i == df.j).sum())
    k = df[(df.wk < T) & (df.i != df.j)]
    a = np.minimum(k.i, k.j).to_numpy(); b = np.maximum(k.i, k.j).to_numpy()
    # SCPD weight: days in the week the route was flown; on the SCPD .mat this equals min(D_ij + D_ji, 7) on ~98.4% of entries
    g = pd.DataFrame({'s': k.wk.to_numpy(), 'a': a, 'b': b, 'one': 1.0}).groupby(['s', 'a', 'b'], sort=True).one.sum().reset_index()
    g['w'] = np.minimum(g.one, 7)
    offsets, r, c, w = pack(T, g.s.to_numpy(), g.a.to_numpy(), g.b.to_numpy(), g.w.to_numpy())
    wk = lambda s: (pd.Timestamp(s) - pd.Timestamp('2020-01-01')).days // 7
    ev = [
        {'name': 'US entry ban on travellers from China (Proclamation 9984)', 'date': '2020-01-31', 'effective': '2020-02-02 17:00 EST = 22:00 UTC', 'index': wk('2020-02-02'),
             'citation': '85 FR 6709, https://www.federalregister.gov/documents/2020/02/05/2020-02424 (signing date 2020-01-31 and effective time verified 2026-10-23); SCPD Sec. 6.2 (attribute China, Fig. 5b)'},
        {'name': 'US ban on travel from 26 Schengen countries (Proclamation 9993)', 'date': '2020-03-11', 'effective': '2020-03-13 23:59 EDT = 2020-03-14 03:59 UTC', 'index': wk('2020-03-14'),
             'citation': '85 FR 15045, https://www.federalregister.gov/documents/2020/03/16/2020-05578 (signing date and effective time verified 2026-10-23); SCPD Sec. 6.2 / Fig. 5a "US bans EU flights"'},
        {'name': 'EU 30-day restriction of non-essential travel into the EU+ area', 'date': '2020-03-17', 'effective': 'agreed by the European Council 2020-03-17 (Commission proposal COM(2020) 115 of 2020-03-16); national implementation dates vary', 'index': wk('2020-03-17'),
             'citation': 'SCPD Sec. 6.2 (link to NYT timeline https://www.nytimes.com/article/coronavirus-timeline.html); COM(2020) 115 on eur-lex; consilium.europa.eu press page returned HTTP 403 to this session'},
    ]
    ev2 = [
        {'name': 'Canada bars entry of foreign nationals except US citizens (air travel)', 'date': '2020-03-18', 'index': wk('2020-03-18'),
             'citation': 'SCPD App. B.3 (footnote https://www.cihi.ca/en/covid-19-intervention-timeline-in-canada, HTTP 404 now; archived copy on web.archive.org); PM of Canada news release 2020-03-16 https://www.pm.gc.ca/en/news/news-releases/2020/03/16/prime-minister-announces-new-actions-under-canadas-covid-19-response ("March 18, 2020")'},
        {'name': 'Canada-US border closed to non-essential travel', 'date': '2020-03-21', 'index': wk('2020-03-21'),
             'citation': 'SCPD App. B.3; US-Canada joint statement 2020-03-20 (dhs.gov page returned HTTP 403 to this session; date as stated by SCPD)'},
    ]
    events = sorted({e['index'] for e in ev})
    meta = {
        'id': 'FL', 'name': 'COVID flight network (SCPD version, OpenSky weekly airport graphs)',
        'source_url': f'{TGB_HOST}/flight_scpd.zip',
        'source_url_in_scpd_readme': 'https://object-arbutus.cloud.computecanada.ca/tgb/flight_scpd.zip (HTTP 404 on 2026-10-23)',
        'upstream': 'OpenSky COVID-19 flight dataset, Zenodo record 3974209 (Olive, Strohmeier, Luebbe; SCPD refs [19,15]); flight_scpd.zip is SCPD\'s cleaned daily edge list: date,origin,destination,flights; top 3000 airports by total degree; flights without known origin/destination removed',
        'sha256': {f: sha256(os.path.join(d, f)) for f in ['flight_scpd.zip', 'edges_all.txt', 'node_order.txt', 'node_country.txt', 'node_continent.txt']},
        'snapshot_rule': 'week t = days 2020-01-01+7t .. +6 (Wednesday to Tuesday), t = 0..29 (2020-01-01..2020-07-28); the 3 days 07-29..07-31 are dropped. '
                      'Vertex index = node_order id - 1 (n = 3000 airports, fixed). Undirected; self-loops (same-airport flights) dropped (SCPD processed graphs have none). '
                      '`weights` = min(D_ij + D_ji, 7), D = directed day count (SCPD: "how many days in a week this edge was observed", [0,7]).',
        'verification': 'Edge pattern equals the SCPD processed graphs (repo subroutines/ADOS/processed.zip, covidflight_all_graphs.mat) in weeks 1..28 exactly; week 0 differs by 1 pair; '
                     'week 29 in the .mat has 5948 fewer entries than days 203..209 of the raw file (cause not found). SCPD .mat matrices are sized to max active id + 1 (not fixed n). '
                     'SCPD weight rule reproduced on 98.4% of entries (the rest are 6 vs 7 and slightly asymmetric in the .mat).',
        'counts': {'raw_rows': len(df), 'self_loop_rows': n_self, 'n': n, 'T': T, 'undirected_edges_total': len(g), 'mean_edges_per_snapshot': float(len(g) / T)},
        'timestamps': [str((pd.Timestamp('2020-01-01') + pd.Timedelta(days=7 * t)).date()) for t in range(T)],
        'timestamp_note': 'SCPD figure labels name each week by the day BEFORE its first day (index 11 is printed "2020-03-17" but covers 2020-03-18..03-24).',
        'scpd_protocol': {'windows': [2, 4], 'percent_ranked': 0.07, 'scored_snapshots': 'indices 0..28 (skynet_gdos.pkl has 30 rows; spectrums[0:-1])', 'n_anomalies': 2,
                           'attribute_runs': 'China (Fig. 5b, china.pkl) and Canada (Fig. 8) use local DOS with an indicator of the country\'s airports (node_country.txt)'},
        'windows': {'scpd_short': 2, 'scpd_long': 4, 'delta': 2, 'note': 'delta (weeks) is a BS choice; SCPD defines no tolerance'},
        'events': ev, 'events_secondary_canada': ev2,
        'event_provenance': 'Named in SCPD Sec. 6.2 / App. B.3 next to its detections (post hoc). Dates verified against primary sources where noted.',
        'used_by': [{'paper': SCPD_ARXIV, 'protocol': 'qualitative: top-7% weeks of the SCPD score (Fig. 5a), attribute runs for China (Fig. 5b) and Canada (Fig. 8); SPOTLIGHT (Fig. 11) and EdgeMonitoring (Fig. 12). No numeric metric.'}],
        'paper_results': {
            'metric': 'none (qualitative)',
            'SCPD': {'detections_week_index': [11, 12], 'printed_labels': ['2020-03-17', '2020-03-24'], 'week_start': ['2020-03-18', '2020-03-25'],
                      'source': 'Fig. 5a; reproduced exactly by BS/scpd_repro.py from skynet_gdos.pkl shipped in the SCPD repo'},
            'SCPD_attribute_China': {'printed_labels': ['2020-02-04', '2020-02-18', '2020-03-10'], 'reproduced_from_china_pkl_week_index': [7, 10],
                                      'note': 'Fig. 5b prints 3 detections; china.pkl with the python pipeline (7%) gives weeks 7 and 10 only; the 02-04 detection (index 5) is not reproduced'},
            'SCPD_attribute_Canada': {'printed_labels': ['2020-03-24', '2020-03-31'], 'source': 'Fig. 8 (MATLAB LDOS run; no python embedding shipped)'},
            'SPOTLIGHT_sum': {'detections': ['2020-02-11 (printed label)', 'index 23', 'index 28'], 'source': 'Fig. 11'},
            'EdgeMonitoring': {'detections': None, 'source': 'Fig. 12 shows scores and thresholds per window, no list; paper text: "also detects mid March"'},
            'runtime_s': {'SCPD': 5}},
    }
    cc = country.set_index('a').cc.reindex(order.sort_values('i').a).to_numpy().astype('U2')
    ct = cont.set_index('a').ct.reindex(order.sort_values('i').a).to_numpy().astype('U2')
    return {'id': 'FL', 'n': n, 'T': T, 'coo': (offsets, r, c), 'seg': np.zeros(T, np.int64), 'events': events, 'tol': 2, 'meta': meta,
                'extra': {'weights': w, 'events_secondary': np.array(sorted({e['index'] for e in ev2}), np.int64),
                     'vertex_ids': order.sort_values('i').a.to_numpy().astype('U8'), 'vertex_country': cc, 'vertex_continent': ct}}


# ----------------------------------------------------------------------------------------------- SCC
SC_EVENTS = [  # SCPD Table 5 (App. B.5), "as stated in Chartalist"; (date, type, text); * = SCPD's two most significant (coloured) rows
    ('2022-04-11', 'positive', 'BlackRock and Fidelity Back USDC in 400 Million Funding Round'),
    ('2022-04-19', 'positive', 'Terra UST takes over BUSD to become third largest stablecoin'),
    ('2022-05-06', 'negative', 'LARGE amounts of UST selling on ANCHOR (approx 500million)'),
    ('2022-05-07', 'negative', 'UST depegs LFG deploys assets to defend peg (750million in BTC sold to buy UST. BTC value drops) *'),
    ('2022-05-08', 'negative', 'UST Depegs again to 35 cents LUNA keeps falling'),
    ('2022-05-11', 'negative', 'LUNA Collapses by almost 99 %'),
    ('2022-05-11', 'negative', 'UST Downfall Brings Record Volumes to Curve: IntoTheBlock'),
    ('2022-05-12', 'negative', 'Terra Halts Its Blockchain and Binance Suspends LUNA Trading'),
    ('2022-05-13', 'negative', 'LUNA essentially worthless (1/100th of a cent)'),
    ('2022-05-15', 'negative', 'Luna Foundation Guard 80000 Bitcoin Lighter After Failing to Defend UST Peg'),
    ('2022-05-26', 'positive', 'Tether Token (USDT) Launches on Polygon'),
    ('2022-06-08', 'positive', 'Tether Tokens (USDT) to Launch on Tezos'),
    ('2022-06-10', 'negative', 'Binance CEO mentions selling 500 million in FTX holdings'),
    ('2022-06-12', 'positive', 'TRON DAO RESERVE DEPLOYS Funds (700million deployed)'),
    ('2022-06-12', 'positive', 'USD 500 million in USDC added to TRON DAO USDD reserve'),
    ('2022-08-07', 'negative', 'Circle freezes blacklisted Tornado Cash smart contract addresses'),
    ('2022-08-10', 'unknown', 'FTX shows sign of trouble and Binance CEO and FTX come to non binding deal for a buyout of FTX'),
    ('2022-08-10', 'negative', 'MakerDAO should seriously consider depegging DAI from USD - Founder'),
    ('2022-08-21', 'negative', 'USDC whales leave and trading volume soars'),
    ('2022-08-30', 'positive', 'Centre Consortium Announces Release of USD Coin Version 2.0'),
    ('2022-08-30', 'positive', 'MakerDAO Christensen Pushes Endgame Plan to Save DAI From Attack'),
    ('2022-09-01', 'negative', 'Binance Announces LUNC and USTC Deposit And Withdrawal Suspension'),
    ('2022-09-04', 'positive', 'Luna price has picked momentum as the community continues burning and staking LUNC tokens'),
    ('2022-09-05', 'negative', 'FTX To Suspend LUNC and USTC Deposits And Withdrawals and Delist ANC'),
    ('2022-09-06', 'negative', 'Binance - LUNC and USTC Deposits and Withdrawal Suspended'),
    ('2022-09-08', 'positive', 'Huobi And BTCEX Joins Binance To Support 1.2% Tax Burn For Terra Classic (LUNC)'),
    ('2022-09-10', 'negative', 'Binance Backs out of Deal with FTX'),
    ('2022-09-11', 'positive', 'Binance Announces Zero-Fee Trading For Luna'),
    ('2022-09-11', 'positive', 'Tether (USDT) Launches on Near Network'),
    ('2022-09-12', 'negative', 'FTX Suspends LUNA and UST Deposits and Withdrawals'),
    ('2022-09-27', 'positive', 'USDC coming to five additional blockchain ecosystems *'),
    ('2022-10-06', 'positive', 'MakerDAO goes ahead with $500M investment in treasuries and bonds'),
    ('2022-10-08', 'positive', 'USTC jumps over 35% in 24 hours as re-peg proposal emerges'),
    ('2022-10-12', 'positive', 'Binance Adds UST to Borrowable Assets List'),
    ('2022-10-12', 'positive', 'Tether change reserves to US treasury bills'),
    ('2022-10-19', 'positive', 'UST Trading Vol Skyrockets By 240%'),
    ('2022-10-19', 'positive', 'Tether announces Available in Over 24000 ATMs Across Brazil on 3 November'),
    ('2022-10-26', 'positive', 'UST climbs 15% in 24 hours reclaiming $ 400 million market cap'),
]
SC_AUDIT = {  # from repo/prereg/LABEL_AUDIT.md (SNAP event_data.csv dates are SCPD date + 1 day for most rows)
    'Binance CEO mentions selling 500 million in FTX holdings': 'contradicted: true date 2022-11-06 (outside the series); removed in the prereg SC build (DEVIATIONS 12)',
    'FTX shows sign of trouble and Binance CEO and FTX come to non binding deal for a buyout of FTX': 'contradicted: true date 2022-11-08 (outside the series); removed in prereg SC',
    'Binance Backs out of Deal with FTX': 'contradicted: true date 2022-11-09 (outside the series); removed in prereg SC',
    'Terra UST takes over BUSD to become third largest stablecoin': 'audit: 2022-04-18 (Cointelegraph 18 Apr 2022)',
    'FTX Suspends LUNA and UST Deposits and Withdrawals': 'audit: the Sept FTX action is LUNC/USTC, in force 2022-09-12 14:00 UTC (announced 09-05); duplicate of the 09-05 row',
    'MakerDAO goes ahead with $500M investment in treasuries and bonds': 'audit: executive vote 2022-10-05, first reports 2022-10-06',
    'UST climbs 15% in 24 hours reclaiming $ 400 million market cap': 'audit: 2022-10-28 (Finbold 28 Oct 2022)',
    'UST Depegs again to 35 cents LUNA keeps falling': 'audit: second depeg 05-09; 35-cent level first on 05-11',
}


def encode(src):
    """File-order rows -> int codes; vocabulary grows chunk by chunk (sorted addresses within a 5M-row chunk)."""
    vocab, T, U, V, W = {}, [], [], [], []
    for ch in pd.read_csv(src, chunksize=5_000_000, dtype={'time': np.int64, 'src': str, 'dst': str, 'weight': np.float64}):
        codes = []
        for col in ('src', 'dst'):
            uniq, inv = np.unique(ch[col].to_numpy(dtype=object), return_inverse=True)
            ids = np.fromiter((vocab.setdefault(a, len(vocab)) for a in uniq), dtype=np.int64, count=len(uniq))
            codes.append(ids[inv].astype(np.int32))
        T.append(ch.time.to_numpy()); U.append(codes[0]); V.append(codes[1]); W.append(ch.weight.to_numpy())
    return np.concatenate(T), np.concatenate(U), np.concatenate(V), np.concatenate(W), len(vocab)


def scpd_days(t):
    """SCPD coin_loader.parse_daily_graphs: rows in file order, a new day starts when t > end of the current day."""
    day = np.empty(len(t), np.int32)
    end, d = (int(t[0]) // 86400) * 86400 + 86400, 0
    for i, x in enumerate(t.tolist()):
        if x > end:
            d += 1
            end += 86400
        day[i] = d
    return day


def scc(raw_dir):
    src = os.path.join(raw_dir, 'coin_edgelistv3.csv')
    time, u, v, w, n = encode(src)
    day = scpd_days(time)
    T = int(day.max()) + 1
    t0 = int(time[0]) // 86400 * 86400
    del time
    n_rows = len(u); n_self = int((u == v).sum())
    keep = u != v
    a = np.minimum(u, v)[keep].astype(np.int64); b = np.maximum(u, v)[keep].astype(np.int64); s = day[keep].astype(np.int64); ww = w[keep]
    del u, v, w
    key = (s * n + a) * n + b                                      # < 215 * 6.8e6^2 ~ 1e16 < 2^63
    o = np.lexsort((np.arange(len(key)), key))                     # stable: last row in file order wins (nx overwrite)
    key, ww = key[o], ww[o]; del o
    last = np.r_[key[1:] != key[:-1], True]
    key, ww = key[last], ww[last]
    b = key % n; key //= n; a = key % n; s = key // n; del key
    offsets, r, c, wf = pack(T, s, a, b, ww)
    dates = [str((pd.Timestamp(t0, unit='s') + pd.Timedelta(days=d)).date()) for d in range(T)]
    ev = []
    for dt, typ, txt in SC_EVENTS:
        e = {'date': dt, 'type': typ, 'event': txt.rstrip(' *'), 'index': dates.index(dt), 'scpd_major': txt.endswith('*'),
                 'citation': 'SCPD Table 5 (App. B.5), "as stated in Chartalist", https://www.chartalist.org/eth/StablecoinAnalysis.html'}
        if e['event'] in SC_AUDIT:
            e['audit_note'] = SC_AUDIT[e['event']]
        ev.append(e)
    events = sorted({e['index'] for e in ev})
    major = sorted({e['index'] for e in ev if e['scpd_major']})
    meta = {
        'id': 'SCC', 'name': 'Stablecoin ERC20 transaction network (SCPD version)',
        'source_url': f'{TGB_HOST}/stablecoin_scpd.zip',
        'source_url_in_scpd_readme': 'https://object-arbutus.cloud.computecanada.ca/tgb/stablecoin_scpd.zip (HTTP 404 on 2026-10-23)',
        'upstream': 'Chartalist stablecoin ERC20 transactions (Shamsi et al., NeurIPS 2022 Datasets and Benchmarks, SCPD ref [20]); the zip holds SCPD\'s coin_edgelistv3.csv (time,src,dst,weight = token value), read by SCPD dos.coin_dos',
        'sha256': {'stablecoin_scpd.zip': sha256(os.path.join(raw_dir, 'stablecoin_scpd.zip')), 'coin_edgelistv3.csv': sha256(src)},
        'snapshot_rule': 'SCPD coin_loader.parse_daily_graphs: rows in FILE order; day 0 starts at floor(t0/86400)*86400 = 2022-04-01 00:00 UTC; a new day starts when t > end '
                      '(a row with t == end stays in the earlier day; 810 rows differ from the UTC calendar day; 14 rows are out of time order by <= 2 s). 215 days 2022-04-01..2022-11-01 (the last day ends 04:53:59 UTC, partial). '
                      'Undirected; self-loops dropped (SCPD keeps them); vertex set = all 6,715,034 addresses of the file (fixed; index = sc_encode.py vocabulary order: 5M-row chunks in file order, sorted addresses within a chunk; addresses in data/SCC_vertices.npy). '
                      '`weights` = value of the last transaction of the pair that day in file order (nx add_edge overwrite).',
        'layout_deviation': 'No `indptr` key: a (T, n+1) int64 array would be 11.5 GB. The key `rows` (int32, same length as `indices`) holds the row of each entry; '
                         'entries are sorted by (row, col) within each snapshot, so indptr_t = cumsum(bincount(rows[off_t:off_t+1], minlength=n)). Use BS/load_bs.py (load one snapshot at a time, or compact to active vertices).',
        'counts': {'raw_rows': n_rows, 'self_loop_rows_dropped': n_self, 'n': n, 'T': T, 'undirected_edges_total': int(offsets[-1] // 2), 'mean_edges_per_snapshot': float(offsets[-1] / 2 / T)},
        'timestamps': dates,
        'scpd_protocol': {'score': 'SCPD DOS (Nmoment=20, nZ=100, 50 bins) on the weighted normalized Laplacian of each day\'s ACTIVE addresses', 'windows': 'not stated in paper or repo for this dataset',
                           'percent_ranked': 'not stated (Fig. 6 marks 8 days)'},
        'windows': {'delta': 3, 'note': 'delta (days) is a BS choice; SCPD defines no tolerance and does not state its windows for this dataset'},
        'events': ev,
        'event_rule': 'events = every distinct date of SCPD Table 5 (as SCPD uses them, no correction); events_major = the two rows SCPD colours as most significant. '
                   'Prereg LABEL_AUDIT corrections are attached as audit_note but NOT applied.',
        'event_provenance': 'Chartalist event list; mostly off-chain news, several are reports or price moves (see repo/prereg/LABEL_AUDIT.md).',
        'used_by': [{'paper': SCPD_ARXIV, 'protocol': 'qualitative: SCPD score with marked anomalies (Fig. 6), compared with the Table 5 event list. No numeric metric. No baseline shown on this dataset.'}],
        'paper_results': {
            'metric': 'none (qualitative)',
            'SCPD': {'detections': ['05-04', '06-09', '06-16', '07-25', '07-31', '09-26', '10-01', '10-23'], 'year': 2022, 'source': 'Fig. 6 labels (not reproducible: no stablecoin embedding is shipped)',
                      'paper_text': '05-04 is the first significant jump, "3 days early" for the Terra Luna crash (05-07); the highest peak is 09-26 (USDC to five more chains, Table 5 date 09-27)'},
            'baselines': 'none reported on this dataset'},
        'prereg_SC_comparison': 'The repo prereg SC (DEVIATIONS 6/11/12/14/18) used SNAP / TGB tgbl-coin daily graphs 2022-04-01..2022-11-01 with event_data.csv dates (mostly SCPD date + 1 day) after audit corrections, and dropped the partial first/last days. '
                             'Its builder and cache are not on disk any more (data/SC/series.pkl missing), so it could not be compared edge by edge.',
    }
    return {'id': 'SCC', 'n': n, 'T': T, 'coo': (offsets, r, c), 'seg': np.zeros(T, np.int64), 'events': events, 'tol': 3, 'meta': meta,
                'extra': {'weights': wf, 'events_major': np.array(major, np.int64)}, 'indptr': False}
