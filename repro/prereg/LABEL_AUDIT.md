# Label audit (independent second check), 2026-10-13

Scope: every label in `data/{PS,HT09,OS,SC,EEG,PM}/labels.json`, plus DEVIATIONS items 1-12.
Method: each source was re-opened or re-downloaded by the auditor; a second source was used where one
exists. No detection score, feature series or `series*.pkl` was opened.

Verdicts: **OK** = date/time, timezone, onset nature and index all confirmed; **OK\*** = confirmed, with a
caveat noted; **WRONG** = shipped value contradicted (correct value and source given);
**UNVERIFIABLE** = no external source found that pins the date. "Type" column: whether the a-priori
V/S/M type is defensible from the event text alone (Yes / Contestable).

## Summary of problems

- **SC: 4 shipped onsets are still wrong** (label 19: 2022-04-18 not 04-20; label 165: the FTX LUNC/USTC
  suspension took effect 2022-09-12 and was announced 09-05, so the label merges into label 161 or is
  removed; label 189: 2022-10-06 (vote 10-05) not 10-07; label 209: 2022-10-28 not 10-27). Two heads are
  UNVERIFIABLE (157 head, 161). Six merged members are off by 1-2 days, with no effect on any onset.
- **OS**: all 7 effective times and UTC days confirmed. The file's seg-1 indices (25, 115) are pre-gap; the
  scorer must shift them to 24 and 114 (DEVIATIONS item 8).
- **EEG**: 19/19 onsets confirmed by summary files and by the WFDB `.seizures` annotations. Type S contestable.
- **PM**: 98/98 change points; per-subject counts equal Table 4; an independent recount gives identical indices.
  Type S contestable.
- **PS / HT09**: all 4 + 22 times confirmed against the paper and a fresh Wayback download; PS onsets are
  nominal (staggered lunch turns).

---

## PS (SocioPatterns Primary School, Lyon, CEST)

Source re-fetched: Stehlé et al., PLoS ONE 6(8):e23176 (2011), Methods "Setting",
https://doi.org/10.1371/journal.pone.0023176. Quote confirmed verbatim:
"The school day runs from 8.30am to 4.30pm, with a lunch break from 12pm to 2pm, and two breaks of
20–25 min around 10.30am and 3.30pm." (The shipped quote drops the sentence "Lunches are served in a
common canteen, and a shared playground is located outside the main building." without an ellipsis;
cosmetic.) Second source: none gives clock times (dataset page silent; Gemmetto et al. 2014 silent).

Time origin: the dataset page says only "Time is measured in seconds" (no origin, no timezone). The builder
assumes t = seconds from local midnight 1 Oct 2009. Supported: first/last contacts 08:40/17:18 (day 1) and
08:33/17:08 (day 2) vs. the paper's 8.45am-5.20pm and 8.30am-5.05pm (all within 5 min; a UTC origin would put
them at 10:40-19:18 local, which contradicts the paper). Residual uncertainty about 1 bin.

Index check: day 1 first bin 08:40 -> 12:00 = 40, 14:00 = 64; day 2 first bin 08:30 -> 12:00 = 42,
14:00 = 66. All correct.

| Seg | Idx | Local time | Event | Type | Type defensible | Verdict |
|---|---|---|---|---|---|---|
| 0 | 40 | 2009-10-01 12:00 +02:00 | lunch start | M | Yes (paper: individuals drop at lunch, cross-class mixing) | OK\* (nominal whole-school time; 2 canteen turns of unknown time, so the true onset is staggered) |
| 0 | 64 | 2009-10-01 14:00 +02:00 | lunch end | M | Yes | OK\* (same caveat) |
| 1 | 42 | 2009-10-02 12:00 +02:00 | lunch start | M | Yes | OK\* |
| 1 | 66 | 2009-10-02 14:00 +02:00 | lunch end | M | Yes | OK\* |

`labels_breaks_approx.json` (secondary): times 10:30/15:30 are "around" times in the paper; correctly
marked `exact: false`. Not verifiable to the bin.

## HT09 (SocioPatterns Hypertext 2009, Turin, CEST)

Source re-downloaded: http://web.archive.org/web/20090619001905/http://www.ht2009.org:80/program.php.
The programme table text (Monday header to "Accepted papers") of the fresh download is identical to the
shipped copy `raw_ht09/programme/wayback_20090619001905_ht2009_program.php.html` (file sha differs only by
Wayback wrapper). Every quoted row and every colspan/rowspan in `labels.json` is confirmed, including:
Mon 10:30-11:45 workshop cell rowspan=2 (so no Mon 10:30 break); "11:45am - noon Coffee break" colspan=3;
"1:30 - 3pm Lunch break" colspan=3; "4:30 - 4:45pm Coffee break" colspan=3; "6:05 - 6:10pm Short break".
Second source (weak): Isella et al., J. Theor. Biol. 271:166 (2011), arXiv:1006.1260, Sect. on spreading:
participants "gather for the coffee break ... between 10:00 and 11:00" and "a second strong increase occurs
during the lunch break" (consistent; no clock times).

Time origin: dataset page "seconds since 8am on Jun 29th 2009 (UNIX ctime 1246255200)"; 1246255200 =
2009-06-29 06:00 UTC = 08:00 CEST. Consistent. KONECT cross-check reported by the builder.

Index check (first bins 08:00 / 07:55 / 08:10): all 22 indices recomputed and correct.

The programme is a plan; actual overruns are not published (applies to all rows).

| Seg | Idx | Local time (CEST) | Event | Type | Type defensible | Verdict |
|---|---|---|---|---|---|---|
| 0 | 45 | Mon 29 Jun 11:45 | coffee break start | M | Yes | OK |
| 0 | 48 | Mon 29 Jun 12:00 | workshops WS2/WS4 resume | M | Yes | OK |
| 0 | 66 | Mon 29 Jun 13:30 | lunch start | M | Yes | OK |
| 0 | 84 | Mon 29 Jun 15:00 | workshops WS1/WS3 start | M | Yes | OK |
| 0 | 102 | Mon 29 Jun 16:30 | coffee break start | M | Yes | OK |
| 0 | 105 | Mon 29 Jun 16:45 | workshops resume | M | Yes | OK |
| 1 | 31 | Tue 30 Jun 10:30 | coffee break start | M | Yes | OK |
| 1 | 34 | Tue 30 Jun 10:45 | keynote Adamic (plenary) | M | Yes | OK |
| 1 | 46 | Tue 30 Jun 11:45 | coffee break start | M | Yes | OK |
| 1 | 49 | Tue 30 Jun 12:00 | poster pitch + posters/demos | M | Yes | OK |
| 1 | 67 | Tue 30 Jun 13:30 | lunch start | M | Yes | OK |
| 1 | 85 | Tue 30 Jun 15:00 | parallel 1.2 / 3.1 | M | Yes | OK |
| 1 | 103 | Tue 30 Jun 16:30 | coffee break start | M | Yes | OK |
| 1 | 106 | Tue 30 Jun 16:45 | parallel 1.3 / 2.1 | M | Yes | OK |
| 2 | 28 | Wed 1 Jul 10:30 | coffee break start | M | Yes | OK |
| 2 | 31 | Wed 1 Jul 10:45 | keynote Baeza-Yates (plenary) | M | Yes | OK |
| 2 | 43 | Wed 1 Jul 11:45 | coffee break start | M | Yes | OK |
| 2 | 46 | Wed 1 Jul 12:00 | parallel 1.4 / 2.3 | M | Yes | OK |
| 2 | 64 | Wed 1 Jul 13:30 | lunch start | M | Yes | OK |
| 2 | 82 | Wed 1 Jul 15:00 | parallel 1.5 / 2.4 / SRC | M | Yes | OK |
| 2 | 100 | Wed 1 Jul 16:30 | coffee break start | M | Yes | OK |
| 2 | 103 | Wed 1 Jul 16:45 | parallel 1.6 / 3.2 / SRC | M | Yes | OK |

Note: Tue 12:00 is a transition from coffee break to a poster session in a shared foyer, i.e. a weaker
"return to rooms" than the other session-start labels. Type M still defensible.

## OS (OpenSky daily airport graphs, UTC days)

All primary URLs re-opened. Index check: seg 0 starts 2019-12-01 (01-23 = 53, 02-02 = 63, 03-14 = 104);
seg 1 starts 2021-11-01 (11-26 = 25, 02-24 = 115). Correct in the shipped (pre-gap-removal) day list.

| Seg | Idx | Effective (UTC) | Event | Effective-time check | Type | Type defensible | Verdict |
|---|---|---|---|---|---|---|---|
| 0 | 53 | 2020-01-23 | Wuhan outbound transport/airport shut | "starting 10 a.m. of Jan. 23" (People's Daily via airport-technology.com, 23 Jan 2020) = 02:00 UTC 23 Jan | S | Yes | OK |
| 0 | 63 | 2020-02-02 | US Procl. 9984 (China) | "effective at 5:00 p.m. eastern standard time on February 2, 2020"; issued 31 Jan (trumpwhitehouse.archives.gov) = 22:00 UTC 2 Feb | S | Yes | OK\* (only last 2 h of the UTC day are affected; most of the effect lands on 02-03, still inside [tau, tau+3]) |
| 0 | 104 | 2020-03-14 | US Procl. 9993 (Schengen), merged with EU external closure | "effective at 11:59 p.m. eastern daylight time on March 13, 2020"; issued 11 Mar. DST began 8 Mar 2020, so EDT = UTC-4 -> 03:59 UTC 14 Mar. Correct UTC day | M (merged S+M) | Yes (merge rule) | OK |
| (merged) | – | 2020-03-17 | EU coordinated restriction of non-essential travel, 30 days | European Council conclusions 17 Mar 2020 (EEAS page; LBC / Ahram 17 Mar). National implementation dates vary (not pinned by any source found); gap 3 days -> merges | M | Yes | OK\* (agreement date, not a legal effective date) |
| 1 | 25 | 2021-11-26 | Omicron: EU/UK flight suspensions from southern Africa | gov.pl page (27.11.2021): EU "as of 26 November 2021"; UK: direct flights from 6 countries suspended from 12:00 GMT Fri 26 Nov to 04:00 GMT 28 Nov (Lewis Silkin; Yahoo live blog) | S | Yes | OK |
| 1 | 115 | 2022-02-24 | Ukraine airspace closed, merged with EU-Russia closures | "closed to civilian flights starting from 00:45 GMT on Thursday" (Al Jazeera, 24 Feb 2022) | S (merged S+S) | Yes | OK |
| (merged) | – | 2022-02-27 | EU-Russia airspace closures | Germany: from 15:00 (CET) 27 Feb for three months (aviacionline; City AM says "2pm") = 13:00-14:00 UTC; Regulation (EU) 2022/334 / Decision (CFSP) 2022/335 of 28 Feb 2022 (published in the Official Journal 28.2.2022; EASA notice; issue number not verified) | S | Yes | OK\* (choice of 27 Feb vs 28 Feb decides the merge; both dates are documented) |

Scorer-side note (not a label error): `labels.json` indices are in the day list that still contains the
missing day 2021-11-23 (seg 1, index 22). Under DEVIATIONS item 8 / INTERPRETATIONS item 11 the day is
removed first, so the seg-1 labels must become 24 (Omicron) and 114 (Ukraine). Both scoring
implementations must apply this shift; the file does not.

## EEG (CHB-MIT, chb01/03/05)

First source: `raw_eeg/chb0{1,3,5}-summary.txt` (sha256 matches PhysioNet SHA256SUMS). Second, independent
source: the per-file WFDB annotation files `chbXX_YY.edf.seizures` downloaded from
https://physionet-open.s3.amazonaws.com/chbmit/1.0.0/ and parsed with `wfdb.rdann` (fs = 256; '[' / ']'
markers). All 19 onsets and offsets agree with both sources to the second. The seizure-file sets are
complete (chb01: 7, chb03: 7, chb05: 5; every other file in the three summaries has 0 seizures).
index = floor(onset/2) recomputed for all 19: correct.

Type: S is defensible (a seizure changes the inter-channel synchronisation pattern). It is **contestable**:
with an absolute |r| >= 0.5 threshold, ictal hypersynchrony is expected to raise the edge count, which is
what the V channel measures (edges), so M is at least as defensible a priori. This does not change any
onset; it matters only for the by-type breakdown.

| Seg | File | Onset s | Offset s | Idx | Summary | .seizures | Type | Verdict |
|---|---|---|---|---|---|---|---|---|
| 0 | chb01_03 | 2996 | 3036 | 1498 | 2996/3036 | 2996/3036 | S (contestable) | OK |
| 1 | chb01_04 | 1467 | 1494 | 733 | match | match | S (contestable) | OK |
| 2 | chb01_15 | 1732 | 1772 | 866 | match | match | S (contestable) | OK |
| 3 | chb01_16 | 1015 | 1066 | 507 | match | match | S (contestable) | OK |
| 4 | chb01_18 | 1720 | 1810 | 860 | match | match | S (contestable) | OK |
| 5 | chb01_21 | 327 | 420 | 163 | match | match | S (contestable) | OK |
| 6 | chb01_26 | 1862 | 1963 | 931 | match | match | S (contestable) | OK |
| 7 | chb03_01 | 362 | 414 | 181 | match | match | S (contestable) | OK |
| 8 | chb03_02 | 731 | 796 | 365 | match | match | S (contestable) | OK |
| 9 | chb03_03 | 432 | 501 | 216 | match | match | S (contestable) | OK |
| 10 | chb03_04 | 2162 | 2214 | 1081 | match | match | S (contestable) | OK |
| 11 | chb03_34 | 1982 | 2029 | 991 | match | match | S (contestable) | OK |
| 12 | chb03_35 | 2592 | 2656 | 1296 | match | match | S (contestable) | OK |
| 13 | chb03_36 | 1725 | 1778 | 862 | match | match | S (contestable) | OK |
| 14 | chb05_06 | 417 | 532 | 208 | match | match | S (contestable) | OK |
| 15 | chb05_13 | 1086 | 1196 | 543 | match | match | S (contestable) | OK |
| 16 | chb05_16 | 2317 | 2413 | 1158 | match | match | S (contestable) | OK |
| 17 | chb05_17 | 2451 | 2571 | 1225 | match | match | S (contestable) | OK |
| 18 | chb05_22 | 2348 | 2465 | 1174 | match | match | S (contestable) | OK |

## PM (PAMAP2, s-GNN recipe)

Table 4 of Sulem et al., Machine Learning 113:1-44 (2024), https://doi.org/10.1007/s10994-023-06405-x
(copy `raw_PM_paper/mlj_vor.txt`, lines 1226-1238) and arXiv:2203.15470 LaTeX (`input_core.tex` 595-604)
agree: change points 13, 13, 10, 11, 13, 13, 12, 13 for subjects 1-8. Shipped labels: 13, 13, 10, 11, 13,
13, 12, 13. **All 8 match.** The activity sets per subject also match the Table-4 check marks (103: no
5/6/7/24; 104: no 5/24; 107: no 24).

Independent recount by the auditor (own script, activity column only, 100-row windows from row 0, keep
windows whose 100 labels are equal and non-zero): kept-snapshot counts 2490, 2618, 1732, 2302, 2710, 2488,
2314, 2607 and change-point index lists **identical** to `labels.json` for all 8 subjects. (Snapshot
counts are +1 vs Table 4 for 105, 106, 108, as the README already states.)

Onset nature: each label is a real activity change, but each is also a splice of non-adjacent recording
parts (DEVIATIONS item 10); the placebo control handles this. Type S: defensible per the paper's rationale;
**contestable** for the same reason as EEG (absolute |r| > 0.2 threshold, so the edge count, i.e. V, is
free to move, e.g. lying -> sitting vs. running -> rope jumping).

| Subject | Labels | Table 4 | Recount | Indices identical | Type | Verdict |
|---|---|---|---|---|---|---|
| 101 | 13 | 13 | 13 | yes | S (contestable) | OK (all 13) |
| 102 | 13 | 13 | 13 | yes | S (contestable) | OK (all 13) |
| 103 | 10 | 10 | 10 | yes | S (contestable) | OK (all 10) |
| 104 | 11 | 11 | 11 | yes | S (contestable) | OK (all 11) |
| 105 | 13 | 13 | 13 | yes | S (contestable) | OK (all 13) |
| 106 | 13 | 13 | 13 | yes | S (contestable) | OK (all 13) |
| 107 | 12 | 12 | 12 | yes | S (contestable) | OK (all 12) |
| 108 | 13 | 13 | 13 | yes | S (contestable) | OK (all 13) |

## SC (SNAP ERC20 stablecoins, UTC days)

Checks done by the auditor: all 38 rows of `raw_SC/x/event_data.csv` have timestamps at exactly 00:00 UTC,
and their UTC days equal `events_unmerged.json`; all 19 label indices equal (day - 2022-04-01). External
dates: every one of the 34 kept events and the 4 removed events was searched (two search passes, each
event's article publication date checked; the four onset-moving findings below were re-opened by the auditor).
Day convention: an event's day = the UTC day of the first public report or of the effective time.

### Per label (onset = the head event's day)

| Idx | Shipped day | Type | Head event | True day found (source) | Onset nature | Type defensible | Verdict |
|---|---|---|---|---|---|---|---|
| 11 | 2022-04-12 | V | BlackRock/Fidelity back USDC $400M round | 2022-04-12 (Circle press room; The Block) | announcement, no on-chain action | Yes | OK |
| 19 | 2022-04-20 | V | UST passes BUSD, third-largest stablecoin | **2022-04-18** (Cointelegraph, pub. 18 Apr 2022: UST "has flipped Binance USD (BUSD) to become the third-largest stablecoin"; bitcoin.com 18 Apr) | milestone of a gradual trend, not an onset | Yes | **WRONG: day = 2022-04-18, index 17** |
| 36 | 2022-05-07 | M | Large UST selling on Anchor / Curve | 2022-05-07 (~21:44-21:57 UTC Curve withdrawals and 85M swap; Jump Crypto post-mortem; CoinDesk 8 May) | true onset of the Terra run | Yes | OK |
| 41 | 2022-05-12 | M | LUNA collapses ~99% | 2022-05-12 (CNBC 12 May) | collapse phase | Yes | OK |
| 45 | 2022-05-16 | M | LFG "80,000 BTC lighter" | 2022-05-16 (Fortune; The Block) | **retrospective report** of reserves spent 8-12 May; not an onset of a new change | Contestable (no new on-chain action that day) | OK\* (date right; onset nature weak) |
| 56 | 2022-05-27 | S | USDT launches on Polygon | 2022-05-27 (CoinDesk; The Block) | launch on another chain (no Ethereum action) | Contestable | OK |
| 69 | 2022-06-09 | S | USDT to launch on Tezos | 2022-06-09 (The Block; Cointelegraph) | announcement of a future launch on another chain | Contestable | OK |
| 73 | 2022-06-13 | S | TRON DAO Reserve deploys ~$700M | 2022-06-13 (Forklog; The Block) | on-chain reserve operation | Yes | OK |
| 129 | 2022-08-08 | S | Circle freezes Tornado Cash addresses | 2022-08-08 (The Block; Forklog; OFAC same day) | on-chain freeze | Yes | OK |
| 143 | 2022-08-22 | M | USDC whales leave, volume soars | 2022-08-22 (Forkast 06:31 UTC; Cointelegraph; Glassnode metric) | report of a trend (22-month low in whale share) | Contestable | OK\* |
| 152 | 2022-08-31 | S | MakerDAO "Endgame" plan to save DAI | 2022-08-31 article (CryptoBriefing; Decrypt); the Endgame v3 roadmap itself was posted ~2022-08-25 | governance proposal, no network action | Contestable (arguably no change) | OK\* (article day; the underlying post is ~6 days earlier) |
| 157 | 2022-09-05 | M | LUNC burning momentum (+ FTX LUNC/USTC suspension announcement, + Binance suspension in force 09-07) | head article not found (UNVERIFIABLE); member "FTX to suspend LUNC/USTC" was announced **2022-09-05** 09:59 UTC (Blockmedia; CoinGape 5 Sep), so the onset day is supported by a member | price trend (head); announcements (members) | M by merge: Yes | OK\* (onset supported by a member, not by the head) |
| 161 | 2022-09-09 | M | Huobi/BTCEX join Binance on LUNC 1.2% burn tax | head UNVERIFIABLE (CoinGape lists the exchanges without a date; Cobo burn-tax notice dated 2022-09-09); members 09-12 confirmed (Binance zero-fee *Convert* for **LUNC**/USTC, CoinGape 12 Sep; USDT on NEAR, Investing.com/CryptoSlate 12 Sep) | announcements | M by merge: Yes | UNVERIFIABLE (plausible) |
| 165 | 2022-09-13 | S | "FTX Suspends LUNA and UST Deposits and Withdrawals" | Sept-2022 FTX action = suspension of LUNC/USTC deposits and withdrawals **in force 2022-09-12 14:00 UTC**, announced 2022-09-05 (CoinGape, pub. 5 Sep 2022: "from September 12 at 14:00 UTC"). No FTX LUNA/UST deposit suspension is reported for 13 Sep. (The only "LUNA/UST" FTX notice is 2022-05-26, a market suspension for the Terra 2.0 airdrop; Decrypt 26 May.) | duplicate of the 09-06 member of label 157 | Yes | **WRONG: true day 2022-09-12 (index 164); under the merge rule it then merges into label 161 (09-09, gap 3), so label 165 should not exist as a separate onset. If it is read as the May notice, it is misdated and falls under item 11 (remove). Either way: SC = 18 labels, S 6.** |
| 180 | 2022-09-28 | S | USDC coming to five more chains | 2022-09-28 (Circle press room, Converge22; Decrypt) | announcement of future multi-chain support | Contestable | OK |
| 189 | 2022-10-07 | M | MakerDAO $500M treasuries/bonds | **executive vote 2022-10-05; first reports 2022-10-06** (The Block, pub. 6 Oct 2022 12:16 EDT: vote "Wednesday", Oct 5; Bloomberg 6 Oct) | governance decision, off-chain asset purchase | M by merge: Yes | **WRONG: day = 2022-10-06 (index 188; 2022-10-05 / index 187 if the vote is used)**. Knock-on: member "USTC +35%" is 2022-10-09 (re-peg proposal) or 2022-10-10 (price move, Finbold / The Crypto Basic 10 Oct). With anchor 10-06 and member 10-10 the gap is 4, so the member would split off as its own V label (index 192). |
| 195 | 2022-10-13 | M | USTC borrowable on Binance Loans (+ Tether cuts commercial paper) | 2022-10-13 (DailyCoin via Coinspectator; CNBC and Kitco 13 Oct for Tether) | announcements | M by merge: Yes | OK |
| 202 | 2022-10-20 | V | UST trading volume +240% (+ Tether 24,000 ATMs in Brazil) | 2022-10-20 (Watcher.guru reports +206%; the 240% figure is not found); Tether ATM announcement 2022-10-20 21:35 UTC (The Block) | volume report | Yes | OK\* (date right; figure unverified) |
| 209 | 2022-10-27 | V | UST +15% in 24 h, $400M cap | **2022-10-28** (Finbold, pub. 28 Oct 2022, same-day market data) | price move | Yes | **WRONG: day = 2022-10-28, index 210** |

### Merged members whose own day is off (no effect on any onset)

| Member (shipped day) | True day | Effect |
|---|---|---|
| LFG $750M BTC defence (05-08) | announced ~03:00-05:00 UTC **05-09** (The Block 9 May 01:03 EDT); also the text says "sold", the source says "lend" | none (still in label 36) |
| UST "depegs again to 35 cents" (05-09) | second depeg 05-09 OK; the 35-cent level first on **05-11** | none (label 36) |
| Terra halt + Binance suspends LUNA (05-13) | first halt 05-12, second halt ~02:00 UTC 05-13; Binance 12-13 May | none (label 41) |
| LUNA essentially worthless (05-14) | ~**05-13** (CNBC 13 May) | none (label 41) |
| $500M USDC to TRON DAO reserve (06-13) | **06-14** 12:40 UTC (tweet; The Block 14 Jun) | none (label 73) |
| FTX to suspend LUNC/USTC, delist ANC (09-06) | announced **09-05**; in force 09-12 14:00 UTC | none (label 157) |
| Binance LUNC/USTC suspension (09-02) and in force (09-07) | 09-02 announced, 09-07 00:00 UTC in force: the two rows are one event | none (labels 152 and 157) |
| "Binance zero-fee trading for Luna" (09-12) | 09-12 OK, but it is zero-fee Convert for LUNC/USTC, not LUNA 2.0 trading | none (label 161) |

### Removed events (DEVIATIONS items 11-12): all four removals confirmed

| Shipped day | Event | True day | Source |
|---|---|---|---|
| 2022-06-11 | CZ: Binance to sell FTT | 2022-11-06 | BNN Bloomberg; Al Jazeera 9 Nov 2022 |
| 2022-08-11 | Binance-FTX non-binding deal | 2022-11-08 | SiliconANGLE 8 Nov; Al Jazeera |
| 2022-09-11 | Binance backs out of FTX deal | 2022-11-09 | TechCrunch 9 Nov; The Block |
| 2022-08-31 | USDC version 2.0 | 2020-08-27 | Circle blog; The Block |

### SC corrected label set (if the item-11 rule "remove or fix events an external source contradicts" is applied to the new findings)

Onset changes: 19 -> 17 (2022-04-18); 165 removed (merged into 161 or removed); 189 -> 188 (2022-10-06),
with "USTC +35%" possibly split off at 192 (2022-10-10); 209 -> 210 (2022-10-28). Result: 18 labels
(19 if the USTC member splits off). Which convention to use for the MakerDAO and USTC rows (vote vs first
report; proposal vs price move) is a protocol decision and must be logged in DEVIATIONS.md before any score.

Onset-nature caveat (all SC): 13 of 34 kept events are off-chain announcements, and several heads are
retrospective reports or trend descriptions (labels 19, 45, 143, 157 head, 202, 209). They are dated
correctly (except as listed) but are not clean onsets of a network change.

## Cross-cutting

1. **OS index shift is not in the file.** Seg-1 indices 25 and 115 assume the missing day 2021-11-23 is
   still present. DEVIATIONS item 8 removes it, so the scorer must use 24 and 114. Check both
   implementations.
2. **Type S for EEG and PM is contestable** (absolute correlation thresholds let the edge count, the V
   channel, move at the change). Not a date error; it affects only the by-type breakdown and the
   "S/G on S-type labels" secondary analysis.
3. **PS onsets are nominal.** The 12:00 lunch start covers two unpublished canteen turns; the true
   transitions are staggered. Times are as published.
4. PS time origin (local midnight) is an inference that fits the paper's recording hours within 5 min;
   the dataset page does not state it.
