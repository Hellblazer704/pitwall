# Data

Everything in pitwall comes from [FastF1](https://docs.fastf1.dev/), which
wraps the Formula 1 live timing feed. This document covers what is used, what is
wrong with it, and what each quirk does to a model that ignores it.

## What is used

| | |
|---|---|
| Source | FastF1 3.8, race sessions only |
| Training seasons | 2022, 2023, 2024 (68 races) |
| Held out | 2025 (24 races) |
| Excluded | 2021 and earlier, 2026 |
| Raw laps ingested | 74,601 |
| Laps in the degradation fit | 33,158 |

### Why these seasons

2022 is the first year of the ground-effect regulations. Cars before that had a
fundamentally different aerodynamic platform, a different tyre working range and
a much larger dirty-air penalty, so pooling 2021 with 2022 would be pooling two
different sports.

2026 is excluded for the same reason in the other direction: new power units and
active aerodynamics. Degradation, fuel effect and overtaking difficulty are all
discontinuous across that boundary. The code will happily ingest it — the
exclusion is a config choice in `conf/data/default.yaml`, not a hard-coded
restriction — but nothing in the results uses it.

2025 is held out completely. The degradation posterior, the per-circuit pit-loss
and neutralisation rates, and the overtaking index are all fitted on 2022–2024.

## Quirks, and what each one costs

The filters below are applied in `pitwall/ingest/clean.py` and every one is
counted, so the attrition table at the end of this file is generated rather than
asserted. Order matters: several filters depend on quantities that must be
computed while the field is still complete.

### In-laps and out-laps

A lap ending in the pit lane carries the pit-entry deceleration; the following
lap carries the pit-exit acceleration and a cold tyre. Neither is a measurement
of tyre pace, and both are 15–25 seconds slow.

FastF1 marks these directly with `PitInTime` and `PitOutTime`, so this is a
direct read rather than an inference from lap times. **Cost of ignoring it:** the
first and last lap of every stint become extreme outliers. Because they land at
tyre age 1 and at the maximum tyre age of the stint, a regression on tyre age
sees a hugely inflated slope at both ends.

Removed: 1,842 out-laps and 1,996 in-laps.

### Deleted laps

Track-limits deletions. The lap was physically driven and is fully timed, so it
looks clean, but the stewards' deletion is a reliable marker that the driver ran
off the road — which usually means they either gained time or lost it.

Removed: 795 laps.

### Laps FastF1 flags inaccurate

`IsAccurate` is FastF1's own integrity flag, set when timing is incomplete or a
pit event could not be resolved. This is the single largest technical filter and
it is worth trusting.

Removed: 2,439 laps.

### Safety car, VSC, yellow and red flags

Lap times under a neutralisation say nothing about tyres — the field is running
to a delta. Red flags are worse: the race stops, tyres cool completely and are
often changed for free under the stoppage, so `TyreLife` continues counting on a
tyre that has been sitting in a blanket.

FastF1 gives a per-lap `TrackStatus` which is the concatenation of every code
seen during that lap, so `"26"` means the lap saw both a yellow and a VSC. Only
pure `"1"` survives.

Codes: 1 green, 2 yellow, 4 safety car, 5 red flag, 6 VSC deployed, 7 VSC ending.

Removed: 1,399 laps. **Cost of ignoring it:** safety-car laps are 40% slower than
green laps. A handful of them in a stint dominates the fitted degradation slope
entirely.

### Traffic and dirty air

This is the largest filter by volume and the least obvious.

A car running within about 1.5s of the one ahead loses several tenths a lap to
dirty air. That loss is not tyre degradation, but it is strongly correlated with
tyre age, because a car that has been in a train for a whole stint is in the
train *because* it is slow, and its tyres are ageing the whole time.

FastF1 does not expose gaps directly, so `gap_ahead_s` is reconstructed from
`LapStartTime`: within one race and lap number, sorting drivers by when they
started the lap gives the running order, and adjacent differences give the gaps.

**This must be computed before any lap-level filtering.** Removing a driver's lap
first makes the next car back appear to be chasing whoever is now adjacent in the
sorted order, which silently corrupts every gap behind the removed car.

Removed: 18,004 laps — 34% of what remained at that point.

Filtering this hard has a purpose beyond cleanliness: it makes the degradation
model a **clean-air** model. The simulator adds its own traffic loss on top, so a
degradation curve that already contained dirty air would double-count it.

*Known gap:* lapped traffic. A leader starting a lap 2s behind a car a lap down
sees clear air in the timing but not on the road. Those laps are rare and are
partly caught by the stint-median filter.

### Rain, and the drying track that follows it

The obvious approach — drop any race with a rainfall reading — costs a third of
the calendar to brief showers and sensor blips.

The subtler problem is worse. A race that starts wet and dries out has a track
whose grip improves for the whole first half, long after the rainfall sensor stops
reading. That improvement is not something a per-race intercept can absorb,
because it varies *within* the race, so it loads straight onto whatever varies
within the race too: tyre age.

**This produced the worst single artefact in the project.** Before the filter
existed, Singapore 2022 fitted −0.70 s/lap of "degradation" — tyres getting
three quarters of a second per lap *faster*. And because the between-circuit
variance is shared, that one runaway circuit inflated the hierarchical spread and
switched off the shrinkage protecting every other circuit.

Two filters now, and the second is the one that matters:

1. Drop the race if more than 15% of laps have a rainfall reading.
2. Drop the race if wet or intermediate tyres ran for more than 5% of laps.
   Wet-tyre usage is a far more reliable signal of a compromised track than the
   rainfall sensor, because it reflects what the drivers could actually feel.

Removed: 9,293 laps from rain-share races and 2,077 from drying tracks.

### Tyre age

`TyreLife` counts laps on the specific set, and it correctly starts above 1 for a
scrubbed set that did laps in qualifying — which is what you want, since a
scrubbed tyre really has been used.

It is missing when the stint's start was not observed, usually a driver whose
timing feed dropped early. Removed: 61 laps.

### Red-flagged and shortened races

`completed_laps` is taken from the maximum lap actually run, not a scheduled
distance, because a red-flagged or timed-out race stops short. Fuel-load
estimates divide by this, so using a scheduled distance would mis-state fuel on
every lap of an interrupted race.

### Classification is not status

FastF1 3.8 reports a lapped finisher's status as `"Lapped"`, while older seasons
use `"+1 Lap"`. Keying "did this car finish" on the status string therefore marks
classified cars as retirements, and differently depending on the season.

Classification comes from `ClassifiedPosition`, which is numeric for a classified
finisher and a letter otherwise (R retired, D disqualified, W withdrawn, N not
classified). This was a real bug: Bahrain 2023 reported 11 finishers instead of
17 before it was fixed.

### Fuel load is an estimate, not a measurement

Fuel mass is never published. It is reconstructed as a linear burn-down from the
110 kg regulation maximum, which is close to the truth under green-flag running.

It is worst under a safety car, where the field burns far less fuel per lap than
the linear model assumes, so the estimate is a little low for the laps after a
long neutralisation. Second-order next to the confounding it exists to resolve,
but it is an approximation and not a measurement.

### Sprint weekends

Only race sessions are ingested. On a sprint weekend the tyre allocation and the
amount of long-run data differ, but the race itself is scored the same way, so
they are kept without special handling.

## Attrition

Generated by `pitwall clean`, written to
`data/processed/laps_<seasons>.attrition.csv`. Training seasons 2022–2024:

| stage | removed | remaining | % of raw | rationale |
|---|---:|---:|---:|---|
| raw laps | 0 | 74,601 | 100.0% | everything FastF1 returned |
| mostly-dry races | 9,293 | 65,308 | 87.5% | race-level rain share above 15% |
| no drying tracks | 2,077 | 63,231 | 84.8% | wet-tyre usage above 5% of race laps |
| dry laps | 119 | 63,112 | 84.6% | individual laps with a rainfall reading |
| slick compounds | 334 | 62,778 | 84.2% | dropped inters, wets and unknown |
| timed laps | 892 | 61,886 | 83.0% | LapTime is NaT when the feed dropped the lap |
| no out-laps | 1,842 | 60,044 | 80.5% | pit-exit deficit plus a cold tyre |
| no in-laps | 1,996 | 58,048 | 77.8% | pit-entry deficit, not tyre pace |
| no lap 1 | 998 | 57,050 | 76.5% | standing start and first-lap scrap |
| IsAccurate | 2,439 | 54,611 | 73.2% | FastF1's own timing-integrity flag |
| not deleted | 795 | 53,816 | 72.1% | track-limits deletion implies going off |
| green flag only | 1,399 | 52,417 | 70.3% | excludes SC, VSC, yellow and red |
| clean air only | 18,004 | 34,413 | 46.1% | started within 1.5s of the car ahead |
| known tyre age | 61 | 34,352 | 46.0% | TyreLife missing |
| traffic / mistakes | 43 | 34,309 | 46.0% | above 1.07x the stint median |
| usable stints | 1,151 | 33,158 | 44.4% | stints too short to identify a slope |

**33,158 laps survive, 44% of raw.** That is 2,133 stints across 25 circuits, 28
drivers and 58 races.

Losing more than half the data is the correct outcome. The discarded laps are not
noisy measurements of tyre pace, they are measurements of something else —
pit lane geometry, safety car deltas, dirty air. Keeping them would not add
information, it would add bias.

## Derived tables

`pitwall/ingest/circuits.py` estimates per-circuit constants from the raw tables
rather than a hardcoded reference, so adding a season updates them:

- **Pit loss**, from in-lap/out-lap pairs against the driver's own race median.
  Stops under a neutralisation are excluded, because the field is slowed and the
  relative loss is roughly halved — and mixing them biases the estimate down
  worst at exactly the circuits with the most safety cars. Bahrain estimates
  24.4s, Melbourne 22.4s, both close to published figures.
- **Safety car and VSC rates**, per circuit, per race.
- **Overtaking difficulty**, from position changes between consecutive green laps
  where neither lap was a pit lap.

All four are shrunk towards the pooled mean by an empirical-Bayes weight, because
four seasons gives at most four observations per circuit.

## Reproducing

```bash
pitwall ingest --seasons 2022,2023,2024,2025
```

Roughly an hour, mostly network-bound, and cached under `data/fastf1_cache/`
(several GB, gitignored). One session failing does not stop the run. Then:

```bash
pitwall clean --seasons 2022,2023,2024
```
