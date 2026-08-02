# pitwall

A Formula 1 race strategy simulator and Monte Carlo optimiser, built on real
timing data.

Given a grid slot, a car's pace and a tyre allocation, it searches over stint
plans — how many stops, on which laps, on which compounds — and returns the full
distribution of finishing positions for each, rather than an expected value. It
also runs the decision teams actually face: when a safety car deploys on lap 23,
do you box?

Everything is fitted from four seasons of FastF1 data. Nothing is a magic number
unless the code says so and DESIGN.md explains why.

---

## The short version

**Tyre degradation and fuel burn are confounded in raw lap times**, and
separating them is the whole game. On lap 12 of a stint the tyre is 12 laps old
and the car is 12 laps lighter, always. Regress lap time on tyre age within a
stint and you estimate *wear minus fuel burn* and call it wear — understating
real degradation by about 0.06 s per lap, which over a 30-lap stint is nearly two
seconds of error sitting directly on top of the stint-length decision. And it is
signed the wrong way: it makes long stints look cheaper than they are.

pitwall fits a hierarchical Bayesian model that separates them, propagates the
posterior uncertainty into a lap-by-lap race simulator, and searches strategies
by Monte Carlo.

Then it checks itself. The model is fitted on 2022–2024 and validated on a 2025
season it has never seen:

| | |
|---|---:|
| Mean absolute finishing-position error | **2.45** |
| Spearman rank correlation | **0.806** |
| Brier score, P(points) | **0.1179** |
| ...against a climatology baseline of | 0.250 |

And it quantifies what realism is worth: Component 5 strips out safety cars,
degradation uncertainty, traffic and pit-stop variance one at a time, and
measures what the resulting recommendations cost when scored under the calibrated
model.

---

## Install

```bash
pip install -e ".[dev]"
```

Python 3.11+. The only heavy dependency is FastF1; the Bayesian sampler is
hand-written numpy, so there is no compiler toolchain to set up.

## Run it

```bash
pitwall ingest --seasons 2022,2023,2024,2025
```

Downloads and flattens every race session to parquet. About an hour, mostly
network-bound, cached under `data/fastf1_cache/`. A failed session is logged and
skipped rather than aborting the run.

```bash
pitwall clean --seasons 2022,2023,2024
```

Builds the modelling frame and prints the attrition table — every filter, how
many laps it removed, and why. 74,601 raw laps become 33,158.

```bash
pitwall fit --seasons 2022,2023,2024
```

Fits the degradation model. Four chains, about ten minutes, and it prints
convergence diagnostics and refuses to pretend a bad fit is a good one.

```bash
pitwall optimize --race 2025:14 --driver NOR
```

Searches strategies for one car in one race and prints ranked candidates, the
finishing-position distribution and the reactive policy's decisions.

```bash
pitwall backtest --seasons 2025
pitwall ablate --race 2025:4,2025:11,2025:16
```

Every command takes Hydra overrides, so an ablation is a config change rather
than a code change:

```bash
pitwall optimize --race 2025:14 simulator.safety_car.enabled=false seed=7
```

## Reproducibility

One integer in `conf/config.yaml` determines every result. Subsystems draw from
independent streams derived from it by label, using `numpy.random.SeedSequence`
spawning, so the streams do not depend on the order work is done in and the Monte
Carlo can be reordered without changing outputs. Two runs with the same seed are
bit-identical; `tests/test_rng.py` and `tests/test_engine.py` assert it.

---

## What is in here

### Component 1 — hierarchical Bayesian degradation

`src/pitwall/degradation/`

Lap time decomposes as:

```
t = race-driver intercept          nuisance: car, driver, circuit, that day
  + fuel coefficient × fuel mass   pooled globally, physical prior
  + track evolution                per circuit
  + compound offset + wear         per circuit and compound, partially pooled
  + driver wear adjustment         per driver
  + noise
```

The race-driver intercept absorbs everything constant across one driver's race,
leaving exactly the within-race contrasts that identify fuel: stints reset tyre
age while fuel keeps falling, so lap 2 of the race and lap 2 of a stint starting
on lap 30 have the same tyre and 50 kg different fuel.

Sampling is a hand-written Gibbs sampler. Every full conditional is closed form,
including the half-Cauchy scale priors via the inverse-gamma mixture, so there is
nothing for a gradient sampler to buy and no compiler dependency. Diagnostics are
split-Rhat and effective sample size, computed here rather than imported.

Fitted result: **0.0309 s/lap/kg**, 90% CI 0.0226–0.0390, which is where the
published figures sit. Wear at 15 laps of age: hard 0.73s, medium 1.26s, soft
1.28s, correctly ordered.

The claim is validated by parameter recovery on synthetic data, not by
plausibility — see `tests/test_degradation.py`.

### Component 2 — race simulator

`src/pitwall/sim/`

Lap-by-lap discrete-event model: degradation sampled from Component 1's
posterior, per-circuit pit loss, dirty air, overtaking as a function of pace
delta and circuit, safety car and VSC as a fitted hazard process, and reliability
retirements.

The structural decision is that the **ensemble** is the vectorisation axis, not
the field. State is `(n_races, n_cars)` and one lap advances every race at once.
Written the obvious way — loop over cars inside loop over laps, called 10,000
times — a single candidate strategy costs about 14 million Python iterations. This
way it is ~1,100 vectorised steps and runs at 0.18 ms per simulated race.

A car cannot drive through the car in front. If its unimpeded lap would put it
ahead, it must complete an overtake or sit at a minimum following gap. That
constraint is what separates this from a spreadsheet, and Component 5 measures
what it is worth.

### Component 3 — strategy optimiser

`src/pitwall/optimize/`

Three-stage search over stop count, pit laps and compound sequence, turning a
40,000-candidate product into a few hundred. Every candidate is scored against
the *same* sampled races — same safety cars, same tyre behaviour, same execution
noise — so the difference between two candidates reflects the strategies rather
than Monte Carlo noise.

Output is the full finishing-position distribution:

```
Best: 2stop M-H-H @18,38
  expected points 15.24 (MC se 0.13), mean finish 4.10
  P(win) 0.21  P(podium) 0.58  P(points) 0.94

  P1   0.213 ###########################################
  P2   0.198 ########################################
  P3   0.167 #################################
  P4   0.121 ########################
  ...
```

The reactive policy re-optimises mid-race. On a deployment it branches the
affected races, evaluates staying out against boxing onto each compound all the
way to the flag, and takes the better one if it clears a threshold. The decision
is made once per deployment lap for the whole situation, not per sampled race —
letting each race pick its own branch would be using knowledge of how that race
turns out.

### Component 4 — validation

`src/pitwall/validate/`

Backtest on held-out 2025. Replaying the strategies teams actually ran isolates
the race model, since the strategy input is the truth. Reports position error,
rank correlation and a reliability curve, because a simulator can have a good
mean finishing position and still be badly overconfident.

Where it fails is as informative as where it works: the worst-predicted races are
Zandvoort and Melbourne, both heavily disrupted, and Melbourne 2025 was wet —
which a dry-tyre model cannot represent at all. The errors concentrate where the
stated assumptions break.

### Component 5 — the ablation

`src/pitwall/ablation/`

Removes realism one feature at a time and measures **regret**: the degraded
model's recommendation, scored under the calibrated model, against what the
calibrated model would have chosen. A naive model always rates its own choice
highly, so self-reported value is not the question — the question is what that
choice is actually worth.

| ablation | regret (pts) | overconfidence | strategy differs |
|---|---:|---:|---:|
| everything removed | **0.366** | **+2.19** | 89% |
| deterministic degradation | 0.194 | +0.54 | 89% |
| no safety car | 0.095 | −0.02 | 67% |
| no traffic | 0.086 | +0.18 | 56% |

The naive simulator rates its own pick at 12.56 expected points when it is
actually worth 10.38. It is not just wrong, it is **overconfident by more than
two points** — larger than the gap between most candidate strategies.

One finding went against expectation and is reported as such: the common claim
is that naive simulators overrate *aggressive* strategies, and that is not what
happened. Only the traffic ablation pushes that way. Safety cars and degradation
uncertainty are both risk channels that create option value in keeping a stop in
hand, so removing them makes the *conservative* one-stopper look safe — mean
recommended stops fall from 1.67 to 1.33. Full table and the sample-size caveat
in DESIGN.md.

---

## Documentation

- **DATA.md** — every FastF1 quirk, what it does to a naive fit, and the full
  attrition table
- **DESIGN.md** — every simulator assumption with the data behind it, marked
  estimated / calibrated / assumed, and a sensitivity ranking
- **EXPLAIN.md** — file-by-file walkthrough, plus the fifteen hardest questions
  about this project with honest answers

## Things that are wrong with it

Kept here rather than buried, and expanded in DESIGN.md:

- **The optimiser is wrong at Monza, measurably.** All twenty cars one-stopped
  there in 2025; the model recommends a two-stop. The cause is soft-tyre
  selection bias — teams run softs short *because* they degrade, so the observed
  soft laps are the ones where the tyre was working, and the fitted curve is
  optimistic. Melbourne has five soft laps in three seasons. Forcing wear to be
  non-decreasing in tyre age (the raw fit had a soft *gaining* 0.16 s/lap over
  its first ten laps) cut the two-stop's margin by ~40% but did not flip it.
  Details and the remaining suspects in DESIGN.md.
- **Safety cars are exogenous.** They are caused by incidents in reality, and
  incidents correlate with close racing. The simulator cannot represent "a
  deployment is more likely because the field is bunched".
- **Fuel burn and track evolution are not separately identified**, only split by
  their pooling structure. What is identified is the common effect versus the
  circuit-specific one.
- **Wet races are excluded entirely**, not modelled.
- **The SC/VSC pit-loss discount is assumed, not fitted**, and it is one of the
  most consequential numbers in the model.

## Layout

```
conf/                 Hydra config tree, split by concern
src/pitwall/
  ingest/             FastF1 → parquet, cleaning, per-circuit estimates
  degradation/        Component 1: design, Gibbs sampler, diagnostics, posterior
  sim/                Component 2: engine, events, params, field reconstruction
  optimize/           Component 3: candidates, Monte Carlo, reactive policy
  validate/           Component 4: backtest
  ablation/           Component 5: ablation study
tests/                78 tests, no network required
```

MIT licensed.
