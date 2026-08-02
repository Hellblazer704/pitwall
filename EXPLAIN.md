# Explain

Part one is a walkthrough of every file and why it exists. Part two is the
fifteen hardest questions I can ask about this project, with honest answers —
including the ones where the answer is "that's a real weakness".

---

# Part 1: file by file

## Top level

**`conf/`** — Hydra config tree, split by concern: `data`, `degradation`,
`simulator`, `optimizer`. Split this way so an ablation is a config override
rather than a code edit, which is what makes Component 5 run the *same* code path
as the real optimiser with one switch flipped. Every non-obvious value carries a
comment saying where it came from.

**`src/pitwall/config.py`** — composes the Hydra tree explicitly rather than via
`@hydra.main`. Several entry points want the same config, and tests and notebooks
need a real config object without Hydra taking over argv or the working
directory.

**`src/pitwall/paths.py`** — filesystem layout, resolved once by walking up for
`pyproject.toml`. Everything else asks this rather than computing paths.

**`src/pitwall/rng.py`** — the seed discipline the reproducibility claim rests
on. One integer in the config; subsystems get independent streams derived from it
by *label* via `SeedSequence` spawning. Labels rather than sequential draws
matter: it means a subsystem's stream does not shift because another one was
constructed first, so work can be reordered or parallelised without changing
results. Uses `blake2b` rather than `hash()`, which is salted per process.

**`src/pitwall/cli.py`** — subcommand dispatch. Trailing arguments are Hydra
overrides. `--seasons` takes a comma-separated list rather than `nargs="+"`
because a greedy list argument swallows the positional overrides.

## Ingest — `src/pitwall/ingest/`

**`schema.py`** — column contracts as dicts of name → dtype, with `coerce()` and
`require_columns()`. FastF1 does rename columns between releases; when it does,
exactly one file needs editing and the failure is loud rather than a `KeyError`
three layers into a Gibbs sampler.

**`fetch.py`** — FastF1 session → four flat parquet tables per race: laps, race
metadata, results, neutralisations. Deliberately **lossless**: every lap is
written with its flags intact, including in-laps, safety-car laps and the ones
marked inaccurate. Deciding what counts as a bad lap is a modelling choice, and
re-downloading four seasons costs an hour, so that decision does not belong here.

Neutralisations are derived from per-lap `TrackStatus` strings with a 50% quorum
across drivers, because an individual car's feed can miss a code entirely if it
pits across the boundary.

**`clean.py`** — raw laps → the modelling frame, applying every filter and
counting each one so the attrition is auditable rather than implicit. Two derived
quantities matter downstream: estimated fuel mass, and `gap_ahead_s`
reconstructed from lap start timestamps.

Order is load-bearing. `gap_ahead_s` is computed **before** any lap-level
filtering, because it is a difference between adjacent cars on the road and
removing a driver's lap first makes the next car back appear to be chasing
whoever is now adjacent in the sorted order.

**`circuits.py`** — per-circuit constants estimated from the races themselves
rather than a hardcoded table: pit loss from in-lap/out-lap pairs, neutralisation
rates, overtaking difficulty from position changes on green laps. All shrunk
toward the pooled mean by an empirical-Bayes weight, because four seasons gives
at most four observations per venue.

## Component 1 — `src/pitwall/degradation/`

**`design.py`** — the design matrices, and the longest docstring in the project,
because the identification argument is the actual intellectual content. Read this
one first. It states what breaks the fuel/wear collinearity, and what remains
confounded and why.

Also: age is centred before fitting (the raw `[1, a, a²]` basis correlates at
~0.97 and the sampler crawls), the medium compound is pinned as the offset
reference, and `max_age` records the oldest tyre age observed per circuit and
compound so predictions can be bounded.

**`gibbs.py`** — the sampler. Every full conditional is closed form, including
the half-Cauchy scale priors via the inverse-gamma scale mixture. Seven blocks
per sweep. The fuel coefficient and per-circuit track evolution are drawn as one
joint block because they are exactly collinear within a race; in separate blocks
the chain random-walks along the ridge (measured: Rhat 2.24, ESS 19).

Per-cell coefficients are solved as a batched linear algebra call grouped by
active-column pattern, rather than a Python loop over ~75 cells per sweep.

**`diagnostics.py`** — split-Rhat and effective sample size, implemented here
rather than imported from ArviZ. The sampler is hand-rolled; a diagnostic you
cannot read is not much of a check. Split-Rhat catches a chain drifting steadily
that unsplit Rhat is blind to.

**`model.py`** — `DegradationPosterior`, the boundary between Components 1 and 2.
The simulator never sees a point estimate: it draws a whole parameter vector per
simulated race, so an ensemble carries model uncertainty as well as race-to-race
noise. One draw per *race*, not per car, which preserves the posterior
correlations between compounds — treating them as independent would make a
compound switch look less risky than it is.

**`fit.py`** — orchestration, and writes a model card with convergence
diagnostics, the fuel estimate, degradation by compound, the collinearity
diagnostic and the driver ranking.

## Component 2 — `src/pitwall/sim/`

**`strategy.py`** — `Strategy` (compounds + stop laps) and expansion to per-lap
arrays. Enforces the two-compound rule, which is real and is the easiest way to
produce a recommendation that would be disqualified.

**`events.py`** — safety car and VSC as an inhomogeneous Bernoulli process over
laps, with a lap-1 multiplier and exponential decay, normalised so the expected
count matches each circuit's observed rate. Documents what it deliberately does
*not* model: deployments are exogenous.

**`params.py`** — resolves config + circuit estimates + posterior into one frozen
`SimParams`, so the engine's inner loop has no config lookups and an ablation is
a matter of handing it a different object.

**`engine.py`** — the lap loop, vectorised over the **ensemble** rather than the
field. This is the single most consequential engineering decision in the project;
the module docstring explains it. Traffic resolution is the one genuinely
sequential part, and the field is gathered into track order once per lap rather
than fancy-indexed per car — worth about 4x.

**`field.py`** — reconstructs a real race's grid, per-car pace and actual
strategies. Pace is not a raw median lap time: the fitted degradation and fuel
terms are subtracted first, because a raw median depends on how long the driver
ran on which compound with how much fuel.

## Component 3 — `src/pitwall/optimize/`

**`candidates.py`** — enumeration, staged. `compound_shortlist` ranks sequences
at an even stint split; `enumerate_candidates` then searches pit laps for the
survivors on a grid that coarsens with stop count; `refine_around` does the final
lap-by-lap pass.

**`mc.py`** — scoring. Common random numbers, the championship points table, and
`StrategyEvaluation` carrying the full position distribution plus a Monte Carlo
standard error so a reported difference can be judged against the noise in the
estimate.

**`reactive.py`** — the online policy. Advances the ensemble in segments, branches
at each deployment, evaluates every branch to the flag. Replicates the facing
races up to the decision budget first, because only a few dozen races out of a
few thousand see a deployment on any one lap.

**`run.py`** — wires the three stages together and renders the report.

## Components 4 and 5

**`validate/backtest.py`** — replays held-out races with the real strategies,
scores position error, rank correlation and probability calibration against a
climatology baseline. `strategy_comparison` compares recommendations to what
teams ran, flagged clearly as a comparison scored inside the model.

**`ablation/study.py`** — `ABLATIONS` maps a name to `SimParams` overrides. Each
arm searches with its own degraded simulator, and the chosen strategy is scored
under the **calibrated** one. That difference is regret, and it is the number
that means something.

## Tests

78 tests, no network. Built around **parameter recovery**: `test_degradation.py`
generates laps from the model with known coefficients and asserts the sampler
returns them. If the identification argument is wrong, that test fails and
nothing else needs to be believed.

`test_engine.py` is mostly invariants — positions are a permutation, retirements
classify behind finishers, a car that cannot overtake stays behind one that can
be passed freely — because a race simulator has few quantities with a known right
answer but plenty of properties that must hold.

---

# Part 2: the fifteen hardest questions

### 1. Fuel and tyre age are perfectly collinear within a stint. How can you possibly separate them?

Within a stint you cannot — the design matrix is singular. The separation comes
from *across* stints. A pit stop resets tyre age to zero but does not put fuel
back in the car, so lap 2 of the race and lap 2 of a stint that began on lap 30
have the same tyre and about 50 kg different fuel. That contrast identifies the
fuel coefficient directly.

Two more things help: stint plans differ across the field, so two cars at the
same circuit on the same lap can be 25 laps apart in tyre age; and the fuel
coefficient is pooled globally while wear is free per circuit, so one number is
estimated from every lap in the dataset.

The test that matters is recovery on synthetic data: generate laps with fuel at
0.030 s/kg and wear slopes of 0.60/1.00/1.50, and the sampler returns 0.0299 and
0.592/0.995/1.505.

### 2. Isn't your fuel coefficient really picking up track evolution?

Partly, yes, and I would not claim otherwise. Both are linear in laps completed
within a race, so they are exactly collinear there.

What the model does is add a per-circuit track-evolution term and separate the
two by pooling structure rather than by within-race data: the fuel coefficient is
one number for the entire dataset carrying a physical prior, while evolution is
free per circuit. What is genuinely identified is the split between the
field-wide effect and the circuit-specific one, not fuel and evolution as
separate physical quantities. A circuit whose true evolution matches the field
average will have part of it absorbed into the fuel term.

I left this out of the first version, and it bit. Compounds are not used at
random points in a race — hards run long and late — so the unmodelled progress
effect was attributed to whichever compound ran then, and the hard tyre came out
0.59 s/lap *faster* than the medium at Melbourne.

### 3. Why hand-write a Gibbs sampler instead of using PyMC or Stan?

The model is a Gaussian hierarchical linear model. Every full conditional is
available in closed form, including the half-Cauchy scale priors via the standard
inverse-gamma scale mixture. There is genuinely nothing for a gradient-based
sampler to buy: no tuning, no warmup adaptation, no divergences to diagnose, and
no compiler dependency, which matters for a project meant to run anywhere from
one `pip install`.

The honest cost is that Gibbs mixes more slowly than HMC when blocks are
correlated, and this model has a real ridge. I paid for that in draws — 16,000
per chain rather than 2,000 — and in one structural fix, blocking the collinear
parameters together. With PyMC I would have got the same answer with fewer draws
and a heavier dependency. It is a defensible trade either way; what would not be
defensible is not knowing which one I was making.

### 4. Your model recommends three-stop soft-tyre strategies at some circuits. Teams do not. Who is wrong?

Mostly the model, and I know the mechanism.

Softs are run in short stints, so the training data contains them almost
exclusively at low tyre age — 3,428 soft laps against 18,694 hard, and Melbourne
has *five* soft laps across three seasons. But the deeper issue is selection:
teams choose softs for short stints precisely *because* they degrade, so the
observed soft laps come disproportionately from situations where the tyre was
working well. The fitted soft curve is therefore optimistic, and the optimiser
believes softs last better than they do.

The sharpest version of this: at Monza 2025 **all twenty cars one-stopped** and
the model recommends a two-stop. That is not a subtle disagreement.

Digging into it found something worse than wide intervals. The unconstrained fit
had the Monza soft **0.16s a lap quicker at ten laps old than new** — negative
wear, which makes short stints look free. Wear is now forced to be
non-decreasing in tyre age, which is a physical fact the likelihood does not
know. That cut the two-stop's margin from 0.39 expected points to 0.23 and
slightly improved held-out accuracy, but did not flip the call.

What is left is honest uncertainty about *which* of two things is still wrong:
residual soft-tyre optimism, or Monza's overtaking index of 1.20 making it too
cheap to recover track position after the extra stop. I have not separated them.

The hierarchy handles the sparsity honestly — sparse circuits get wide intervals
and shrink to the population. It does not handle the selection, because
correcting that needs a model of the compound choice itself. That is the first
thing I would fix, and it is in DESIGN.md as a measured failure with numbers
attached rather than a hedge.

### 5. Why is the simulator vectorised over races rather than over cars?

Because the alternative does not fit in the time budget. The natural structure is
a loop over laps containing a loop over cars, called 10,000 times per candidate:
roughly 14 million Python iterations for one strategy, which puts a single
optimiser run into the tens of minutes and makes the reactive policy impossible.

With `(n_races, n_cars)` state, the lap loop is Python but the work inside is
numpy across all races at once. A 12,000-race ensemble over 57 laps is about
1,100 vectorised steps rather than 13.7 million scalar ones, and runs at 0.18 ms
per simulated race.

The one part that stays sequential is traffic resolution, because whether a car
is held up depends on what the car ahead just did. That is `n_cars` iterations of
numpy over `n_races`, and hoisting the track-order gather out of it was worth
about 4x on its own.

### 6. What actually stops a faster car from just driving through a slower one?

An explicit constraint. Each lap, walking the field from the leader backwards, a
car's unimpeded finishing time is compared against the car ahead's. If it would
end up in front, it must either complete an overtake — probability from a
logistic in pace delta, scaled by the circuit's difficulty index — or it is held
to a 0.35s minimum following gap.

Without this you have a spreadsheet. Summing lap times and sorting hands every
faster car a free pass, which systematically overvalues any strategy that gives
up track position, which is to say most aggressive strategies. It is one of the
ablation arms for exactly that reason.

### 7. How do you know your overtaking model is right?

It is fitted, from 16,004 real attack events: a car that began a green lap
within 1.2s of the car directly ahead on the road, and whether it got past by
the end of the next lap. Penalised logistic regression gives an intercept of
−1.887, a slope of 1.463 per s/lap of pace advantage, and per-circuit offsets.

It was *not* fitted originally — it was hand-calibrated — and finding out how
wrong that was is the most useful thing I did to this project. Two failures:

The logistic **saturated**. The hand-set slope tracked the data well up to about
1 s/lap of advantage and then ran away towards certainty, where the observed
rate above 1 s/lap is 0.53. That is exactly the regime a car on brand new tyres
after an extra stop is in, so it carved through the field for free and an extra
pit stop stopped costing anything. That was the whole Monza bug.

The **circuit index measured the wrong quantity**. Passes per racing lap
conflates "hard to pass here" with "the field was spread out here". Conditional
on a real attack, Monaco's pass rate is 0.019 against Barcelona's 0.299 — a
factor of fifteen, where the old index had a factor of two. And a multiplier
cannot express near-impossibility at all: scaling an already-saturated
probability by 0.6 still leaves it near certain. The circuit now enters in logit
space.

What I would still not claim: the absolute rate for one specific move. Counting
net position gains undercounts a pass that is immediately re-passed, so this is
a lower bound on wheel-to-wheel activity.

### 8. Your safety cars are exogenous. Isn't that a serious problem?

Yes, and it is the limitation I would flag first about the simulator.

In reality a safety car is *caused* by an incident, and incidents correlate with
close racing, first-lap chaos, bunched restarts and rain. The model draws
deployments independently of the race state, so it cannot represent "a
deployment is more likely because these two are fighting". The lap-1 multiplier
is a crude stand-in for the largest single piece of that.

The direction of the error matters: because deployments cannot cluster with the
situations that cause them, the simulator probably *understates* the correlation
between a chaotic race and a strategy-scrambling neutralisation, which means it
understates the variance of aggressive strategies rather than overstating it.
Fixing it properly needs incident data with causes, which FastF1 does not provide
in usable form.

### 9. Why report distributions instead of expected values?

Because the expected value is frequently not the decision.

A strategy with a better mean finishing position can easily be the wrong call. A
team defending a championship lead cares about the left tail — the probability of
scoring nothing. A team needing a result cares about P(podium) and will accept a
worse mean to get it. Those two can point at different strategies from the same
ensemble, and an expected value collapses the very information that
distinguishes them.

It also matters for honesty about precision. Reporting "15.24 expected points"
with a Monte Carlo standard error of 0.13 makes clear that a rival candidate at
15.19 is not distinguishable, which a bare ranking hides.

### 10. What are common random numbers and why do you need them?

Every candidate strategy is evaluated against the *same* sampled races: the same
safety-car schedule, the same degradation draws, the same execution noise. Only
the strategy differs.

Without it you are comparing two noise draws. Candidates often differ by a few
hundredths of a point in expected points, while the standard deviation of a
single race outcome is several points; the difference of two independent
estimates is dominated by sampling noise unless the ensemble is enormous. Pairing
cancels the shared randomness and leaves the effect of the strategy.

It constrains the engine: the number of random draws must not depend on the
strategy. That is why pit-stop noise is drawn on every lap even when nobody
stops — guarding it behind `if pits.any()` silently desynchronised the streams
and made the whole comparison noise. `tests/test_optimize.py` asserts the paired
variance is lower than the unpaired.

### 11. How is the reactive policy evaluated without cheating?

This is the subtle one. The trap is letting each simulated race choose its own
branch after seeing how that race turns out, which is clairvoyance and would
flatter the policy enormously.

So the decision is made **once per deployment lap** and applied to every race
facing it. A strategist seeing a safety car on lap 23 makes one call for the
situation, not twelve thousand different calls informed by twelve thousand
futures.

The second issue is sample size. Only a few dozen races out of a few thousand see
a deployment on any given lap, and comparing branches on sixty samples is
comparing noise — the first version did exactly that and the decisions were
essentially random. The facing races are now replicated up to the decision
budget, each replica starting from the same real state and running forward under
its own draws. That is the right Monte Carlo: many futures from one actual
situation.

### 12. Your backtest MAE is 2.54 positions. Is that good?

It needs a baseline to mean anything, and there are two worth stating.

Predicting the grid order gives roughly 3.5 positions of mean absolute error on
this data, so the simulator adds real information. Spearman rank correlation of
0.788 against actual finishing order says the same. But the number that matters
more is calibration: Brier score of 0.124 on P(points) against a climatology
baseline of 0.250 — the model is twice as good as predicting the base rate for
everyone, and the reliability curve shows predicted-versus-observed gaps under 8
points in every bin.

An MAE of 2.54 also should not be read as "the model is 2.5 places wrong". Races
are genuinely stochastic; a perfect model of the process would still have
substantial position error because the process itself has substantial variance.
The right question is whether the *distribution* is honest, and the calibration
figures say it roughly is.

### 13. Where does the model fail worst, and do you know why?

Zandvoort at 5.12 mean absolute error and Melbourne at 4.29, against a 2.54
average.

Melbourne 2025 was wet. The dry-tyre degradation model cannot represent it at
all, and races like it are excluded from *training* but not from the backtest —
deliberately, because hiding them would flatter the result. Zandvoort was heavily
disrupted. Las Vegas at 4.35 is a different failure: it is a new circuit with two
seasons of data, so its per-circuit estimates are thin and shrunk hard toward the
population.

That the errors concentrate exactly where the stated assumptions break is the
reassuring failure mode. The worrying one would be large errors at a
well-sampled, dry, uneventful race, and there aren't any.

### 14. The ablation compares a naive model's choice against yours. Isn't that rigged?

It would be if I scored each model under itself, which is precisely why I do not.

Each degraded simulator picks its own best strategy. That strategy is then scored
under the **calibrated** simulator, and regret is the gap to what the calibrated
model would have chosen. A naive model always rates its own pick highly — that is
what being wrong looks like from the inside — so self-reported value is
uninformative.

The fair objection is that regret is still measured in my model's units, so it
assumes the calibrated model is right. It is not a claim about reality; it is a
claim about internal consistency: *given* that safety cars happen at the observed
rate and traffic costs what the data says, here is what ignoring them costs. The
external check is the backtest, and the two answer different questions.

The result also went against what I expected, which is some evidence it is not
rigged. The standard story is that naive simulators overrate *aggressive*
strategies. Only the traffic ablation did that. Safety cars and degradation
uncertainty are both risk channels that give real option value to holding a stop
in hand, so removing them made the conservative one-stopper look safe — mean
recommended stops fell from 1.67 to 1.33. If I had been fitting the experiment to
a conclusion, that is not the conclusion I would have fitted it to.

What does survive cleanly is the sharper claim: the naive model recommends a
different strategy 89% of the time, and rates its own pick at 12.56 points when
it is worth 10.38. Being wrong is survivable; being wrong while reporting two
points of value that are not there is what actually costs you a race.

### 15. If you had another week, what would you do?

Four things, in order.

First, the soft-tyre selection bias — a model of compound choice, so the wear
curves are not conditioned on the situations teams chose to use each tyre in.
That is the largest identified bias in the project and it drives visibly wrong
recommendations.

Second, fit the SC/VSC pit-loss discount from paired stops instead of assuming
0.42 and 0.58. Those two numbers set the value of the reactive policy's central
decision and they are currently the least grounded inputs in the simulator.

Third, better blocking in the sampler. The remaining ridge is between the
track-evolution terms and the compound offsets; sampling each circuit's evolution
jointly with its compound coefficients would cut the draw count several-fold.

Fourth, per-team reliability. Treating every car as equally fragile understates
the variance of a team's two-car outcome, which matters for constructors'
championship questions even though it barely moves a single car's strategy.

What I would *not* spend the week on is more circuits or more seasons. The
regulation boundaries make older data actively misleading, and the binding
constraint is model structure, not sample size.
