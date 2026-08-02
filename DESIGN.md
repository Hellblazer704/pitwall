# Design

Every behavioural assumption in the simulator, what it was fitted or calibrated
from, and what happens if it is wrong. This is the document to argue with.

The convention throughout: **estimated** means fitted from the ingested data,
**calibrated** means chosen to match a published or observable quantity, and
**assumed** means neither, in which case the ablation in Component 5 or the
sensitivity note says how much it matters.

---

## 1. Degradation

### The fuel/wear identification

This is the hard part of the project, so it gets the most space.

Inside a single stint, tyre age and laps completed are the same number. On lap
12 of a stint the tyre is 12 laps old and the car is 12 laps lighter, always.
A regression of lap time on tyre age within a stint therefore estimates

> wear − fuel burn

and reports it as wear. `tests/test_degradation.py` asserts this explicitly on
synthetic data: the naive slope comes out at exactly `true_slope − fuel_per_lap`.

**What breaks the collinearity**, in decreasing order of how much work it does:

1. **Stints reset tyre age but not fuel.** Lap 2 of the race and lap 2 of a
   stint beginning on lap 30 have the same tyre and ~50 kg different fuel.
   Comparing equal-age laps at different fuel loads identifies the fuel
   coefficient directly.
2. **Stint plans differ across the field.** Two cars at the same circuit on the
   same lap can be 25 laps apart in tyre age.
3. **The fuel coefficient is pooled globally.** One number for every circuit and
   season, scaled per circuit by median lap time, while wear is free per circuit
   and compound. Pooling what is physically common and freeing what genuinely
   varies is what makes the separation stable rather than merely possible.

**Estimated:** 0.0309 s/lap/kg (90% CI 0.0226–0.0390), equivalently 0.060 s/lap
of burn-off. Published figures sit around 0.03 s/kg, so this lands where it
should.

**Validated by recovery**, not by plausibility. `tests/test_degradation.py`
generates laps from the model with known coefficients and checks the sampler
returns them: fuel 0.030 → 0.0299, wear slopes 0.60/1.00/1.50 →
0.592/0.995/1.505, residual sd 0.20 → 0.2022.

### Track evolution

**Assumed structure, estimated magnitude.** A per-circuit coefficient on race
fraction.

The first version of the model left this out, reasoning that a field-wide effect
would cancel out of between-car contrasts. That was wrong in an instructive way.
Compounds are *not* used at random points in a race — hards run the long middle
and late stints, softs run early and short — so "late in the race" and "on the
hard tyre" are strongly correlated, and any race-progress effect the model
cannot represent is attributed to whichever compound happens to run then.

The symptom was a hard tyre fitted 0.59 s/lap faster than the medium at
Melbourne. Downstream, the optimiser began recommending three-stop soft-tyre
strategies.

Race fraction and fuel mass are *exactly* collinear within one race, both being
affine in laps completed. They are separated only by their pooling structure:
`phi` is one number for the entire dataset with an informative physical prior,
`theta_c` is free per circuit. What is identified is the split between the common
effect and the circuit-specific one, **not** fuel and evolution as separate
physical quantities. A circuit whose true evolution matches the field average
will have part of it absorbed into `phi`.

Because they are collinear they are drawn as a single joint block. In separate
Gibbs blocks the chain random-walks along the ridge: measured Rhat 2.24, ESS 19.

**Sanity check the model passes convincingly.** Monaco comes out at −5.03s across
a race distance (90% CI −5.89 to −4.19), an order of magnitude more evolution
than anywhere else; every other circuit sits between −0.9s and −0.1s with
intervals spanning zero. Monaco is the one circuit on the calendar famous for
rubbering in dramatically — a dusty street track that starts green — and the
model found that from lap times alone, with no circuit-type feature to lean on.

### Curve shape

**Assumed:** quadratic in tyre age, with age centred before fitting.

Centring is not cosmetic. With an all-positive regressor the linear and
quadratic columns correlate at about 0.97 and the sampler crawls along the
resulting ridge — ESS in the twenties.

The quadratic carries a much tighter hierarchical prior (half-Cauchy scale 0.15
against 0.50 for the linear term) because it is the weakest-identified
coefficient in the model, and because the scale is shared across circuits a
single runaway cell inflates it for everyone and switches off the shrinkage that
was supposed to catch it.

Beyond the oldest tyre age actually observed for a circuit and compound, the
curve is continued **linearly** at the slope it had reached. A quadratic fitted
to stints that never passed 12 laps says nothing usable about lap 35.

### Compound offsets

**Estimated**, with the medium pinned as reference. Without a reference the
offsets and the race-driver intercepts share a flat direction that mixes badly.

Note the offset is pace at the *mean* tyre age, not on a new tyre, because age
is centred. At mean age a hard genuinely can be quicker than a medium, since the
medium has worn more by then. `compound_offsets()` reports both.

### Driver effects

**Estimated:** a per-driver adjustment to the wear slope, hierarchically pooled.
Only a handful of drivers have intervals excluding zero, which is the honest
result — most of the field manages tyres comparably and the model says so.

### Known limitation: soft-tyre curves are weak

Softs are used for short stints, so the data has them almost exclusively at low
tyre age — 3,428 laps against 18,694 for the hard, and the 95th percentile of
soft tyre age is 24 laps against 38 for the hard. At some circuits it is far
worse: Melbourne has **five** soft laps in three seasons.

Two consequences, both real:

- Per-circuit soft curves are dominated by the pooled population, and the
  reported intervals are correspondingly wide. That is the hierarchy working.
- There is a **selection effect the model does not correct**. Teams choose softs
  for short stints *because* they degrade, so the observed soft laps are
  disproportionately from situations where the tyre was working well. The fitted
  soft curve is therefore optimistic, and the optimiser over-favours soft-heavy
  multi-stop strategies at circuits with sparse soft data.

Correcting this needs a selection model on the compound choice itself, which is
out of scope here. It is the first thing I would fix.

---

## 2. Simulator

### Pit loss

**Estimated per circuit** from in-lap/out-lap pairs against the driver's own
race median: `loss = (t_in + t_out) − 2 × baseline`.

Stops under a neutralisation are excluded. Sanity: Monaco 25.3s (long narrow pit
lane), Spa 21.6s (short), Melbourne 22.4s, Bahrain 24.4s. All within a second or
so of published pit-loss figures.

Tyre warm-up is deliberately *not* modelled separately, because this estimate is
measured across the in-lap and out-lap pair and therefore already contains it.

### Stop execution

**Estimated per circuit**, with a calibrated floor. Stop-to-stop variability uses
the empirical spread of measured pit losses at that venue — 3.8s at Bahrain, for
instance — falling back to a configured 0.35s execution sd where the circuit
estimate is tighter than that. The empirical number is the better quantity: it
carries pit-lane traffic and in/out-lap execution as well as the crew.

On top of that, a 2% chance of a botched stop costing an extra ~6s on an
exponential tail. Fat-tailed rather than Gaussian on purpose — a slow wheel gun
or an unsafe release is not a mild perturbation of a normal stop, and the tail is
what makes each extra stop carry genuine execution risk.

**Known double-count:** the empirical spread is measured over stops that include
some botched ones, so adding an explicit botch tail counts part of that twice.
The effect is small and its sign is the safe one — the pit model is slightly
over-dispersed, which makes extra stops look riskier rather than free. Removing
it properly means estimating the botch rate and the clean-stop spread separately
from the same stop distribution, which is worth doing and is not done here.

### Neutralisation discount

**Assumed:** a stop under safety car costs 0.42× the green-flag loss, under VSC
0.58×.

These come from the physics — the field is circulating at roughly 60–70% of
racing speed, so the pit lane transit costs proportionally less relative to
rivals — rather than from a direct fit. They are the least well-grounded numbers
in the simulator and they matter a great deal, because they set the value of the
reactive policy's central decision. Worth fitting properly from paired stops.

### Neutralisation frequency

**Estimated per circuit**, as an inhomogeneous Bernoulli process over laps:

```
h_c(l) ∝  m₁ if l == 1 else exp(-k · l / L)
```

normalised so the expected count matches the circuit's observed rate. Lap 1 gets
its own multiplier (standing start, full field in close company); hazard decays
through the race because early incidents dominate.

Sanity: Melbourne and São Paulo highest at 1.19 safety cars per race, Barcelona
and Yas Marina lowest at 0.36. That matches the character of those circuits.

**Not modelled: endogeneity.** A safety car is *caused* by an incident, and
incidents correlate with close racing and first-lap chaos. Deployments here are
independent of the race state, so the simulator cannot represent "a safety car is
more likely because the field is bunched". The lap-1 multiplier is a crude
stand-in for the largest part of that.

### Safety car bunching

**Calibrated:** field compressed to 1.1s gaps behind the leader.

This is the mechanism that makes a deployment worth so much: a stop under it
loses the pit-lane time but not the gap to the field, and anyone who has already
stopped watches their lead evaporate. Getting it right matters more to a strategy
recommendation than almost anything else in the simulator.

### Traffic and dirty air

**Assumed shape, calibrated magnitude:** up to 0.45 s/lap, decaying linearly to
zero at a 1.6s gap, plus a 0.30s one-off for emerging from the pits into a pack.

The real relationship is closer to inverse-square in gap, but the linear form is
within the noise of what lap data can resolve and has one fewer parameter to
justify.

The magnitude is consistent with the ~0.3–0.5 s/lap that the degradation model's
own clean-air filter removes: the 18,004 laps dropped for running within 1.5s
are, on average, slower by about this much than clean-air laps at the same tyre
age.

### Position resolution

**Structural.** A car whose unimpeded lap would put it ahead of the car in front
must either complete an overtake or be held to a 0.35s minimum following gap.

This is what separates the model from a spreadsheet. Summing lap times and
sorting lets every faster car through for free, which systematically overvalues
any strategy that gives up track position. It is the `no_traffic` ablation.

### Overtaking

**Estimated per circuit, assumed functional form.**

`P(pass | lap) = logistic(a + b · pace_delta) × difficulty_c`, where
`difficulty_c` is the circuit's on-track pass rate relative to the median,
estimated from position changes between consecutive green laps where neither lap
was a pit lap, then clipped to [0.15, 3.0].

Sanity: Monaco 0.62 (hardest), Marina Bay 0.74, Montréal 0.76; Las Vegas 1.44,
Spa 1.38, Barcelona 1.21. Exactly the right ordering.

The intercept and pace coefficient are calibrated rather than jointly fitted,
because a proper fit needs wheel-to-wheel event data this project does not have.
Counting net position gains undercounts a pass that is immediately re-passed,
which is why the result is used as a relative index across circuits, not an
absolute probability.

### Reliability

**Calibrated:** 0.075% mechanical and 0.06% incident hazard per lap, plus a 1.2%
lap-1 incident probability. Produces an 8.4% retirement rate against a real-world
figure around 7–10%.

Not modelled: reliability differs enormously between teams, and correlates within
a team (same power unit). Treating it as identical across the field understates
the variance of a team's two-car outcome.

### Lapped traffic

**Not modelled as distinct.** Cars are ordered by cumulative time, so a car a lap
down sorts behind the lead-lap field and the following-gap constraint applies to
it as if it were racing. Real lapped cars yield under blue flags, so the
simulator slightly overstates what leaders lose to backmarkers.

---

## 3. Optimiser

### Search structure

**Assumed separable.** The full product of stop count × pit laps × compound
sequence is over 40,000 candidates on a 57-lap race. The search stages it: rank
compound sequences at an even stint split, give the survivors a pit-lap search,
refine the best lap by lap.

This assumes which compounds to run and when to stop are close to separable.
They interact only weakly — the interaction is second order next to each effect
alone — but a compound sequence that is only good at an unusual stint split can
be screened out.

The lap grid also coarsens in proportion to stop count, since combinations grow
as `C(grid_points, n_stops)`.

### Common random numbers

**Structural.** Every candidate is scored against the same sampled races: the
same safety-car schedule, the same degradation draws, the same execution noise.
Without it, comparing candidates that differ by a few hundredths of a point
across 12,000 noisy races is comparing two noise draws.

This constrains the engine: it must draw the same number of random values in the
same order regardless of the strategy being run, which is why pit noise is drawn
unconditionally even on laps where nobody stops.

`tests/test_optimize.py` asserts that paired variance is lower than unpaired.

### Objective

**Assumed:** expected championship points, which is what teams maximise and is
strongly non-linear in position — P1 to P2 is worth more than P6 to P10.

The full finishing-position distribution is always reported, because a better
mean finish is regularly the wrong call: a team fighting a championship cares
about the left tail and a team needing a result cares about P(podium), and the
two can point in opposite directions.

### Reactive policy

**Structural.** On a deployment, branch the affected races and evaluate staying
out against boxing onto each compound, all the way to the flag.

Two choices worth defending:

- The decision is made **once per deployment lap** and applied to every race
  facing it, not per sampled race. Letting each sampled race pick its own branch
  uses knowledge of how that race turns out, which is clairvoyance and would
  flatter the policy badly.
- Races facing a decision are **replicated** up to the decision budget first.
  Only a few dozen races out of a few thousand see a deployment on any given lap,
  and comparing branches on that few samples is comparing noise.

A switch threshold in expected points stops the policy churning on differences it
cannot resolve.

---

## 4. Validation

Backtest on 2025, which nothing has seen. Replaying the strategies teams actually
ran isolates the race model, because the strategy input is the truth.

| metric | value | note |
|---|---:|---|
| Mean absolute position error | 2.54 | over 479 car-races |
| Median absolute error | 1.92 | |
| Spearman rank correlation | 0.788 | mean over 24 races |
| Within one place | 49.9% | |
| Brier score, P(points) | 0.124 | |
| Brier baseline (climatology) | 0.250 | |

Calibration is good: predicted-versus-observed gaps stay under 8 points across
all five probability bins, and the model beats climatology by half.

**Where it is worst, and why:** Zandvoort (MAE 5.12) and Melbourne (4.29). Both
2025 races were heavily disrupted — Melbourne was wet, which the dry-tyre model
cannot represent at all. Las Vegas (4.35) is a first- and second-year circuit
with correspondingly thin per-circuit estimates. The errors concentrate exactly
where the model's stated assumptions break, which is the reassuring failure mode.

---

## Sensitivity ranking

Roughly, most to least consequential if wrong:

1. Neutralisation rate and the SC/VSC pit-loss discount — sets the entire value
   of having a stop in hand.
2. Degradation slope by compound and circuit — sets the stint-length decision.
3. Pit loss — sets the stop-count decision.
4. Overtaking difficulty — sets how much track position is worth.
5. Dirty-air magnitude — matters most in the midfield.
6. Reliability — mostly widens the distribution rather than moving the choice.
