"""Component 2: the race simulator.

A lap-by-lap discrete-event model of a Grand Prix, vectorised over an
*ensemble* of races rather than over cars. See :mod:`pitwall.sim.engine` for
why that choice is what makes 10,000-race Monte Carlo affordable.

Every behavioural assumption in here has an entry in DESIGN.md naming the data
it was fitted or calibrated from, and every one of them can be switched off
from the config, which is what Component 5 does.
"""

from pitwall.sim.engine import EnsembleResult, simulate_ensemble
from pitwall.sim.params import SimParams, load_sim_params
from pitwall.sim.strategy import Strategy

__all__ = [
    "EnsembleResult",
    "SimParams",
    "Strategy",
    "load_sim_params",
    "simulate_ensemble",
]
