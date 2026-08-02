"""Component 3: Monte Carlo strategy search.

Given a grid slot, a pace and a tyre allocation, enumerate candidate stint
plans and score each one over a large ensemble of simulated races, reporting
the full distribution of finishing positions rather than an expected value.

:mod:`pitwall.optimize.reactive` adds the online half: when a neutralisation
deploys mid-race, the remaining distance is re-optimised from the state the
race is actually in.
"""

from pitwall.optimize.candidates import enumerate_candidates
from pitwall.optimize.mc import StrategyEvaluation, evaluate_candidates

__all__ = ["StrategyEvaluation", "enumerate_candidates", "evaluate_candidates"]
