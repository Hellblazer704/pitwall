"""Component 1: hierarchical Bayesian tyre degradation.

The model estimates, for every compound at every circuit, how much lap time a
tyre loses per lap of age, with the fuel-burn effect separated out rather than
absorbed into the degradation slope.

See :mod:`pitwall.degradation.design` for the identification argument, which is
the part that actually matters, and :mod:`pitwall.degradation.gibbs` for the
sampler.
"""

from pitwall.degradation.design import DesignData, build_design
from pitwall.degradation.model import DegradationPosterior

__all__ = ["DegradationPosterior", "DesignData", "build_design"]
