"""lightsail: idealized 1-D simulation of laser-propelled lightsails (classical and relativistic).

Modules
-------
params       input parameters (SailParams) and derived quantities, UCSB-calculator conventions
physics      vectorized closed-form solutions and semi-analytic time integration
simulate     direct numerical integration with scipy.integrate.solve_ivp
sensitivity  log-elasticities and one-at-a-time (tornado) sensitivity analysis
uncertainty  Monte Carlo and first-order uncertainty propagation
"""

from .constants import C, G_N, LY, AU, YEAR, GW, GRAM
from .params import SailParams, baseline, PRESETS
from .physics import CLASSICAL, RELATIVISTIC, MODELS, mission_summary

__all__ = ["C", "G_N", "LY", "AU", "YEAR", "GW", "GRAM", "SailParams", "baseline", "PRESETS",
           "CLASSICAL", "RELATIVISTIC", "MODELS", "mission_summary"]
