"""Input parameters of the idealized 1-D laser-driven lightsail and the quantities derived from them.

Conventions follow the UCSB Laser Propulsion calculators (Lubin group) and
P. Lubin, "A Roadmap to Interstellar Flight", JBIS 69, 40 (2016), arXiv:1604.01356:

* sail mass          m_s = xi * D^2 * h * rho          (xi = 1 square, pi/4 circular)
* total mass         m   = m_s + m0
* momentum factor    eta = 2 R + (1 - R) A             (R reflected, A absorbed fraction of the rest)
* force (near field) F0  = eta * eps_beam * P / c
* L0                 L0  = d D / (2 lambda alpha)      (beam spot equals the sail size)
* "optimal sail"     D   = sqrt(m0 / (xi h rho))       (sail mass equals payload mass)

The beam (P, d, lambda, alpha) is treated as a fixed, idealized input: this project studies the
spacecraft's motion, not the design of the laser.
"""

from __future__ import annotations

import math

import numpy as np
from dataclasses import dataclass, replace, asdict

from .constants import (
    C, G_N, SIGMA_SB, LY, GW, GRAM,
    ALPHA_CIRCULAR_ARRAY, ALPHA_SQUARE_ARRAY, SAIL_AREA_FACTOR,
)


@dataclass(frozen=True)
class SailParams:
    # --- Beam (idealized, fixed inputs) ---
    power: float = 100 * GW            # total optical power P [W]
    array_size: float = 10_000.0       # laser array size d [m]
    wavelength: float = 1060e-9        # lambda [m]
    circular_array: bool = True        # alpha = 1.22 (circular) or 1 (square)
    beam_quality: float = 1.0          # extra multiplier on alpha (>= 1 for a non-ideal beam)
    beam_efficiency: float = 1.0       # fraction of P in the main beam
    # --- Sail and payload ---
    sail_size: float | None = None     # D [m]; None -> "optimal sail" (m_sail = m0)
    thickness: float = 1e-6            # h [m]
    density: float = 1000.0            # rho [kg/m^3]
    sail_shape: str = "square"         # "square" or "circular"
    reflectivity: float = 1.0          # R, fraction of incident light reflected
    absorption: float = 1.0            # A, fraction of the non-reflected light that is absorbed
    emissivity_front: float = 1.0
    emissivity_back: float = 1.0
    payload_mass: float = 1 * GRAM     # m0 [kg]
    # --- Mission ---
    target_distance: float = 4.37 * LY # [m]
    illumination_stop: float = math.inf  # laser switched off at x = illumination_stop * L0

    # ------------------------------------------------------------------ derived quantities
    @property
    def areal_density(self) -> float:
        """Sail mass per unit area sigma = h * rho [kg/m^2]."""
        return self.thickness * self.density

    @property
    def area_factor(self) -> float:
        return SAIL_AREA_FACTOR[self.sail_shape]

    @property
    def D(self) -> float:
        """Sail size D [m] (side of a square sail or diameter of a circular one)."""
        if self.sail_size is not None:
            return self.sail_size
        return np.sqrt(self.payload_mass / (self.area_factor * self.areal_density))

    @property
    def sail_area(self) -> float:
        return self.area_factor * self.D ** 2

    @property
    def sail_mass(self) -> float:
        return self.areal_density * self.sail_area

    @property
    def total_mass(self) -> float:
        return self.sail_mass + self.payload_mass

    @property
    def alpha(self) -> float:
        base = ALPHA_CIRCULAR_ARRAY if self.circular_array else ALPHA_SQUARE_ARRAY
        return base * self.beam_quality

    @property
    def momentum_factor(self) -> float:
        """eta = 2R + (1-R)A: photon momentum delivered per unit of incident momentum (0..2)."""
        R = self.reflectivity
        return 2.0 * R + (1.0 - R) * self.absorption

    @property
    def beam_power(self) -> float:
        return self.beam_efficiency * self.power

    @property
    def force(self) -> float:
        """Near-field force on a stationary sail F0 = eta P0 / c [N]."""
        return self.momentum_factor * self.beam_power / C

    @property
    def a0(self) -> float:
        """Peak (near-field, low-speed) acceleration [m/s^2]."""
        return self.force / self.total_mass

    @property
    def L0(self) -> float:
        """Distance at which the diffraction-limited spot equals the sail size [m]."""
        return self.array_size * self.D / (2.0 * self.wavelength * self.alpha)

    @property
    def x_stop(self) -> float:
        return self.illumination_stop * self.L0

    @property
    def beta0_classical(self) -> float:
        """Classical speed at L0 in units of c: beta0 = sqrt(2 a0 L0)/c. Sets the size of relativistic effects."""
        return np.sqrt(2.0 * self.a0 * self.L0) / C

    @property
    def t0_classical(self) -> float:
        """Classical time to reach L0, sqrt(2 L0 / a0) [s]."""
        return np.sqrt(2.0 * self.L0 / self.a0)

    @property
    def flux(self) -> float:
        """Near-field flux on the sail [W/m^2]."""
        return self.beam_power / self.sail_area

    @property
    def absorbed_flux(self) -> float:
        return self.flux * (1.0 - self.reflectivity) * self.absorption

    @property
    def equilibrium_temperature(self) -> float:
        """Peak radiative-equilibrium sail temperature [K] (UCSB calculator definition)."""
        eps = self.emissivity_front + self.emissivity_back
        return (self.absorbed_flux / (SIGMA_SB * eps)) ** 0.25

    # ------------------------------------------------------------------ helpers
    def resolved(self) -> "SailParams":
        """Copy with the sail size fixed to its current value (so later changes to h or m0 keep D)."""
        return replace(self, sail_size=self.D)

    def with_(self, **changes) -> "SailParams":
        return replace(self, **changes)

    def with_areal_density(self, sigma: float) -> "SailParams":
        """Copy with sail mass per area sigma [kg/m^2], achieved by changing the thickness."""
        return replace(self, thickness=sigma / self.density)

    def summary(self) -> dict:
        return {
            "D_m": self.D, "areal_density_g_m2": self.areal_density / GRAM,
            "sail_mass_g": self.sail_mass / GRAM, "total_mass_g": self.total_mass / GRAM,
            "momentum_factor": self.momentum_factor, "alpha": self.alpha,
            "force_N": self.force, "a0_m_s2": self.a0, "a0_gn": self.a0 / G_N,
            "L0_m": self.L0, "beta0_classical": self.beta0_classical,
            "t0_classical_s": self.t0_classical,
            "T_eq_K": self.equilibrium_temperature,
        }

    def as_dict(self) -> dict:
        return asdict(self)


def baseline() -> SailParams:
    """Default inputs of the UCSB 1-D calculators: 100 GW, 10 km circular array, 1.06 um,
    1 g payload, 1 um x 1 g/cc square 'optimal' sail (1 m, 1 g), perfect reflector, target 4.37 ly."""
    return SailParams()


PRESETS = {
    "UCSB calculator baseline": baseline(),
    "Thicker sail (10 g/m^2)": baseline().with_(thickness=10e-6),
    "Lossy sail (R = 0.6)": baseline().with_(reflectivity=0.6),
    "Gram-scale, 1 TW-class beam": baseline().with_(power=1000 * GW),
    # Kulkarni, Lubin & Zhang (2018), Table 1, first row: 10 km square array, 1064 nm, 1 um x 1.4 g/cm^3 sail
    "Kulkarni et al. 2018, Table 1 (1 g)": baseline().with_(circular_array=False, wavelength=1064e-9, density=1400.0),
}
