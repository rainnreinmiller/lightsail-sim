"""Physical constants, unit conversions and mission landmarks (SI units throughout)."""

import math

C = 299_792_458.0            # speed of light [m/s]
G_N = 9.80665                # standard gravity [m/s^2]
SIGMA_SB = 5.670374419e-8    # Stefan-Boltzmann constant [W m^-2 K^-4]

# Lengths
KM = 1e3
AU = 149_597_870_700.0                 # astronomical unit [m]
LY = 9.4607e15                         # light year [m] (same value as the UCSB calculators)
# Times
HOUR = 3600.0
DAY = 86_400.0
YEAR = 365.25 * DAY                    # Julian year [s]
# Mass / power
GRAM = 1e-3
GW = 1e9

# Diffraction factor for a circular (Airy) vs square aperture: spot ~ 2*alpha*lambda*L/d
ALPHA_CIRCULAR_ARRAY = 1.22
ALPHA_SQUARE_ARRAY = 1.0

# Sail area factor xi: area = xi * D^2
SAIL_AREA_FACTOR = {"square": 1.0, "circular": math.pi / 4.0}

# Interstellar targets [m]
TARGETS = {
    "Alpha Centauri A/B": 4.37 * LY,      # default target in the UCSB calculators
    "Proxima Centauri": 4.2465 * LY,
    "Barnard's Star": 5.96 * LY,
}

# Landmarks drawn on the distance track (distance from Earth, order-of-magnitude scale) [m]
LANDMARKS = {
    "Moon": 3.844e8,
    "1 AU": AU,
    "Neptune": 30.07 * AU,
    "Heliopause": 120.0 * AU,
    "Inner Oort cloud": 2_000.0 * AU,
}
