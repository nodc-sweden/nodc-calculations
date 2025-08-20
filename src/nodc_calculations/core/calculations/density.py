from gsw import pot_rho_t_exact
from gsw.conversions import p_from_z
import numpy as np


def density_core(rows, oxygen_source_column: str = "oxygen", salinity_source_column: str = "salinity", temperature_source_column: str = "temperature",latitude: float = 58):
    """
    Calculate potential density anomaly from salt, temp, and depth.
    Input: list of dicts with keys 'salt', 'temp', 'depth'
    Output: list of floats
    """
    results = []
    for row in rows:
        sal = row.get(salinity_source_column)
        temp = row.get(temperature_source_column)
        depth = row.get("DEPH")

        if sal is None or temp is None or depth is None:
            results.append(np.nan)
            continue

        # pressure from depth
        pressure = p_from_z(-depth, latitude)

        # density referenced to 0 dbar
        dens = pot_rho_t_exact(sal, temp, pressure, 0)
        results.append(dens)

    return results
