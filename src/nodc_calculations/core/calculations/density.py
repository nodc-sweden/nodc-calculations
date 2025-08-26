from gsw import SA_from_SP, pt_from_t, CT_from_pt, pot_rho_t_exact, rho
from gsw.conversions import p_from_z
import numpy as np


def potential_density_core(
    rows,
    salinity_source_column: str = "salinity",
    temperature_source_column: str = "temperature",
):
    """
    Calculate potential density from salt, temp, and depth.
    Input: list of dicts with keys 'salt', 'temp', 'depth'
    Output: list of floats
    """
    results = []
    for row in rows:
        practical_salinity = row.get(salinity_source_column)
        temperature = row.get(temperature_source_column)
        depth = row.get("DEPH")
        latitude = row.get("sample_latitude_dd", 58)
        longitude = row.get("sample_longitude_dd", 11)

        if practical_salinity is None or temperature is None or depth is None:
            results.append(np.nan)
            continue

        # pressure from depth
        pressure = p_from_z(-depth, latitude)

        # absolute salinity
        absolute_salinity = SA_from_SP(
            practical_salinity, pressure, longitude, latitude
        )

        # density referenced to 0 dbar
        print("calculated potential dens")
        dens = pot_rho_t_exact(absolute_salinity, temperature, pressure, p_ref=0)
        print(dens)
        results.append(dens)

    return results


def in_situ_density_core(
    rows,
    salinity_source_column: str = "salinity",
    temperature_source_column: str = "temperature",
):
    """
    Calculate in situ density from salt, temp, and depth.
    Input: list of dicts with keys 'salt', 'temp', 'depth'
    Output: list of floats
    """
    results = []
    for row in rows:
        practical_salinity = row.get(salinity_source_column)
        temperature = row.get(temperature_source_column)
        depth = row.get("DEPH")
        latitude = row.get("sample_latitude_dd", row.get("LATIT_DD", 58))
        longitude = row.get("sample_longitude_dd", row.get("LONGI_DD", 11))

        if practical_salinity is None or temperature is None or depth is None:
            results.append(np.nan)
            continue

        # pressure from depth
        pressure = p_from_z(-depth, latitude)

        # absolute salinity
        absolute_salinity = SA_from_SP(
            practical_salinity, pressure, longitude, latitude
        )

        # potential temperature
        potential_temperature = pt_from_t(
            absolute_salinity, temperature, pressure, p_ref=0
        )

        # conservative temperature
        conservative_temperature = CT_from_pt(absolute_salinity, potential_temperature)

        # density
        print("calculated density")
        dens = rho(absolute_salinity, conservative_temperature, pressure)
        print(dens)
        results.append(dens)

    return results
