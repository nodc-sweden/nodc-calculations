from nodc_calculations.core.utils import is_valid, is_below_det
from nodc_calculations.core.calculations.density import in_situ_density_core
from gsw import O2sol_SP_pt, SA_from_SP, pt_from_t
from gsw.conversions import p_from_z
import numpy as np


def oxygen_core(rows):
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise TypeError("Expected list of dicts")

    results = []

    for row in rows:
        # Extract values
        DOXY_BTL, Q_DOXY_BTL = row.get("DOXY_BTL"), row.get("Q_DOXY_BTL", "")
        DOXY_CTD, Q_DOXY_CTD = row.get("DOXY_CTD"), row.get("Q_DOXY_CTD", "")
        H2S, Q_H2S = row.get("H2S"), row.get("Q_H2S", "")

        valid_btl = is_valid(
            DOXY_BTL, q_flag=Q_DOXY_BTL, bad_flags_pattern=r"B|S|<|4|3|6"
        )
        valid_ctd = is_valid(
            DOXY_CTD, q_flag=Q_DOXY_CTD, bad_flags_pattern=r"B|S|<|4|3|6"
        )
        valid_h2s = is_valid(H2S, Q_H2S, r"B|S|Z|<|4|3|6")
        below_det_btl = is_below_det(
            DOXY_BTL, q_flag=Q_DOXY_BTL, below_det_flags_pattern=r"<|6"
        )
        below_det_ctd = is_below_det(
            DOXY_CTD, q_flag=Q_DOXY_CTD, below_det_flags_pattern=r"<|6"
        )
        below_det_h2s = is_below_det(H2S, q_flag=Q_H2S, below_det_flags_pattern=r"<|6")

        rules = [
            # Rule 1 # h2s valid --> 0
            (valid_h2s, lambda: 0),
            # Rule 2 both h2s and oxygen below det --> 0
            (
                below_det_h2s and (below_det_btl or (below_det_ctd and not valid_btl)),
                lambda: 0,
            ),
            # Rule 3 O2 BTL is valid -> O2 BTL
            (valid_btl or below_det_btl, lambda: DOXY_BTL),
            # Rule 4 # O2 CTD exists and Q O2 CTD is not B|S|< -> O2 CTD
            (valid_ctd and not below_det_ctd, lambda: DOXY_CTD),
            # Rule 5 below_det_ctd --> 0
            (below_det_ctd, lambda: 0),
        ]

        # --- STEP 2: Decide output based on return type priority ---
        # --- Evaluate rules ---
        oxygen = None
        for cond, action in rules:
            if cond:
                oxygen = action()
                break

        results.append(oxygen)

    return results


def oxygen_saturation_core(
    rows,
    oxygen_source_column: str = "oxygen",
    salinity_source_column: str = "salinity",
    temperature_source_column: str = "temperature",
    latitude: float = 58,
):
    """
    Calculate oxygen saturation percentage.
    Input: list of dicts with keys 'salt', 'temp', 'depth', and oxygen_source_column
    Output: list of floats (oxygen_sat)
    """
    results = []
    for row in rows:
        practical_salinity = row.get(salinity_source_column)
        temperature = row.get(temperature_source_column)
        depth = row.get("DEPH")
        oxy = row.get(oxygen_source_column)
        latitude = row.get("sample_latitude_dd", row.get("LATIT_DD", 58.0))
        longitude = row.get("sample_longitude_dd", row.get("LONGI_DD", 11.0))

        if (
            practical_salinity is None
            or temperature is None
            or depth is None
            or oxy is None
            or (isinstance(practical_salinity, float) and np.isnan(practical_salinity))
            or (isinstance(temperature, float) and np.isnan(temperature))
            or (isinstance(depth, float) and np.isnan(depth))
            or (isinstance(oxy, float) and np.isnan(oxy))
        ):
            results.append(np.nan)
            continue

        # pressure from pressure
        pressure = p_from_z(-depth, latitude)

        # absolute salinity
        absolute_salinity = SA_from_SP(
            practical_salinity, pressure, longitude, latitude
        )

        # potential temperature
        potential_temperature = pt_from_t(
            absolute_salinity, temperature, pressure, p_ref=0
        )

        # density
        dens = in_situ_density_core([row], salinity_source_column="salt")[0]
        # dens = pot_rho_t_exact(sal, temp, p_from_z(-depth, latitude), 0)

        # oxygen solubility (converted to µmol/L)
        gsw_val = (
            O2sol_SP_pt(practical_salinity, potential_temperature)
            * (dens / 1000)
            / 44.661
        )

        # saturation percentage
        oxy_sat = oxy / gsw_val * 100 if gsw_val else np.nan

        results.append(oxy_sat)

    return results
