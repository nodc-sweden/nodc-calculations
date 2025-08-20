# src/nodc_calculations/core/calculations/ntrz.py
import numpy as np
from typing import Optional
from nodc_calculations.core.utils import is_valid, is_below_det, safe_sum


def ntrz_core(rows) -> Optional[float]:
    """
    Calculate corrected NTRZ (nitrate + nitrite) from given parameters.

    - If NTRZ missing and both NTRA & NTRI are below detection -> use NTRA
    - If NTRZ missing but NTRA valid -> use NTRA + NTRI
    - Else if NTRZ valid -> use NTRZ
    - Else -> None
    """
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise TypeError("Expected list of dicts")

    results = []

    for row in rows:
        # Extract values
        NTRA, Q_NTRA = row.get("NTRA"), row.get("Q_NTRA", "")
        NTRI, Q_NTRI = row.get("NTRI"), row.get("Q_NTRI", "")
        NTRZ, Q_NTRZ = row.get("NTRZ"), row.get("Q_NTRZ", "")

        valid_NTRA = is_valid(NTRA, Q_NTRA, r"4|3|B|S")
        below_det_NTRA = is_below_det(NTRA, Q_NTRA, r"6|<")

        below_det_NTRI = is_below_det(NTRI, Q_NTRI, r"6|<")

        valid_NTRZ = is_valid(NTRZ, Q_NTRZ, r"4|3|B|S")

        # define rules
        rules = [
            # Rule 1 no NTRZ NTRA and NTRI below det, return NTRA
            (
                (NTRZ is None or (isinstance(NTRZ, float) and np.isnan(NTRZ)))
                and below_det_NTRA
                and below_det_NTRI,
                lambda: NTRA,
            ),
            # Rule 2 no NTRZ, valid NTRA, return sum of NTRA and NTRI, ignore nan/None NTRI
            (
                (NTRZ is None or (isinstance(NTRZ, float) and np.isnan(NTRZ)))
                and valid_NTRA,
                lambda: safe_sum([NTRA, NTRI]),
            ),
            # Rule 3 valid NTRZ return this
            (valid_NTRZ, lambda: NTRZ),
        ]

        # --- STEP 2: Decide output based on return type priority ---
        # --- Evaluate rules ---
        ntrz_corr = None
        for cond, action in rules:
            if cond:
                ntrz_corr = action()
                break

        results.append(ntrz_corr)

    return results
