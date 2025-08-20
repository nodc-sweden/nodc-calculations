import re
import numpy as np
from .ntrz import ntrz_core
from nodc_calculations.core.utils import is_valid, is_below_det, safe_sum


def dissolved_inorganic_nitrogen_core(rows):
    """
    | Rule # | Description                                                                             | Matches Cases                                 | Expected Result    |
    | ------ | --------------------------------------------------------------------------------------- | --------------------------------------------- | ------------------ |
    | **1**  | **Valid H₂S** and **valid AMON** → **return AMON only**                                 | 1                                             | `AMON`             |
    | **2**  | **Low DOXY** and **valid AMON** → **NTRZ\_corr + AMON**                                 | 2, 5, 6, 7                                    | `NTRZ_corr + AMON` |
    | **3**  | **NTRZ or NTRA below detection** **and** **AMON below detection** → **NTRZ\_corr only** | 3, 10, 11, 12                                 | `NTRZ_corr`        |
    | **4**  | **Normal case**: AMON valid, no low DOXY, no H₂S → **NTRZ\_corr + AMON**                | 8, 9, 13                                      | `NTRZ_corr + AMON` |
    | **5**  | **AMON below detection**, no low DOXY, no H₂S → **NTRZ\_corr only**                     | (possibly backup for some edge case overlaps) | `NTRZ_corr`        |
    """
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise TypeError("Expected list of dicts")

    results = []

    for row in rows:
        # Extract values
        NTRA, Q_NTRA = row.get("NTRA"), row.get("Q_NTRA", "")
        NTRZ, Q_NTRZ = row.get("NTRZ"), row.get("Q_NTRZ", "")
        H2S, Q_H2S = row.get("H2S"), row.get("Q_H2S", "")
        AMON, Q_AMON = row.get("AMON"), row.get("Q_AMON", "")
        DOXY, Q_DOXY = row.get("DOXY_BTL"), row.get("Q_DOXY_BTL", "")

        # Validity checks
        valid_H2S = is_valid(H2S, Q_H2S, r"6|4|3|B|S|<")
        valid_AMON = is_valid(AMON, Q_AMON, r"4|3|B|S")

        below_det_NTRA = is_below_det(NTRA, Q_NTRA, r"6|<")
        below_det_NTRZ = is_below_det(NTRZ, Q_NTRZ, r"6|<")
        below_det_AMON = is_below_det(AMON, Q_AMON, r"6|<")

        low_doxy = (
            (DOXY is not None)
            and not np.isnan(DOXY)
            and (DOXY <= 2)
            and not re.search(r"4|3|B|S", str(Q_DOXY) if Q_DOXY is not None else "")
        )

        # Compute NTRZ_corr
        NTRZ_corr = ntrz_core([row])[0]

        # define rules
        rules = [
            # Rule 1
            # Case 1
            (valid_H2S and valid_AMON, lambda: AMON),
            # Rule 2
            # Cases 2, 5, 6, 7
            (
                low_doxy and valid_AMON,
                lambda: safe_sum(v for v in [NTRZ_corr, AMON] if v is not None),
            ),
            # Rule 3
            # Cases 3, 10, 11, 12
            (
                (below_det_NTRZ or below_det_NTRA) and below_det_AMON and NTRZ_corr,
                lambda: NTRZ_corr,
            ),
            # Rule 4
            # Cases 8, 9, 13
            (
                valid_AMON
                and not low_doxy
                and not valid_H2S
                and not below_det_AMON
                and NTRZ_corr,
                lambda: NTRZ_corr + AMON,
            ),
            # Rule 5
            # Backup case for AMON below det, no low DOXY, no H₂S
            (
                not low_doxy and not valid_H2S and below_det_AMON and NTRZ_corr,
                lambda: NTRZ_corr,
            ),
        ]

        # --- STEP 2: Decide output based on return type priority ---
        # --- Evaluate rules ---
        din = None
        for cond, action in rules:
            if cond:
                din = action()
                break

        results.append(din)

    return results
