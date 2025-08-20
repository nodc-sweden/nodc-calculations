import numpy as np


def get_prio_core(rows, par1, par2, q1, q2, bad_flags={"B", "4"}):
    """
    Generic core function for prioritizing between two parameters.

    Parameters
    ----------
    rows : list of dicts
        Each dict is one observation with parameter and quality flag values.
    par1 : str
        Column name of the first-priority parameter.
    par2 : str
        Column name of the second-priority parameter.
    q1 : str
        Column name of the quality flag for par1.
    q2 : str
        Column name of the quality flag for par2.

    Returns
    -------
    list
        Values selected according to the prioritization rules.
    """
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise TypeError("Expected list of dicts")

    results = []
    bad_flags = normalize_flags(bad_flags)

    for row in rows:
        v1, v2 = row.get(par1), row.get(par2)
        qv1, qv2 = normalize_flags(row.get(q1, ""), row.get(q2, ""))

        # Rule 1: par1 is valid
        if v1 is not None and not (isinstance(v1, float) and np.isnan(v1)):
            if qv1 not in bad_flags:
                results.append(v1)
                continue

        # Rule 2: par2 is valid
        if qv2 not in bad_flags:
            results.append(v2)
            continue

        # Otherwise NaN
        results.append(np.nan)

    return results


def salinity_core(rows):
    return get_prio_core(rows, "SALT_CTD", "SALT_BTL", "Q_SALT_CTD", "Q_SALT_BTL")


def temperature_core(rows):
    return get_prio_core(rows, "TEMP_CTD", "TEMP_BTL", "Q_TEMP_CTD", "Q_TEMP_BTL")


def normalize_flags(flags):
    """
    Normalize a sequence of quality flags so everything is comparable as strings.
    - Floats like 4.0 -> "4"
    - Ints -> "4"
    - Strings kept as-is
    - None / NaN skipped
    """
    normalized = set()
    for flag in flags:
        if flag is None:
            normalized.add(None)
        if isinstance(flag, float):
            if np.isnan(flag):
                normalized.add(None)
            normalized.add(str(int(flag)))
        elif isinstance(flag, int):
            normalized.add(str(flag))
        else:
            normalized.add(str(flag))
    return normalized
