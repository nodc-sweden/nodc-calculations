import numpy as np
import re


def is_valid(value, q_flag, bad_flags_pattern):
    """Check if a value is valid based on its quality flag and missing/NaN status."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    # Handle missing q_flag (None or NaN) → treat as "1"
    if q_flag is None or (isinstance(q_flag, float) and np.isnan(q_flag)):
        q_flag = "1"
    elif isinstance(q_flag, (float, int)):
        q_flag = str(int(q_flag))  # convert numeric flags to string
    return not re.search(bad_flags_pattern, q_flag or "")


def is_below_det(value, q_flag, below_det_flags_pattern):
    """Check if a value is below detection limit based on quality flag."""
    if isinstance(q_flag, (float, int)):
        q_flag = str(int(q_flag))
    return re.search(below_det_flags_pattern, q_flag or "") is not None


def safe_sum(values):
    """Sum values ignoring None and NaN."""
    return sum(
        v
        for v in values
        if v is not None and not (isinstance(v, float) and np.isnan(v))
    )
