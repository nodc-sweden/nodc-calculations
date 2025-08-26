import numpy as np
import pytest

from nodc_calculations.core.utils import is_valid, is_below_det


@pytest.mark.parametrize(
    "value, q_flag, expected",
    [
        (5.0, None, True),  # None q_flag → treat as "1"
        (5.0, np.nan, True),  # NaN q_flag → treat as "1"
        (5.0, 1, True),  # numeric flag 1 → valid
        (5.0, "B", False),  # bad flag B → invalid
        (None, None, False),  # missing value → invalid
        (5.0, "1_0461_3", False),  # str of multiple flags, → invalid
    ],
)
def test_is_valid_handles_missing_q_flag(value, q_flag, expected):
    assert is_valid(value, q_flag, r"B|S|<|4|3|6") == expected


@pytest.mark.parametrize(
    "value, q_flag, expected",
    [
        (5.0, 6, True),  # numeric single q_flag → true
        (5.0, "<", True),  # str single q_flag → true
        (5.0, 1, False),  # numeric flag 1 → false
        (5.0, "6", True),  # missing value → invalid
    ],
)
def test_is_below_det_handles_float_str_longstr_q_flag(value, q_flag, expected):
    assert is_below_det(value, q_flag, r"<|6") == expected
