import numpy as np
import pytest
from nodc_calculations.core.calculations.ntrz import ntrz_core


@pytest.mark.parametrize(
    "input_rows, expected_ntrz",
    [
        # Case 1: no ntrz valid ntra and ntri
        (
            [
                {
                    "NTRZ": np.nan,
                    "Q_NTRZ": "1_0",
                    "NTRA": 2,
                    "Q_NTRA": "1_0",
                    "NTRI": 1,
                    "Q_NTRI": "1_0",
                }
            ],
            [3],
        ),
        # case 2: as case 1 but NTRI nan, return NTRA
        (
            [
                {
                    "NTRZ": np.nan,
                    "Q_NTRZ": "1_0",
                    "NTRA": 3,
                    "Q_NTRA": "1_0",
                    "NTRI": np.nan,
                    "Q_NTRI": "1_0",
                }
            ],
            [3],
        ),
        # case 3: all valid return ntrz
        (
            [
                {
                    "NTRZ": 5,
                    "Q_NTRZ": "1_0",
                    "NTRA": 3,
                    "Q_NTRA": "1_0",
                    "NTRI": 1,
                    "Q_NTRI": "1_0",
                }
            ],
            [5],
        ),
        # case 3: only ntri valid return nan
        (
            [
                {
                    "NTRZ": 5,
                    "Q_NTRZ": "4_0",
                    "NTRA": 3,
                    "Q_NTRA": "4_0",
                    "NTRI": 1,
                    "Q_NTRI": "1_0",
                }
            ],
            [None],
        ),
        # case 3: only ntri not nan return nan
        (
            [
                {
                    "NTRZ": None,
                    "Q_NTRZ": "1_0",
                    "NTRA": None,
                    "Q_NTRA": "1_0",
                    "NTRI": 1,
                    "Q_NTRI": "1_0",
                }
            ],
            [None],
        ),
    ],
)
def test_din_single_row(input_rows, expected_ntrz):
    """Test NTRZ calculation"""
    print(input_rows)
    ntrz_result = ntrz_core(input_rows)
    print(ntrz_result)
    np.testing.assert_equal(ntrz_result, expected_ntrz)
