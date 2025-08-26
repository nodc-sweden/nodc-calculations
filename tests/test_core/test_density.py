import numpy as np
import pytest
from nodc_calculations.core.calculations.density import (
    in_situ_density_core,
    potential_density_core,
)


@pytest.mark.parametrize(
    "input_rows, expected_density",
    (
        # case 1: all valid
        (
            [{"temperature": 15, "salt": 0.5, "DEPH": 0}],
            [999.55629],
        ),
        # case 2: all valid
        (
            [{"temperature": -1.8, "salt": 34.8, "DEPH": 100}],
            [1028.507005],
        ),
        # case 3: one is nan
        (
            [{"temperature": 10, "salt": np.nan, "DEPH": 0}],
            [np.nan],
        ),
    ),
)
def test_in_situ_density(input_rows, expected_density):
    print(input_rows)
    result = in_situ_density_core(input_rows, salinity_source_column="salt")
    print(result)
    np.testing.assert_equal(np.round(result, 3), np.round(expected_density, 3))


@pytest.mark.parametrize(
    "input_rows, expected_density",
    (
        # case 1: all valid
        (
            [{"temperature": 15, "salt": 0.5, "DEPH": 0}],
            [999.556808],
        ),
        # case 2: all valid
        (
            [{"temperature": -1.8, "salt": 34.8, "DEPH": 100}],
            [1028.02108],
        ),
        # case 3: one is nan
        (
            [{"temperature": 10, "salt": np.nan, "DEPH": 0}],
            [np.nan],
        ),
    ),
)
def test_potential_density(input_rows, expected_density):
    print(input_rows)
    result = potential_density_core(input_rows, salinity_source_column="salt")
    print(result)
    np.testing.assert_equal(np.round(result, 3), np.round(expected_density, 3))
