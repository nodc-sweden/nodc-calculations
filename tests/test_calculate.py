import pytest
import numpy as np
import pandas as pd
from nodc_calculations import calculate


@pytest.mark.parametrize(
    "given_data, expected_din",
    (
        # case 1: h2s and nh4 valid, use only nh4
        ({
            "h2s": [5],
            "qh2s": ["1_0"],
            "nh4": [3],
            "qnh4": ["1_0"],
            "o2": [5],
            "qo2": ["1_0"],
            "nox": [np.nan],
            "qnox": ["1_0"],
            "no3": [2],
            "qno3": ["1_0"],
            "no2": [1],
            "qno2": ["1_0"],
        },
        3.0),
    (
        # case 2: no h2s or nh4, below det o2, correct nox, stb use nh4
        {
            "h2s": [np.nan],
            "qh2s": ["4_0"],
            "nh4": [np.nan],
            "qnh4": ["4_0"],
            "o2": [0.5],
            "qo2": ["6_0"],
            "nox": [3],
            "qnox": ["1_0"],
            "no3": [np.nan],
            "qno3": ["4_0"],
            "no2": [2],
            "qno2": ["1_0"],
        },
        np.nan,
    ),
    # case 3: incorrect h2s and nh4, below det o2, correct nox, stb use nh4
    (
        {
            "h2s": [5],
            "qh2s": ["4_0"],
            "nh4": [10],
            "qnh4": ["4_0"],
            "o2": [0.5],
            "qo2": ["6_0"],
            "nox": [3],
            "qnox": ["1_0"],
            "no3": [3],
            "qno3": ["4_0"],
            "no2": [2],
            "qno2": ["1_0"],
        },
        np.nan,
    ),
    # case 4: low correct o2, no nh4 data, set to nan
    ( 
        {
            "h2s": [np.nan],
            "qh2s": ["1_0"],
            "nh4": [np.nan],
            "qnh4": ["1_0"],
            "o2": [1],
            "qo2": ["1_0"],
            "nox": [3],
            "qnox": ["1_0"],
            "no3": [3],
            "qno3": ["4_0"],
            "no2": [2],
            "qno2": ["1_0"],
        },
        np.nan,
    ),
    # case 5: low correct o2, nh4 below det, set to sum of all
    ( 
        {
            "h2s": [np.nan],
            "qh2s": ["1_0"],
            "nh4": [1],
            "qnh4": ["6_0"],
            "o2": [1],
            "qo2": ["1_0"],
            "nox": [3],
            "qnox": ["1_0"],
            "no3": [3],
            "qno3": ["4_0"],
            "no2": [2],
            "qno2": ["1_0"],
        },
        4,
    ),
    # case 6: correct o2, nh4, no3, no2, no nox
    ( 
        {
            "h2s": [np.nan],
            "qh2s": ["0_0"],
            "nh4": [5],
            "qnh4": ["1_0"],
            "o2": [5],
            "qo2": ["1_0"],
            "nox": [np.nan],
            "qnox": ["1_0"],
            "no3": [3],
            "qno3": ["1_0"],
            "no2": [2],
            "qno2": ["1_0"],
        },
        10,
    ),
    # case 7: as case 6 but no2 nan, return nh4+no3
    ( 
        {
            "h2s": [np.nan],
            "qh2s": ["0_0"],
            "nh4": [5],
            "qnh4": ["1_0"],
            "o2": [5],
            "qo2": ["1_0"],
            "nox": [np.nan],
            "qnox": ["1_0"],
            "no3": [3],
            "qno3": ["1_0"],
            "no2": [np.nan],
            "qno2": ["1_0"],
        },
        8,
    ),
    # case 8: all valid, no h2s, return nh4+nox
    ( 
        {
            "h2s": [np.nan],
            "qh2s": ["0_0"],
            "nh4": [5],
            "qnh4": ["1_0"],
            "o2": [5],
            "qo2": ["1_0"],
            "nox": [10],
            "qnox": ["1_0"],
            "no3": [7],
            "qno3": ["1_0"],
            "no2": [2],
            "qno2": ["1_0"],
        },
        15,
    ),
    # case 9: all valid no nox, no h2s, return nh4+no3+no2
    ( 
        {
            "h2s": [np.nan],
            "qh2s": ["0_0"],
            "nh4": [5],
            "qnh4": ["1_0"],
            "o2": [5],
            "qo2": ["1_0"],
            "nox": [np.nan],
            "qnox": ["1_0"],
            "no3": [7],
            "qno3": ["1_0"],
            "no2": [2],
            "qno2": ["1_0"],
        },
        14,
    ),
    # case 10: nh4 below det, no h2s, return nox
    ( 
        {
            "h2s": [np.nan],
            "qh2s": ["0_0"],
            "nh4": [1],
            "qnh4": ["6_0"],
            "o2": [5],
            "qo2": ["1_0"],
            "nox": [11],
            "qnox": ["1_0"],
            "no3": [7],
            "qno3": ["1_0"],
            "no2": [2],
            "qno2": ["1_0"],
        },
        12,
    ),
    ),
)
def test_din(given_data, expected_din):
    data = pd.DataFrame(given_data)

    calculate.dissolved_inorganic_nitrogen(data)

    # testar ny funktion mot förväntat
    np.testing.assert_equal(data['din'].values[0], expected_din)

    din = calculate.get_DIN(given_data)

    # testar stb mot förväntat
    # np.testing.assert_equal(din, expected_din)
