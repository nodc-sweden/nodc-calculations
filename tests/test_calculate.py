import pytest
import numpy as np
import pandas as pd
from nodc_calculations import calculate


@pytest.mark.parametrize(
    "given_data, expected_din",
    (
        # case 1: H2S and AMON valid, use only AMON
        (
            {
                "H2S": [5],
                "Q_H2S": ["1_0"],
                "AMON": [3],
                "Q_AMON": ["1_0"],
                "doxy": [5],
                "Q_doxy": ["1_0"],
                "NTRZ": [np.nan],
                "Q_NTRZ": ["1_0"],
                "NTRA": [2],
                "Q_NTRA": ["1_0"],
                "NTRI": [1],
                "Q_NTRI": ["1_0"],
            },
            3.0,
        ),
        # case 2: no H2S or AMON, below det doxy, correct NTRZ, stb use AMON
        (
            {
                "H2S": [np.nan],
                "Q_H2S": ["4_0"],
                "AMON": [np.nan],
                "Q_AMON": ["4_0"],
                "doxy": [0.5],
                "Q_doxy": ["6_0"],
                "NTRZ": [3],
                "Q_NTRZ": ["1_0"],
                "NTRA": [np.nan],
                "Q_NTRA": ["4_0"],
                "NTRI": [2],
                "Q_NTRI": ["1_0"],
            },
            np.nan,
        ),
        # case 3: incorrect H2S and AMON, below det doxy, correct NTRZ, stb use AMON
        (
            {
                "H2S": [5],
                "Q_H2S": ["4_0"],
                "AMON": [10],
                "Q_AMON": ["4_0"],
                "doxy": [0.5],
                "Q_doxy": ["6_0"],
                "NTRZ": [3],
                "Q_NTRZ": ["1_0"],
                "NTRA": [3],
                "Q_NTRA": ["4_0"],
                "NTRI": [2],
                "Q_NTRI": ["1_0"],
            },
            np.nan,
        ),
        # case 4: low correct doxy, no AMON data, set to nan
        (
            {
                "H2S": [np.nan],
                "Q_H2S": ["1_0"],
                "AMON": [np.nan],
                "Q_AMON": ["1_0"],
                "doxy": [1],
                "Q_doxy": ["1_0"],
                "NTRZ": [3],
                "Q_NTRZ": ["1_0"],
                "NTRA": [3],
                "Q_NTRA": ["4_0"],
                "NTRI": [2],
                "Q_NTRI": ["1_0"],
            },
            np.nan,
        ),
        # case 5: low correct doxy, AMON below det, set to sum of all
        (
            {
                "H2S": [np.nan],
                "Q_H2S": ["1_0"],
                "AMON": [1],
                "Q_AMON": ["6_0"],
                "doxy": [1],
                "Q_doxy": ["1_0"],
                "NTRZ": [3],
                "Q_NTRZ": ["1_0"],
                "NTRA": [3],
                "Q_NTRA": ["4_0"],
                "NTRI": [2],
                "Q_NTRI": ["1_0"],
            },
            4,
        ),
        # case 6: correct doxy, AMON, NTRA, NTRI, no NTRZ
        (
            {
                "H2S": [np.nan],
                "Q_H2S": ["0_0"],
                "AMON": [5],
                "Q_AMON": ["1_0"],
                "doxy": [5],
                "Q_doxy": ["1_0"],
                "NTRZ": [np.nan],
                "Q_NTRZ": ["1_0"],
                "NTRA": [3],
                "Q_NTRA": ["1_0"],
                "NTRI": [2],
                "Q_NTRI": ["1_0"],
            },
            10,
        ),
        # case 7: as case 6 but NTRI nan, return AMON+NTRA
        (
            {
                "H2S": [np.nan],
                "Q_H2S": ["0_0"],
                "AMON": [5],
                "Q_AMON": ["1_0"],
                "doxy": [5],
                "Q_doxy": ["1_0"],
                "NTRZ": [np.nan],
                "Q_NTRZ": ["1_0"],
                "NTRA": [3],
                "Q_NTRA": ["1_0"],
                "NTRI": [np.nan],
                "Q_NTRI": ["1_0"],
            },
            8,
        ),
        # case 8: all valid, no H2S, return AMON+NTRZ
        (
            {
                "H2S": [np.nan],
                "Q_H2S": ["0_0"],
                "AMON": [5],
                "Q_AMON": ["1_0"],
                "doxy": [5],
                "Q_doxy": ["1_0"],
                "NTRZ": [10],
                "Q_NTRZ": ["1_0"],
                "NTRA": [7],
                "Q_NTRA": ["1_0"],
                "NTRI": [2],
                "Q_NTRI": ["1_0"],
            },
            15,
        ),
        # case 9: all valid no NTRZ, no H2S, return AMON+NTRA+NTRI
        (
            {
                "H2S": [np.nan],
                "Q_H2S": ["0_0"],
                "AMON": [5],
                "Q_AMON": ["1_0"],
                "doxy": [5],
                "Q_doxy": ["1_0"],
                "NTRZ": [np.nan],
                "Q_NTRZ": ["1_0"],
                "NTRA": [7],
                "Q_NTRA": ["1_0"],
                "NTRI": [2],
                "Q_NTRI": ["1_0"],
            },
            14,
        ),
        # case 10: AMON below det, no H2S, return NTRZ
        (
            {
                "H2S": [np.nan],
                "Q_H2S": ["0_0"],
                "AMON": [1],
                "Q_AMON": ["6_0"],
                "doxy": [5],
                "Q_doxy": ["1_0"],
                "NTRZ": [11],
                "Q_NTRZ": ["1_0"],
                "NTRA": [7],
                "Q_NTRA": ["1_0"],
                "NTRI": [2],
                "Q_NTRI": ["1_0"],
            },
            11,
        ),
    ),
)
def test_din(given_data, expected_din):
    data = pd.DataFrame(given_data)

    calculate.dissolved_inorganic_nitrogen(data)
    # test function against expected
    np.testing.assert_equal(data["din"].values[0], expected_din)

    din = calculate.get_DIN(given_data)
    # test stb against expected
    np.testing.assert_equal(din, expected_din)


@pytest.mark.parametrize(
    "given_data, expected_din",
    (
        # case 1: H2S and AMON valid, use only AMON
        (
            {
                "H2S": [np.nan, np.nan],
                "Q_H2S": ["1_0", "1_0"],
                "AMON": [1, 3],
                "Q_AMON": ["6_0", "1_0"],
                "doxy": [6, 6],
                "Q_doxy": ["1_0", "1_0"],
                "NTRZ": [np.nan, np.nan],
                "Q_NTRZ": ["1_0", "1_0"],
                "NTRA": [2, 2],
                "Q_NTRA": ["1_0", "1_0"],
                "NTRI": [1, 1],
                "Q_NTRI": ["1_0", "1_0"],
            },
            {"din": [3.0, 6.0]},
        ),
    )
)
def test_din_return_row_sum(given_data, expected_din):
    data = pd.DataFrame(given_data)
    expected_data = pd.DataFrame(expected_din)

    calculate.dissolved_inorganic_nitrogen(data)
    print(data)
    # test function against expected
    np.testing.assert_equal(data["din"].values[0], expected_data["din"].values[0])
    np.testing.assert_equal(data["din"].values[1], expected_data["din"].values[1])


@pytest.mark.parametrize(
    "given_data, expected_oxysat",
    (
        # case 1: all valid
        (
            {
                "doxy": [5],
                "temp": [10],
                "salt": [30],
                "depth": 0
            },
            76.559,
        ),
         # case 2: all valid
        (
            {
                "doxy": [5],
                "temp": [10],
                "salt": [30],
                "depth": 500
            },
            76.559,
        ),
        # case 3: one is nan
        (
            {
                "doxy": [5],
                "temp": [10],
                "salt": [np.nan],
                "depth": 0
            },
            np.nan,
        ),
    ),
)
def test_oxyen_saturation(given_data, expected_oxysat):
    data = pd.DataFrame(given_data)

    gsw, sw, _ = calculate.oxygen_saturation(data)

    result = float("{:.3f}".format(data["oxygen_saturation"].values[0]))
    # test gsw against expected
    np.testing.assert_equal(result, expected_oxysat)
    # test gsw against sw
    np.testing.assert_equal(float("{:.1f}".format(gsw[0])), float("{:.1f}".format(sw[0])))


@pytest.mark.parametrize(
    "given_data, expected",
    (
        # case 1: all valid
        (
            {
                "doxy": [5, 6, 7],
                "temp": [10, 15, 20],
                "salt": [30, 31, 35],
                "depth": [0, 5, 10]
            },
            [76.559275, 102.671144, 135.297787]
        ),
    )
)
def test_oxygen_saturation_on_dataframe_with_many_rows(given_data, expected):
    data = pd.DataFrame(given_data)

    _, _, _ = calculate.oxygen_saturation(data)
    print(data.head())
    assert(len(data['oxygen_saturation']) == len(data['doxy']))
