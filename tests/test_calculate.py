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
                "DOXY_BTL": [5],
                "Q_DOXY_BTL": ["1_0"],
                "NTRZ": [np.nan],
                "Q_NTRZ": ["1_0"],
                "NTRA": [2],
                "Q_NTRA": ["1_0"],
                "NTRI": [1],
                "Q_NTRI": ["1_0"],
            },
            3.0,
        ),
        # case 2: no H2S or AMON, below det DOXY_BTL, correct NTRZ, stb use AMON
        (
            {
                "H2S": [np.nan],
                "Q_H2S": ["4_0"],
                "AMON": [np.nan],
                "Q_AMON": ["4_0"],
                "DOXY_BTL": [0.5],
                "Q_DOXY_BTL": ["6_0"],
                "NTRZ": [3],
                "Q_NTRZ": ["1_0"],
                "NTRA": [np.nan],
                "Q_NTRA": ["4_0"],
                "NTRI": [2],
                "Q_NTRI": ["1_0"],
            },
            np.nan,
        ),
        # case 3: incorrect H2S and AMON, below det DOXY_BTL, correct NTRZ, stb use AMON
        (
            {
                "H2S": [5],
                "Q_H2S": ["4_0"],
                "AMON": [10],
                "Q_AMON": ["4_0"],
                "DOXY_BTL": [0.5],
                "Q_DOXY_BTL": ["6_0"],
                "NTRZ": [3],
                "Q_NTRZ": ["1_0"],
                "NTRA": [3],
                "Q_NTRA": ["4_0"],
                "NTRI": [2],
                "Q_NTRI": ["1_0"],
            },
            np.nan,
        ),
        # case 4: low correct DOXY_BTL, no AMON data, set to nan
        (
            {
                "H2S": [np.nan],
                "Q_H2S": ["1_0"],
                "AMON": [np.nan],
                "Q_AMON": ["1_0"],
                "DOXY_BTL": [1],
                "Q_DOXY_BTL": ["1_0"],
                "NTRZ": [3],
                "Q_NTRZ": ["1_0"],
                "NTRA": [3],
                "Q_NTRA": ["4_0"],
                "NTRI": [2],
                "Q_NTRI": ["1_0"],
            },
            np.nan,
        ),
        # case 5: low correct DOXY_BTL, AMON below det, set to sum of all (AMON+NTRZ or AMON+NTRI+NTRA)
        # this test differs to sharktoolbox get_din() which returns 3 (NTRZ) while the new function returns (AMON+NTRA) 
        (
            {
                "H2S": [np.nan],
                "Q_H2S": ["1_0"],
                "AMON": [1],
                "Q_AMON": ["6_0"],
                "DOXY_BTL": [1],
                "Q_DOXY_BTL": ["1_0"],
                "NTRZ": [3],
                "Q_NTRZ": ["1_0"],
                "NTRA": [3],
                "Q_NTRA": ["4_0"],
                "NTRI": [2],
                "Q_NTRI": ["1_0"],
            },
            4,
        ),
        # case 6: correct DOXY_BTL, AMON, NTRA, NTRI, no NTRZ
        (
            {
                "H2S": [np.nan],
                "Q_H2S": ["0_0"],
                "AMON": [5],
                "Q_AMON": ["1_0"],
                "DOXY_BTL": [5],
                "Q_DOXY_BTL": ["1_0"],
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
                "DOXY_BTL": [5],
                "Q_DOXY_BTL": ["1_0"],
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
                "DOXY_BTL": [5],
                "Q_DOXY_BTL": ["1_0"],
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
                "DOXY_BTL": [5],
                "Q_DOXY_BTL": ["1_0"],
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
                "DOXY_BTL": [5],
                "Q_DOXY_BTL": ["1_0"],
                "NTRZ": [11],
                "Q_NTRZ": ["1_0"],
                "NTRA": [7],
                "Q_NTRA": ["1_0"],
                "NTRI": [2],
                "Q_NTRI": ["1_0"],
            },
            11,
        ),
        # case 11: AMON, NTRZ, NTRA, NTRI below det return NTRZ det
        (
            {
                "H2S": [np.nan],
                "Q_H2S": ["0_0"],
                "AMON": [0.5],
                "Q_AMON": ["6_0"],
                "DOXY_BTL": [8],
                "Q_DOXY_BTL": ["1_0"],
                "NTRZ": [4],
                "Q_NTRZ": ["6_0"],
                "NTRA": [3],
                "Q_NTRA": ["6_0"],
                "NTRI": [2],
                "Q_NTRI": ["6_0"],
            },
            4,
        ),
    ),
)
def test_din(given_data, expected_din):
    data = pd.DataFrame(given_data)

    calculate.dissolved_inorganic_nitrogen(data)
    # test function against expected
    np.testing.assert_equal(data["din"].values[0], expected_din)

    # turn on to test against sharktoolbox get_din function
    # din = calculate._get_DIN(given_data)
    # # test stb against expected
    # np.testing.assert_equal(din, expected_din)


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
                "DOXY_BTL": [6, 6],
                "Q_DOXY_BTL": ["1_0", "1_0"],
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
                "oxygen": [5],
                "temp": [10],
                "salt": [30],
                "depth": 0
            },
            76.559,
        ),
         # case 2: all valid
        (
            {
                "oxygen": [5],
                "temp": [10],
                "salt": [30],
                "depth": 500
            },
            76.559,
        ),
        # case 3: one is nan
        (
            {
                "oxygen": [5],
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
                "oxygen": [5, 6, 7],
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
    assert(len(data['oxygen_saturation']) == len(data['oxygen']))

@pytest.mark.parametrize(
    "given_data, expected_o2",
    (
        # case 1: H2S and o2 BTL valid, use default H2S (0)
        (
            {
                "H2S": [5],
                "Q_H2S": ["1_0"],
                "DOXY_BTL": [2],
                "Q_DOXY_BTL": ["1_0"],                
                "DOXY_CTD": [np.nan],
                "Q_DOXY_CTD": ["1_0"],
            },
            0,
        ),
        # case 2: H2S invalid (S or B) and o2 <, use default H2S (0)
        (
            {
                "H2S": [5],
                "Q_H2S": ["S_0"],
                "DOXY_BTL": [0.5],
                "Q_DOXY_BTL": ["<_0"],
                "DOXY_CTD": [np.nan],
                "Q_DOXY_CTD": ["1_0"],
            },
            0.5,
        ),
        # case 3: H2S < and o2 <, use default H2S (0)
        (
            {
                "H2S": [5],
                "Q_H2S": ["<_0"],
                "DOXY_BTL": [0.5],
                "Q_DOXY_BTL": ["<_0"],
                "DOXY_CTD": [np.nan],
                "Q_DOXY_CTD": ["1_0"],
            },
            0,
        ),
        # case 4: H2S < and o2 valid, use o2
        (
                {
                    "H2S": [5],
                    "Q_H2S": ["<_0"],
                    "DOXY_BTL": [0.5],
                    "Q_DOXY_BTL": ["1_0"],
                    "DOXY_CTD": [np.nan],
                    "Q_DOXY_CTD": ["1_0"],
                },
                0.5,
        ),
        # case 5: H2S is nan and o2 valid, use o2
        (
                {
                    "H2S": [np.nan],
                    "Q_H2S": ["1_0"],
                    "DOXY_BTL": [0.5],
                    "Q_DOXY_BTL": ["1_0"],
                    "DOXY_CTD": [np.nan],
                    "Q_DOXY_CTD": ["1_0"],
                },
                0.5,
        ),
        # case 6: H2S is valid and o2 nan, use H2S default (0)
        (
                {
                    "H2S": [5],
                    "Q_H2S": ["1_0"],
                    "DOXY_BTL": [np.nan],
                    "Q_DOXY_BTL": ["1_0"],
                    "DOXY_CTD": [np.nan],
                    "Q_DOXY_CTD": ["1_0"],
                },
                0,
        ),
        # case 7: o2 BTL is valid and o2 CTD is valid, use o2 BTL
        (
                {
                    "H2S": [np.nan],
                    "Q_H2S": ["1_0"],
                    "DOXY_BTL": [5],
                    "Q_DOXY_BTL": ["1_0"],
                    "DOXY_CTD": [10],
                    "Q_DOXY_CTD": ["1_0"],
                },
                5.0,
        ),
        # case 8: o2 BTL is not valid and o2 CTD is valid, use o2 CTD
        (
                {
                    "H2S": [np.nan],
                    "Q_H2S": ["1_0"],
                    "DOXY_BTL": [5],
                    "Q_DOXY_BTL": ["S"],
                    "DOXY_CTD": [10],
                    "Q_DOXY_CTD": ["1_0"],
                },
                10,
        ),
        # case 9: o2 BTL valid and o2 CTD is not valid, use o2 BTL
        (
                {
                    "H2S": [np.nan],
                    "Q_H2S": ["1_0"],
                    "DOXY_BTL": [5],
                    "Q_DOXY_BTL": ["1_0"],
                    "DOXY_CTD": [10],
                    "Q_DOXY_CTD": ["S"],
                },
                5.0,
        ),
        # case 10: all are non valid
        (
                {
                    "H2S": [np.nan],
                    "Q_H2S": ["1_0"],
                    "DOXY_BTL": [np.nan],
                    "Q_DOXY_BTL": ["1_0"],
                    "DOXY_CTD": [np.nan],
                    "Q_DOXY_CTD": ["1_0"],
                },
                np.nan,
        ),
        # case 11: H2S valid, o2 btl nan and o2 CTD valid
        (
                {
                    "H2S": [5],
                    "Q_H2S": ["1_0"],
                    "DOXY_BTL": [np.nan],
                    "Q_DOXY_BTL": ["1_0"],
                    "DOXY_CTD": [10],
                    "Q_DOXY_CTD": ["1_0"],
                },
                0,
        ),
        # case 12: H2S invalid, o2 btl nan and o2 <CTD valid
        (
                {
                    "H2S": [5],
                    "Q_H2S": ["S_0"],
                    "DOXY_BTL": [np.nan],
                    "Q_DOXY_BTL": ["1_0"],
                    "DOXY_CTD": [10],
                    "Q_DOXY_CTD": ["<_0"],
                },
                0,
        ),
        # case 13: H2S valid and >, o2 btl valid and o2 <CTD valid gives H2S 0
        (
                {
                    "H2S": [100],
                    "Q_H2S": [">_0"],
                    "DOXY_BTL": [2],
                    "Q_DOXY_BTL": ["1_0"],
                    "DOXY_CTD": [10],
                    "Q_DOXY_CTD": ["1_0"],
                },
                0,
        ),
        # case 14: H2S, ctd np.nan and o2 valid < gives H2S 0
        (
                {
                    "H2S": [np.nan],
                    "Q_H2S": ["1_0"],
                    "DOXY_BTL": [2],
                    "Q_DOXY_BTL": ["<_0"],
                    "DOXY_CTD": [np.nan],
                    "Q_DOXY_CTD": ["1_0"],
                },
                2,
        ),
    ),
)
def test_oxygen(given_data, expected_o2):
    data = pd.DataFrame(given_data)

    calculate.oxygen(data)
    # test function against expected
    np.testing.assert_equal(data["oxygen"].values[0], expected_o2)