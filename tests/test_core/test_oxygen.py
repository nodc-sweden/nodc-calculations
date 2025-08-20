import numpy as np
import pytest
from nodc_calculations.core.calculations.oxygen import (
    oxygen_core,
    oxygen_saturation_core,
)


@pytest.mark.parametrize(
    "input_rows, expected_o2",
    (
        # case 1: H2S and o2 BTL valid, use default H2S (0)
        (
            [
                {
                    "H2S": 5,
                    "Q_H2S": "1_0",
                    "DOXY_BTL": 2,
                    "Q_DOXY_BTL": "1_0",
                    "DOXY_CTD": None,
                    "Q_DOXY_CTD": "1_0",
                }
            ],
            [0],
        ),
        # case 2: H2S invalid (S or B) and o2 <, use default H2S (0)
        (
            [
                {
                    "H2S": 5,
                    "Q_H2S": "S_0",
                    "DOXY_BTL": 0.5,
                    "Q_DOXY_BTL": "<_0",
                    "DOXY_CTD": None,
                    "Q_DOXY_CTD": "1_0",
                }
            ],
            [0.5],
        ),
        # case 3: H2S < and o2 <, use default H2S (0)
        (
            [
                {
                    "H2S": 5,
                    "Q_H2S": "<_0",
                    "DOXY_BTL": 0.5,
                    "Q_DOXY_BTL": "<_0",
                    "DOXY_CTD": None,
                    "Q_DOXY_CTD": "1_0",
                }
            ],
            [0],
        ),
        # case 4: H2S < and o2 valid, use o2
        (
            [
                {
                    "H2S": 5,
                    "Q_H2S": "<_0",
                    "DOXY_BTL": 0.5,
                    "Q_DOXY_BTL": "1_0",
                    "DOXY_CTD": None,
                    "Q_DOXY_CTD": "1_0",
                }
            ],
            [0.5],
        ),
        # case 5: H2S is nan and o2 valid, use o2
        (
            [
                {
                    "H2S": None,
                    "Q_H2S": "1_0",
                    "DOXY_BTL": 0.5,
                    "Q_DOXY_BTL": "1_0",
                    "DOXY_CTD": None,
                    "Q_DOXY_CTD": "1_0",
                }
            ],
            [0.5],
        ),
        # case 6: H2S is valid and o2 nan, use H2S default (0)
        (
            [
                {
                    "H2S": 5,
                    "Q_H2S": "1_0",
                    "DOXY_BTL": None,
                    "Q_DOXY_BTL": "1_0",
                    "DOXY_CTD": None,
                    "Q_DOXY_CTD": "1_0",
                }
            ],
            [0],
        ),
        # case 7: o2 BTL is valid and o2 CTD is valid, use o2 BTL
        (
            [
                {
                    "H2S": None,
                    "Q_H2S": "1_0",
                    "DOXY_BTL": 5,
                    "Q_DOXY_BTL": "1_0",
                    "DOXY_CTD": 10,
                    "Q_DOXY_CTD": "1_0",
                }
            ],
            [5.0],
        ),
        # case 8: o2 BTL is not valid and o2 CTD is valid, use o2 CTD
        (
            [
                {
                    "H2S": None,
                    "Q_H2S": "1_0",
                    "DOXY_BTL": 5,
                    "Q_DOXY_BTL": "S",
                    "DOXY_CTD": 10,
                    "Q_DOXY_CTD": "1_0",
                }
            ],
            [10],
        ),
        # case 9: o2 BTL valid and o2 CTD is not valid, use o2 BTL
        (
            [
                {
                    "H2S": None,
                    "Q_H2S": "1_0",
                    "DOXY_BTL": 5,
                    "Q_DOXY_BTL": "1_0",
                    "DOXY_CTD": 10,
                    "Q_DOXY_CTD": "S",
                }
            ],
            [5.0],
        ),
        # case 10: all are non valid
        (
            [
                {
                    "H2S": None,
                    "Q_H2S": "1_0",
                    "DOXY_BTL": None,
                    "Q_DOXY_BTL": "1_0",
                    "DOXY_CTD": None,
                    "Q_DOXY_CTD": "1_0",
                }
            ],
            [None],
        ),
        # case 11: H2S valid, o2 btl nan and o2 CTD valid
        (
            [
                {
                    "H2S": 5,
                    "Q_H2S": "1_0",
                    "DOXY_BTL": None,
                    "Q_DOXY_BTL": "1_0",
                    "DOXY_CTD": 10,
                    "Q_DOXY_CTD": "1_0",
                }
            ],
            [0],
        ),
        # case 12: H2S invalid, o2 btl nan and o2 <CTD valid
        (
            [
                {
                    "H2S": 5,
                    "Q_H2S": "S_0",
                    "DOXY_BTL": None,
                    "Q_DOXY_BTL": "1_0",
                    "DOXY_CTD": 10,
                    "Q_DOXY_CTD": "<_0",
                }
            ],
            [0],
        ),
        # case 13: H2S valid and >, o2 btl valid and o2 <CTD valid gives H2S 0
        (
            [
                {
                    "H2S": 100,
                    "Q_H2S": ">_0",
                    "DOXY_BTL": 2,
                    "Q_DOXY_BTL": "1_0",
                    "DOXY_CTD": 10,
                    "Q_DOXY_CTD": "1_0",
                }
            ],
            [0],
        ),
        # case 14: H2S, ctd None and o2 valid < gives H2S 0
        (
            [
                {
                    "H2S": None,
                    "Q_H2S": "1_0",
                    "DOXY_BTL": 2,
                    "Q_DOXY_BTL": "<_0",
                    "DOXY_CTD": None,
                    "Q_DOXY_CTD": "1_0",
                }
            ],
            [2],
        ),
    ),
)
def test_oxygen(input_rows, expected_o2):
    print(input_rows)
    result = oxygen_core(input_rows)
    print(result)
    np.testing.assert_equal(result, expected_o2)


@pytest.mark.parametrize(
    "input_rows, expected_oxysat",
    (
        # case 1: all valid
        (
            [{"oxygen": 5, "temperature": 10, "salt": 30, "DEPH": 0}],
            [76.559],
        ),
        # case 2: all valid
        (
            [{"oxygen": 5, "temperature": 10, "salt": 30, "DEPH": 500}],
            [76.559],
        ),
        # case 3: one is nan
        (
            [{"oxygen": 5, "temperature": 10, "salt": np.nan, "DEPH": 0}],
            [np.nan],
        ),
    ),
)
def test_oxyen_saturation(input_rows, expected_oxysat):
    print(input_rows)
    result = oxygen_saturation_core(input_rows, salinity_source_column="salt")
    print(result)
    np.testing.assert_equal(np.round(result, 3), np.round(expected_oxysat, 3))
