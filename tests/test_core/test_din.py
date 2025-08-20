import numpy as np
import pytest
from nodc_calculations.core.calculations.din import dissolved_inorganic_nitrogen_core


@pytest.mark.parametrize(
    "input_rows, expected_din",
    [
        # Case 1: H2S and AMON valid, use only AMON
        (
            [
                {
                    "H2S": 5,
                    "Q_H2S": "1_0",
                    "AMON": 3,
                    "Q_AMON": "1_0",
                    "DOXY_BTL": 5,
                    "Q_DOXY_BTL": "1_0",
                    "NTRZ": np.nan,
                    "Q_NTRZ": "1_0",
                    "NTRA": 2,
                    "Q_NTRA": "1_0",
                    "NTRI": 1,
                    "Q_NTRI": "1_0",
                },
            ],
            [3.0],
        ),
        # case 2: no H2S or AMON, below det DOXY_BTL, correct NTRZ, stb use AMON
        (
            [
                {
                    "H2S": np.nan,
                    "Q_H2S": "4_0",
                    "AMON": np.nan,
                    "Q_AMON": "4_0",
                    "DOXY_BTL": 0.5,
                    "Q_DOXY_BTL": "6_0",
                    "NTRZ": 3,
                    "Q_NTRZ": "1_0",
                    "NTRA": np.nan,
                    "Q_NTRA": "4_0",
                    "NTRI": 2,
                    "Q_NTRI": "1_0",
                },
            ],
            [None],
        ),
        # case 3: incorrect H2S and AMON, below det DOXY_BTL, correct NTRZ, stb use AMON
        (
            [
                {
                    "H2S": 5,
                    "Q_H2S": "4_0",
                    "AMON": 10,
                    "Q_AMON": "4_0",
                    "DOXY_BTL": 0.5,
                    "Q_DOXY_BTL": "6_0",
                    "NTRZ": 3,
                    "Q_NTRZ": "1_0",
                    "NTRA": 3,
                    "Q_NTRA": "4_0",
                    "NTRI": 2,
                    "Q_NTRI": "1_0",
                },
            ],
            [None],
        ),
        # case 4: low correct DOXY_BTL, no AMON data, set to nan
        (
            [
                {
                    "H2S": np.nan,
                    "Q_H2S": "1_0",
                    "AMON": np.nan,
                    "Q_AMON": "1_0",
                    "DOXY_BTL": 1,
                    "Q_DOXY_BTL": "1_0",
                    "NTRZ": 3,
                    "Q_NTRZ": "1_0",
                    "NTRA": 3,
                    "Q_NTRA": "4_0",
                    "NTRI": 2,
                    "Q_NTRI": "1_0",
                },
            ],
            [None],
        ),
        # case 5: low correct DOXY_BTL, AMON below det, set to sum of all (AMON+NTRZ or AMON+NTRI+NTRA)
        # this test differs to sharktoolbox get_din() which returns 3 (NTRZ) while the new function returns (AMON+NTRA)
        (
            [
                {
                    "H2S": np.nan,
                    "Q_H2S": "1_0",
                    "AMON": 1,
                    "Q_AMON": "6_0",
                    "DOXY_BTL": 1,
                    "Q_DOXY_BTL": "1_0",
                    "NTRZ": 3,
                    "Q_NTRZ": "1_0",
                    "NTRA": 3,
                    "Q_NTRA": "4_0",
                    "NTRI": 2,
                    "Q_NTRI": "1_0",
                },
            ],
            [4],
        ),
        # case 6: correct DOXY_BTL, AMON, NTRA, NTRI, no NTRZ
        (
            [
                {
                    "H2S": np.nan,
                    "Q_H2S": "0_0",
                    "AMON": 5,
                    "Q_AMON": "1_0",
                    "DOXY_BTL": 5,
                    "Q_DOXY_BTL": "1_0",
                    "NTRZ": np.nan,
                    "Q_NTRZ": "1_0",
                    "NTRA": 3,
                    "Q_NTRA": "1_0",
                    "NTRI": 2,
                    "Q_NTRI": "1_0",
                },
            ],
            [
                10,
            ],
        ),
        # case 7: as case 6 but NTRI nan, return AMON+NTRA
        (
            [
                {
                    "H2S": np.nan,
                    "Q_H2S": "0_0",
                    "AMON": 5,
                    "Q_AMON": "1_0",
                    "DOXY_BTL": 5,
                    "Q_DOXY_BTL": "1_0",
                    "NTRZ": np.nan,
                    "Q_NTRZ": "1_0",
                    "NTRA": 3,
                    "Q_NTRA": "1_0",
                    "NTRI": np.nan,
                    "Q_NTRI": "1_0",
                },
            ],
            [8],
        ),
        # case 8: all valid, no H2S, return AMON+NTRZ
        (
            [
                {
                    "H2S": np.nan,
                    "Q_H2S": "0_0",
                    "AMON": 5,
                    "Q_AMON": "1_0",
                    "DOXY_BTL": 5,
                    "Q_DOXY_BTL": "1_0",
                    "NTRZ": 10,
                    "Q_NTRZ": "1_0",
                    "NTRA": 7,
                    "Q_NTRA": "1_0",
                    "NTRI": 2,
                    "Q_NTRI": "1_0",
                },
            ],
            [15],
        ),
        # case 9: all valid no NTRZ, no H2S, return AMON+NTRA+NTRI
        (
            [
                {
                    "H2S": np.nan,
                    "Q_H2S": "0_0",
                    "AMON": 5,
                    "Q_AMON": "1_0",
                    "DOXY_BTL": 5,
                    "Q_DOXY_BTL": "1_0",
                    "NTRZ": np.nan,
                    "Q_NTRZ": "1_0",
                    "NTRA": 7,
                    "Q_NTRA": "1_0",
                    "NTRI": 2,
                    "Q_NTRI": "1_0",
                },
            ],
            [14],
        ),
        # case 10: AMON below det, no H2S, return NTRZ
        (
            [
                {
                    "H2S": np.nan,
                    "Q_H2S": "0_0",
                    "AMON": 1,
                    "Q_AMON": "6_0",
                    "DOXY_BTL": 5,
                    "Q_DOXY_BTL": "1_0",
                    "NTRZ": 11,
                    "Q_NTRZ": "1_0",
                    "NTRA": 7,
                    "Q_NTRA": "1_0",
                    "NTRI": 2,
                    "Q_NTRI": "1_0",
                },
            ],
            [11],
        ),
        # case 11: AMON, NTRZ, NTRA, NTRI below det return NTRZ det
        (
            [
                {
                    "H2S": np.nan,
                    "Q_H2S": "0_0",
                    "AMON": 0.5,
                    "Q_AMON": "6_0",
                    "DOXY_BTL": 8,
                    "Q_DOXY_BTL": "1_0",
                    "NTRZ": 4,
                    "Q_NTRZ": "6_0",
                    "NTRA": 3,
                    "Q_NTRA": "6_0",
                    "NTRI": 2,
                    "Q_NTRI": "6_0",
                },
            ],
            [4],
        ),
    ],
)
def test_din_single_row(input_rows, expected_din):
    """Test DIN calculation for a single row where H2S and AMON are valid."""
    din_result = dissolved_inorganic_nitrogen_core(input_rows)
    print(input_rows)
    np.testing.assert_equal(din_result, expected_din)


@pytest.mark.parametrize(
    "input_rows, expected_din",
    [
        # Case 2: Two rows, second row has AMON valid, first AMON below detection
        (
            [
                {
                    "H2S": np.nan,
                    "Q_H2S": "1_0",
                    "AMON": 1,
                    "Q_AMON": "6_0",
                    "DOXY_BTL": 6,
                    "Q_DOXY_BTL": "1_0",
                    "NTRZ": np.nan,
                    "Q_NTRZ": "1_0",
                    "NTRA": 2,
                    "Q_NTRA": "1_0",
                    "NTRI": 1,
                    "Q_NTRI": "1_0",
                },
                {
                    "H2S": np.nan,
                    "Q_H2S": "1_0",
                    "AMON": 3,
                    "Q_AMON": "1_0",
                    "DOXY_BTL": 6,
                    "Q_DOXY_BTL": "1_0",
                    "NTRZ": np.nan,
                    "Q_NTRZ": "1_0",
                    "NTRA": 2,
                    "Q_NTRA": "1_0",
                    "NTRI": 1,
                    "Q_NTRI": "1_0",
                },
            ],
            [3.0, 6.0],
        ),
    ],
)
def test_din_multiple_rows(input_rows, expected_din):
    """Test DIN calculation for multiple rows with mixed valid/invalid AMON."""
    print(type(input_rows))
    assert isinstance(input_rows, list)
    din_result = dissolved_inorganic_nitrogen_core(input_rows)
    np.testing.assert_equal(din_result, expected_din)


"""
Case-by-Case Mapping
    - H₂S & AMON valid → Rule 1 → AMON

    - Low DOXY, valid AMON → Rule 2 → NTRZ_corr + AMON (AMON may be nan, so sum handles it)

    - Incorrect H₂S & AMON, low DOXY, NTRZ_corr valid → Rule 3 → NTRZ_corr

    - Low DOXY, no AMON data → None of the first 4 rules, result nan (no valid NTRZ_corr + AMON possible)

    - Low DOXY, AMON below det → Rule 2 → NTRZ_corr + AMON

    - Correct DOXY, AMON, NTRA, NTRI, no NTRZ → Rule 2 (low DOXY assumed here) → NTRZ_corr + AMON

    - As case 6 but NTRI nan → Rule 2 → NTRZ_corr + AMON

    - All valid, no H₂S → Rule 4 → NTRZ_corr + AMON

    - All valid, no NTRZ → Rule 4 → NTRZ_corr + AMON

    - AMON below det, no H₂S → Rule 3 → NTRZ_corr

    - AMON valid, NTRZ/NTRA/NTRI below det → Rule 3 → NTRZ_corr

    - H₂S nan, AMON below det, valid DOXY, nan NTRZ, valid NTRA & NTRI → Rule 3 → NTRZ_corr (= NTRA+NTRI)

    - As 12 but AMON valid → Rule 4 → NTRZ_corr + AMON
"""
