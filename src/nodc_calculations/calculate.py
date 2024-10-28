import pandas as pd
import numpy as np


def get_DIN(data: dict):
    """
    Returns a vector calculated DIN.
    If H2S is present NH4 is returned
    If NO3 is not present value is np.nan
    If no H2S and NH4 qflag is < nox is returned
    """

    no2 = data["no2"][0]
    no3 = data["no3"][0]
    nox = data["nox"][0]
    nh4 = data["nh4"][0]
    h2s = data["h2s"][0]
    qh2s = data["qh2s"][0].split("_")[0]
    o2 = data["o2"][0]
    qo2 = data["qo2"][0].split("_")[0]
    qnh4 = data["qnh4"][0].split("_")[0]
    qno3 = data["qno3"][0].split("_")[0]

    print(qo2)

    # LENA added: and any([qh2s not in [u'6', u'4', u'3'], (qh2s == u'6' and qno3 in [u'4', u'6', u'3'])])
    if not np.isnan(h2s) and any(
        [qh2s not in ["6", "4", "3"], (qh2s == "6" and qno3 in ["4", "6", "3"])]
    ):
        if any([np.isnan(nh4), qnh4 in ["4", "3"]]):
            din = np.nan
        else:
            din = nh4
    elif o2 < 2.0 and qo2 not in ["4", "3"]:
        if any([np.isnan(nh4), qnh4 in ["4", "3"]]):
            din = np.nan
        else:
            if np.isnan(nox):
                din = np.nan

                if not np.isnan(no3):
                    if qno3 in ["4", "3"]:
                        din = np.nan
                    else:
                        din = no3

                    if not np.isnan(no2) and not np.isnan(din):
                        din += no2

                    if (
                        not np.isnan(nh4)
                        and qnh4 not in ["6", "4", "3"]
                        and not np.isnan(din)
                    ):
                        # LENA added and not np.isnan(nh4) and and qnh4 not in [u'6', u'4', u'3']
                        din += nh4
            else:
                din = nox

                if not np.isnan(nh4) and qnh4 not in ["6", "4", "3"]:
                    # LENA added and not np.isnan(nh4) and qnh4 not in [u'6', u'4' ,u'3']
                    din += nh4
    else:
        if np.isnan(nox):
            din = np.nan

            if not np.isnan(no3):
                if qno3 in ["4", "3"]:
                    din = np.nan
                else:
                    din = no3

                if not np.isnan(no2) and not np.isnan(din):
                    din += no2

                if (
                    not np.isnan(nh4)
                    and qnh4 not in ["6", "4", "3"]
                    and not np.isnan(din)
                ):
                    # LENA added and not np.isnan(nh4) and and qnh4 not in [u'6', u'4', u'3']
                    din += nh4
        else:
            din = nox

            if not np.isnan(nh4) and qnh4 not in ["6", "4", "3"]:
                # LENA added and not np.isnan(nh4) and qnh4 not in [u'6', u'4' ,u'3']
                din += nh4

    return float(din)


def dissolved_inorganic_nitrogen(df: pd.DataFrame):
    """
    Returns a vector calculated DIN.
    If H2S is present NH4 is returned
    If NO3 or NOx is not present value is np.nan
    If no H2S and NH4 qflag is < NOx is returned
    """
    valid_h2s = np.logical_and(~pd.isna(df.h2s), ~df.qh2s.str.contains("6|4|3"))
    valid_nox = np.logical_and(~pd.isna(df.nox), ~df.qnox.str.contains("6|4|3"))
    below_det_nox = np.logical_and(~pd.isna(df.nox), df.qnox.str.contains("6"))
    valid_no3 = np.logical_and(~pd.isna(df.no3), ~df.qno3.str.contains("4|3"))
    below_det_no3 = np.logical_and(~pd.isna(df.no3), df.qno3.str.contains("6"))
    valid_no2 = np.logical_and(~pd.isna(df.no2), ~df.qno2.str.contains("4|3"))
    below_det_no2 = np.logical_and(~pd.isna(df.no2), df.qno2.str.contains("6"))
    valid_nh4 = np.logical_and(~pd.isna(df.nh4), ~df.qnh4.str.contains("4|3"))
    below_det_nh4 = np.logical_and(~pd.isna(df.nh4), df.qnh4.str.contains("6"))
    valid_low_o2 = np.logical_and(df.o2 <= 2, ~df.qo2.str.contains("4|3"))
    # where valid_h2s and valid_nh4 din = nh4
    # where valid_o2 din = sum of nh4, no3, no2 at valid_din
    # Initialisera din-kolumnen
    print(df)
    df["din"] = np.nan

    df.loc[:, "din"] = np.where(
        np.logical_and(valid_h2s, valid_nh4),
        df.nh4,
        np.where(
            np.logical_or((below_det_nh4 & below_det_no2 & below_det_no3),(below_det_nh4 & below_det_nox)),
            df.nh4+df.nox,
            np.where(
                np.logical_and(valid_low_o2, ~valid_nh4),
                np.nan,
                np.where(
                    valid_nox & valid_nh4,
                    df.nox + df.nh4,
                    np.where(
                        valid_no3 & valid_no2 & valid_nh4,
                        (df.no3 + df.no2 + df.nh4),
                        np.where(
                            valid_nox,
                            df.nox,
                            np.where(
                                valid_no3 & valid_no2,
                                df.no3 + df.no2,
                                np.where(
                                    valid_no3 & valid_nh4, df.no3 + df.nh4, np.nan
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
