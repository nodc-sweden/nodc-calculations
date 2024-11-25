from seawater import satO2
from gsw import O2sol_SP_pt, O2sol, pot_rho_t_exact
from gsw.conversions import p_from_z, t90_from_t68, pt_from_CT
from gsw.density import rho
import pandas as pd
import numpy as np
from nodc_calculations.convert import oxygen_ml2umol

def DIVA_oxygen(data: pd.DataFrame):

    valid_btl = np.logical_and(~pd.isna(data.o2_btl), ~data.qo2_btl.str.contains("B|S|<"))
    below_det_btl = np.logical_and(~pd.isna(data.o2_btl), data.qo2_btl.str.contains("<"))
    valid_ctd = np.logical_and(~pd.isna(data.o2_ctd), ~data.qo2_ctd.str.contains("B|S|<"))
    below_det_ctd = np.logical_and(~pd.isna(data.o2_ctd), data.qo2_ctd.str.contains("<"))
    valid_h2s = np.logical_and(~pd.isna(data.h2s), ~data.qh2s.str.contains("B|S|Z|<"))
    below_det_h2s = np.logical_and(~pd.isna(data.h2s), data.qh2s.str.contains("<"))

    # Apply nested np.where for all conditions
    data["o2"] = np.where(
        # h2s valid-> h2s default (0.01)
        valid_h2s, 0.01,
            # h2s not valid and o2< gives h2s default (0.01)
            np.where((~valid_h2s) & (below_det_btl), 0.01,
                #  o2 BTL is valid gives o2 BTL
                np.where((valid_btl),
                    data.o2_btl,
                    # O2 CTD exists and O2 CTD is not S gives O2 CTD
                    np.where((valid_ctd),
                        data.o2_ctd,
                            np.where((below_det_ctd),0.01,

                            # Default case gives NaN
                            np.nan
                        )
                    )
                )
            )
        )
    return data

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
                        din += nh4
            else:
                din = nox

                if not np.isnan(nh4) and qnh4 not in ["6", "4", "3"]:
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
                    din += nh4
        else:
            din = nox

            if not np.isnan(nh4) and qnh4 not in ["6", "4", "3"]:
                din += nh4

    return float(din)


def dissolved_inorganic_nitrogen(df: pd.DataFrame):
    """
    Calculates DIN values based on nitrogen components, oxygen and hydroggen sulphide and quality flags
    """

    # define booleans for valid data and lmtq for nox, no3, no2
    valid_no3 = np.logical_and(~pd.isna(df.no3), ~df.qno3.str.contains("4|3"))
    below_det_no3 = np.logical_and(~pd.isna(df.no3), df.qno3.str.contains("6"))

    valid_no2 = np.logical_and(~pd.isna(df.no2), ~df.qno2.str.contains("4|3"))
    below_det_no2 = np.logical_and(~pd.isna(df.no2), df.qno2.str.contains("6"))

    valid_nox = np.logical_and(~pd.isna(df.nox), ~df.qnox.str.contains("4|3"))
    below_det_nox = np.logical_and(~pd.isna(df.nox), df.qnox.str.contains("6"))

    # Create noc column from no3+no2 when nox not valid
    df["nox_corrected"] = np.where(
        pd.isna(df.nox)
        & below_det_no3
        & below_det_no2,  # both below lmtq
        df.no3,
        np.where(
            pd.isna(df.nox) & valid_no3,  # at least no3 valid
            np.nansum([df.no3, df.no2]),
            np.where(
                valid_nox,
                df.nox,  # nox valid
                np.nan,
            ),
        ),
    )

    # define booleans for valid data and lmtq for other parameters
    valid_nox_corrected = ~pd.isna(df.nox_corrected)
    valid_h2s = np.logical_and(~pd.isna(df.h2s), ~df.qh2s.str.contains("6|4|3"))
    valid_nh4 = np.logical_and(~pd.isna(df.nh4), ~df.qnh4.str.contains("4|3"))
    below_det_nh4 = np.logical_and(~pd.isna(df.nh4), df.qnh4.str.contains("6"))
    valid_low_o2 = np.logical_and(df.o2 <= 2, ~df.qo2.str.contains("4|3"))

    df["din"] = np.nan

    # Fall där H2S är giltigt och NH4 är giltigt
    df.loc[valid_h2s & valid_nh4, "din"] = df.nh4

    # Steg 2: I låga syrehalter beräkna din endast
    # om nh4 finns, antingen som summa nh4+nox_corrected eller endast som nh4 om nox_corrected är nan.
    df["din"] = np.where(
        valid_low_o2 & valid_nh4,
        np.nansum([df.nox_corrected, df.nh4], axis=0),
        df.din,  # Om inget av ovanstående gäller, lämna som `din`
    )

    # Typiskt sommaren när alla är under det
    df["din"] = np.where(
        (below_det_nox | below_det_no3)
        & below_det_nh4
        & ~pd.isna(df.nox_corrected),  # Använd nox_corrected + nh4 om båda är giltiga
        df.nox_corrected,
        df.din,  # Om inget av ovanstående gäller, lämna som `din`
    )

    # Övriga fall där nox används som huvudsaklig parameter
    df["din"] = np.where(
        valid_nh4
        & ~valid_low_o2
        & ~valid_h2s
        & ~below_det_nh4
        & ~pd.isna(df.nox_corrected),  # Använd nox_corrected + nh4 om båda är giltiga
        df.nox_corrected + df.nh4,
        np.where(
            ~valid_low_o2 & ~valid_h2s & below_det_nh4 & ~pd.isna(df.nox_corrected),
            df.nox_corrected,
            df.din,
        ),  # Om inget av ovanstående gäller, lämna som `din`
    )

    return df


def density(df: pd.DataFrame):
    """
    the sea pressure calculated from depth and latitude has very little effect on the results
    use constant latitude, comnsider using constant z as well
    """
    df.loc[:, "density"]  = pot_rho_t_exact(df.SALT, df.TEMP, p_from_z(-df.DEPH, 58), 0)



def oxygen_saturation(df: pd.DataFrame):
    # oxygen_ml2umol(df, oxygen_column_name=oxygen_column_name)
    pt = pt_from_CT(df.SALT, df.TEMP)
    density(df)
    gsw = O2sol_SP_pt(df.SALT, pt) * (df.density/1000) / 44.661
    sw = satO2(df.SALT, df.TEMP)

    df.loc[:, f"oxygen_saturation"] = df.DOXY_BTL / gsw * 100

    return gsw, sw, df
