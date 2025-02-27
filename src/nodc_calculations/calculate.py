from seawater import satO2
from gsw import O2sol_SP_pt, pot_rho_t_exact
from gsw.conversions import p_from_z, pt_from_CT
import pandas as pd
import numpy as np


def _get_DIN(data: dict):
    """
    This is the function from sharktoolbox, only here for comparison
    Returns a vector calculated DIN.
    If H2S is present AMON is returned
    If NTRS is not present value is np.nan
    If no H2S and NH4 Q_flag is < NTRZ is returned
    """

    NTRI = data["NTRI"][0]
    NTRA = data["NTRA"][0]
    NTRZ = data["NTRZ"][0]
    AMON = data["AMON"][0]
    H2S = data["H2S"][0]
    Q_H2S = data["Q_H2S"][0].split("_")[0]
    doxy = data["DOXY_BTL"][0]
    Q_doxy = data["Q_DOXY_BTL"][0].split("_")[0]
    Q_AMON = data["Q_AMON"][0].split("_")[0]
    Q_NTRA = data["Q_NTRA"][0].split("_")[0]

    if not np.isnan(H2S) and any(
        [Q_H2S not in ["6", "4", "3"], (Q_H2S == "6" and Q_NTRA in ["4", "6", "3"])]
    ):
        if any([np.isnan(AMON), Q_AMON in ["4", "3"]]):
            din = np.nan
        else:
            din = AMON
    elif doxy < 2.0 and Q_doxy not in ["4", "3"]:
        if any([np.isnan(AMON), Q_AMON in ["4", "3"]]):
            din = np.nan
        else:
            if np.isnan(NTRZ):
                din = np.nan

                if not np.isnan(NTRA):
                    if Q_NTRA in ["4", "3"]:
                        din = np.nan
                    else:
                        din = NTRA

                    if not np.isnan(NTRI) and not np.isnan(din):
                        din += NTRI

                    if (
                        not np.isnan(AMON)
                        and Q_AMON not in ["6", "4", "3"]
                        and not np.isnan(din)
                    ):
                        din += AMON
            else:
                din = NTRZ

                if not np.isnan(AMON) and Q_AMON not in ["6", "4", "3"]:
                    din += AMON
    else:
        if np.isnan(NTRZ):
            din = np.nan

            if not np.isnan(NTRA):
                if Q_NTRA in ["4", "3"]:
                    din = np.nan
                else:
                    din = NTRA

                if not np.isnan(NTRI) and not np.isnan(din):
                    din += NTRI

                if (
                    not np.isnan(AMON)
                    and Q_AMON not in ["6", "4", "3"]
                    and not np.isnan(din)
                ):
                    din += AMON
        else:
            din = NTRZ

            if not np.isnan(AMON) and Q_AMON not in ["6", "4", "3"]:
                din += AMON

    return float(din)


def dissolved_inorganic_nitrogen(df: pd.DataFrame):
    """
    Calculates DIN values based on nitrogen components, oxygen and hydroggen sulphide and Q_uality flags
    """

    # define booleans for valid data and lmtQ_ for NTRZ, NTRA, NTRI
    valid_NTRA = np.logical_and(~pd.isna(df.NTRA), ~df.Q_NTRA.str.contains("4|3|B|S"))
    below_det_NTRA = np.logical_and(~pd.isna(df.NTRA), df.Q_NTRA.str.contains("6|<"))

    valid_NTRI = np.logical_and(~pd.isna(df.NTRI), ~df.Q_NTRI.str.contains("4|3|B|S"))
    below_det_NTRI = np.logical_and(~pd.isna(df.NTRI), df.Q_NTRI.str.contains("6|<"))

    valid_NTRZ = np.logical_and(~pd.isna(df.NTRZ), ~df.Q_NTRZ.str.contains("4|3|B|S"))
    below_det_NTRZ = np.logical_and(~pd.isna(df.NTRZ), df.Q_NTRZ.str.contains("6|<"))

    # Create NTRZ column from NTRA+NTRI when NTRZ not valid
    df["NTRZ_corrected"] = np.where(
        pd.isna(df.NTRZ) & below_det_NTRA & below_det_NTRI,  # both below lmtQ_
        df.NTRA,
        np.where(
            pd.isna(df.NTRZ) & valid_NTRA,  # at least NTRA valid
            np.nansum([df.NTRA, df.NTRI], axis=0),
            np.where(
                valid_NTRZ,
                df.NTRZ,  # NTRZ valid
                np.nan,
            ),
        ),
    )

    # define booleans for valid data and lmtQ_ for other parameters
    valid_NTRZ_corrected = ~pd.isna(df.NTRZ_corrected)
    valid_H2S = np.logical_and(~pd.isna(df.H2S), ~df.Q_H2S.str.contains("6|4|3|B|S|<"))
    valid_AMON = np.logical_and(~pd.isna(df.AMON), ~df.Q_AMON.str.contains("4|3|B|S"))
    below_det_AMON = np.logical_and(~pd.isna(df.AMON), df.Q_AMON.str.contains("6|<"))
    valid_low_doxy = np.logical_and(df.DOXY_BTL <= 2, ~df.Q_DOXY_BTL.str.contains("4|3|B|S"))

    df["din"] = np.nan

    # Fall där H2S är giltigt och NH4 är giltigt
    df.loc[valid_H2S & valid_AMON, "din"] = df.AMON

    # Steg 2: I låga syrehalter beräkna din endast
    # om AMON finns, antingen som summa AMON+NTRZ_corrected eller endast som AMON om NTRZ_corrected är nan.
    df["din"] = np.where(
        valid_low_doxy & valid_AMON,
        np.nansum([df.NTRZ_corrected, df.AMON], axis=0),
        df.din,  # Om inget av ovanstående gäller, lämna som `din`
    )

    # Typiskt sommaren när alla är under det
    df["din"] = np.where(
        (below_det_NTRZ | below_det_NTRA)
        & below_det_AMON
        & ~pd.isna(
            df.NTRZ_corrected
        ),  # Använd NTRZ_corrected + AMON om båda är giltiga
        df.NTRZ_corrected,
        df.din,  # Om inget av ovanstående gäller, lämna som `din`
    )

    # Övriga fall där NTRZ används som huvudsaklig parameter
    df["din"] = np.where(
        valid_AMON
        & ~valid_low_doxy
        & ~valid_H2S
        & ~below_det_AMON
        & ~pd.isna(
            df.NTRZ_corrected
        ),  # Använd NTRZ_corrected + AMON om båda är giltiga
        df.NTRZ_corrected + df.AMON,
        np.where(
            ~valid_low_doxy & ~valid_H2S & below_det_AMON & ~pd.isna(df.NTRZ_corrected),
            df.NTRZ_corrected,
            df.din,
        ),  # Om inget av ovanstående gäller, lämna som `din`
    )

    return df


def density(df: pd.DataFrame):
    """
    the sea pressure calculated from depth and latitude has very little effect on the results
    use constant latitude, comnsider using constant z as well
    """
    df.loc[:, "density"] = pot_rho_t_exact(df.salt, df.temp, p_from_z(-df.depth, 58), 0)


def oxygen_saturation(df: pd.DataFrame):
    # oxygen_ml2umol(df, oxygen_column_name=oxygen_column_name)
    pt = pt_from_CT(df.salt, df.temp)
    density(df)
    gsw = O2sol_SP_pt(df.salt, pt) * (df.density / 1000) / 44.661
    sw = satO2(df.salt, df.temp)

    df.loc[:, "oxygen_saturation"] = df.oxygen / gsw * 100

    return gsw, sw, df


def oxygen(df: pd.DataFrame):
    print(df)
    valid_btl = np.logical_and(
        ~pd.isna(df.DOXY_BTL), ~df.Q_DOXY_BTL.str.contains("B|S|<|4|3|6")
    )
    below_det_btl = np.logical_and(
        ~pd.isna(df.DOXY_BTL), df.Q_DOXY_BTL.str.contains("<|6")
    )
    valid_ctd = np.logical_and(
        ~pd.isna(df.DOXY_CTD), ~df.Q_DOXY_CTD.str.contains("B|S|<|4|3|6")
    )
    below_det_ctd = np.logical_and(
        ~pd.isna(df.DOXY_CTD), df.Q_DOXY_CTD.str.contains("<")
    )

    # Handle missing columns for H2S and Q_H2S
    if "H2S" in df.columns and "Q_H2S" in df.columns:
        valid_h2s = np.logical_and(
            ~pd.isna(df.H2S), ~df.Q_H2S.str.contains("B|S|Z|<|4|3|6")
        )
        below_det_h2s = np.logical_and(~pd.isna(df.H2S), df.Q_H2S.str.contains("<|6"))
    elif "H2S" in df.columns:
        valid_h2s = ~pd.isna(df.H2S)
        below_det_h2s = False
    else:
        # If either column is missing, set valid_h2s to False
        valid_h2s = False
        below_det_h2s = False
    
    # Apply nested np.where for all conditions
    df["oxygen"] = np.where(
        # h2s valid-> h2s default (0)
        valid_h2s,
        0,
        # both h2s and oxygen below det
        np.where(
            np.logical_and(
                below_det_h2s,
                np.logical_or(below_det_btl, ((below_det_ctd) & (~valid_btl))),
            ),
            0,
            #  O2 BTL is valid -> O2 BTL
            np.where(
                np.logical_or(valid_btl, below_det_btl),
                df.DOXY_BTL,
                # O2 CTD exists and Q O2 CTD is not B|S|< -> O2 CTD
                np.where(
                    (valid_ctd),
                    df.DOXY_CTD,
                    np.where(
                        (below_det_ctd),
                        0,
                        # Default case gives NaN
                        np.nan,
                    ),
                ),
            ),
        ),
    )

    return df
