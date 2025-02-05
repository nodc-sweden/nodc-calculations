import pandas as pd


def oxygen_ml2umol(data: pd.DataFrame, oxygen_column_name: str):
    data.loc[:, f"{oxygen_column_name}_umol"] = (
        data.loc[:, f"{oxygen_column_name}"] * 44.661
    )

    return data


def gram_per_liter_to_mol_per_liter(
    data: pd.DataFrame, nutrient: str, incoming_column_name: str, out_column_name
):
 
    # molar mass of relevant nutrients
    gram_per_mol = {
        "N": 14.006720,
        "P": 30.973762,
        "SI": 28.085530,
    }

    if nutrient not in gram_per_mol:
        return data

    # convert g/l to mol/l by dividing with the molar mass

    data.loc[:, out_column_name] = data.loc[:, incoming_column_name] / gram_per_mol[nutrient]

    return data
