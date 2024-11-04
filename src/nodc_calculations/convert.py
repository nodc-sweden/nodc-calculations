import pandas as pd

def oxygen_ml2umol(data: pd.DataFrame, oxygen_column_name: str):

    data.loc[:, f"{oxygen_column_name}_umol"] = data.loc[:, f"{oxygen_column_name}"]*44.661

    return data
