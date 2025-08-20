# tests/test_adapters/test_din_adapter.py
import pandas as pd
import polars as pl
from nodc_calculations.core.calculations import dissolved_inorganic_nitrogen_core
from nodc_calculations import dissolved_inorganic_nitrogen_pandas
from nodc_calculations import dissolved_inorganic_nitrogen_polars

sample_data = [
    {
        "NTRA": 1.0, "Q_NTRA": "", 
        "NTRI": 2.0, "Q_NTRI": "",
        "NTRZ": None, "Q_NTRZ": "",
        "H2S": None, "Q_H2S": "",
        "AMON": 0.5, "Q_AMON": "",
        "DOXY_BTL": 1.0, "Q_DOXY_BTL": ""
    },
    # Add more cases here
]

def test_core_vs_adapters():
    expected = dissolved_inorganic_nitrogen_core(sample_data)

    df_pd = pd.DataFrame(sample_data)
    out_pd = dissolved_inorganic_nitrogen_pandas(df_pd)
    assert out_pd["din"].tolist() == expected

    df_pl = pl.DataFrame(sample_data)
    out_pl = dissolved_inorganic_nitrogen_polars(df_pl)
    assert out_pl["din"].to_list() == expected
