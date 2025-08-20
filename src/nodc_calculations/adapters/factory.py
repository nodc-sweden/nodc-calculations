# nodc_calculations/adapters/factory.py
import pandas as pd
import polars as pl


def make_pandas_adapter(core_func, default_output_col):
    def adapter(df: pd.DataFrame, output_col=None, **kwargs):
        # Allow overriding output_col from the function call
        col_name = output_col or default_output_col

        # Convert to list of dicts
        rows = df.to_dict(orient="records")

        # Call core function with only kwargs that belong to it
        results = core_func(
            rows, **{k: v for k, v in kwargs.items() if k != "output_col"}
        )

        # Assign result
        df[col_name] = pd.Series(results, dtype="float")
        return df

    return adapter


def make_polars_adapter(core_func, default_output_col):
    def adapter(df: pl.DataFrame, output_col=None, **kwargs):
        col_name = output_col or default_output_col

        rows = df.to_dicts()

        results = core_func(
            rows, **{k: v for k, v in kwargs.items() if k != "output_col"}
        )

        df = df.with_columns(pl.Series(name=col_name, values=results, dtype=pl.Float64))
        return df

    return adapter
