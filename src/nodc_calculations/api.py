# nodc_calculations/api.py
from nodc_calculations.core.calculations import dissolved_inorganic_nitrogen_core
from nodc_calculations.core.calculations.ntrz import calculate_ntrz_corr
from nodc_calculations.adapters.factory import make_pandas_adapter, make_polars_adapter

# Create adapters using the factory
dissolved_inorganic_nitrogen_pandas = make_pandas_adapter(
    dissolved_inorganic_nitrogen_core,
    output_col="din"
)
dissolved_inorganic_nitrogen_polars = make_polars_adapter(
    dissolved_inorganic_nitrogen_core,
    output_col="din"
)

calculate_ntrz_corr_pandas = make_pandas_adapter(
    calculate_ntrz_corr,
    output_col="NTRZ_corr"
)
calculate_ntrz_corr_polars = make_polars_adapter(
    calculate_ntrz_corr,
    output_col="NTRZ_corr"
)

__all__ = [
    # Core
    "dissolved_inorganic_nitrogen_core",
    "calculate_ntrz_corr",

    # Pandas
    "dissolved_inorganic_nitrogen_pandas",
    "calculate_ntrz_corr_pandas",

    # Polars
    "dissolved_inorganic_nitrogen_polars",
    "calculate_ntrz_corr_polars",
]
