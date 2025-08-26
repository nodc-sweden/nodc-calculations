# nodc_calculations/api.py
from nodc_calculations.core.calculations import dissolved_inorganic_nitrogen_core
from nodc_calculations.core.calculations.ntrz import ntrz_core
from nodc_calculations.adapters.factory import make_pandas_adapter, make_polars_adapter

# Create adapters using the factory
dissolved_inorganic_nitrogen_pandas = make_pandas_adapter(
    dissolved_inorganic_nitrogen_core, output_col="din"
)
dissolved_inorganic_nitrogen_polars = make_polars_adapter(
    dissolved_inorganic_nitrogen_core, output_col="din"
)

ntrz_core_pandas = make_pandas_adapter(ntrz_core, output_col="NTRZ_corr")
ntrz_core_polars = make_polars_adapter(ntrz_core, output_col="NTRZ_corr")

__all__ = [
    # Core
    "dissolved_inorganic_nitrogen_core",
    "ntrz_core",
    # Pandas
    "dissolved_inorganic_nitrogen_pandas",
    "ntrz_core_pandas",
    # Polars
    "dissolved_inorganic_nitrogen_polars",
    "ntrz_core_polars",
]
