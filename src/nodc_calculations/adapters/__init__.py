# nodc_calculations/adapters/__init__.py

from ..core.calculations.din import dissolved_inorganic_nitrogen_core
from ..core.calculations.ntrz import ntrz_core
from ..core.calculations.oxygen import oxygen_core, oxygen_saturation_core
from ..core.calculations.combine_ctd_btl import salinity_core, temperature_core
from ..core.calculations.density import density_core
from ..adapters.factory import make_pandas_adapter, make_polars_adapter

# DIN
dissolved_inorganic_nitrogen_pandas = make_pandas_adapter(
    dissolved_inorganic_nitrogen_core,
    default_output_col="din"
)
dissolved_inorganic_nitrogen_polars = make_polars_adapter(
    dissolved_inorganic_nitrogen_core,
    default_output_col="din"
)

# NTRZ_corr
ntrz_corr_pandas = make_pandas_adapter(
    ntrz_core,
    default_output_col="NTRZ_corr"
)
ntrz_corr_polars = make_polars_adapter(
    ntrz_core,
    default_output_col="NTRZ_corr"
)

# oxygen
oxygen_pandas = make_pandas_adapter(
    oxygen_core,
    default_output_col="oxygen"
)
oxygen_polars = make_polars_adapter(
    oxygen_core,
    default_output_col="oxygen"
)

# oxygen_saturation
oxygen_saturation_pandas = make_pandas_adapter(
    oxygen_saturation_core,
    default_output_col="oxygen_saturation"
)
oxygen_saturation_polars = make_polars_adapter(
    oxygen_saturation_core,
    default_output_col="oxygen_saturation"
)

# salinity
salinity_pandas = make_pandas_adapter(
    salinity_core,
    default_output_col="salinity"
)
salinity_polars = make_polars_adapter(
    salinity_core,
    default_output_col="salinity"
)

# temperature
temperature_pandas = make_pandas_adapter(
    temperature_core,
    default_output_col="temperature"
)
temperature_polars = make_polars_adapter(
    temperature_core,
    default_output_col="temperature"
)

# density
density_pandas = make_pandas_adapter(
    density_core,
    default_output_col="density"
)
density_polars = make_polars_adapter(
    density_core,
    default_output_col="density"
)