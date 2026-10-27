# StreamQual toy

A small Python sandbox for prototyping **stream water-quality models**
(water temperature now; salinity and others later) on a single reach at a
USGS gauge, before committing to an implementation in
[ASYNCH](https://github.com/Iowa-Flood-Center/asynch). It follows the layout of
the HLM models playground: models are registered right-hand sides that
`scipy.integrate.solve_ivp` integrates, and a Streamlit app compares them.

The first model is the **River Network Streamflow Temperature Model (RNSTM)**
([García-Múnera et al., 2026](https://doi.org/10.1111/1752-1688.70140)),
ported from its asynch implementation: a heat budget of solar radiation,
long-wave radiation, evaporation, convection and cooling by rain (rain
`dT_rain` °C colder than the air) over a depth `d = c_depth·Q^f_depth`.

## Contents

| File | What it is |
|---|---|
| [`models.py`](models.py) | Model registry, the water-temperature models (`RNSTM`, `RNSTM_LWobs`) and `simulate`. |
| [`forcings.py`](forcings.py) | Hourly NASA POWER forcings (`fetch_nasa_power`) and the piecewise-constant `HourlyForcing` the models sample. |
| [`usgs.py`](usgs.py) | USGS gauge search, site info, instantaneous values (Q, gage height, depth, water temperature) and field measurements, plus the `d = c·Q^f` fit. |
| [`metrics.py`](metrics.py) | Step-by-step relative error and summary scores (MRE, MARE, bias, RMSE, NSE, KGE). |
| [`app.py`](app.py) | **Stream temperature explorer** web app. |
| [`http_cache.py`](http_cache.py) | On-disk cache of the downloaded data (`.cache/`). |
| [`docs/temperature_model.md`](docs/temperature_model.md) | Equations, parameters, forcings, and **the differences from (and bugs found in) the C code**. |
| [`docs/adding_a_model.md`](docs/adding_a_model.md) | How to add a model or a new constituent. |

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate      # source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
streamlit run app.py
```

The app needs internet access for NASA POWER and the USGS services. The
responses are cached in `.cache/`; delete it to download again.

## Stream temperature explorer

In the sidebar:

1. **USGS gauge**: type a site number, or use *Find a gauge with water
   temperature* to list a state's active gauges. The default is
   02232400, St. Johns River near Cocoa, FL, in Brevard County.
2. **Simulation**: start date (local standard time of the gauge) and length
   (30 days by default).
3. **Streamflow**: the steady-state Q. It defaults to the gauge's median Q on
   the first day. The run can also be driven with the observed hourly Q.
4. **Initial state**: water temperature at the start. It defaults to the
   first observation.
5. **Channel depth**: `c_depth`, `f_depth` and `d_min`. They are fitted to the
   gauge's USGS field measurements (mean depth = area / width) whenever a
   gauge is loaded.

On the left, tick up to three models and edit their parameters; *Equations*
shows each model's equations. The plots update on every change:

1. streamflow: model Q and USGS Q;
2. depth: model depth, the same rating applied to the observed Q, USGS
   measured depths, and gage height on the right axis;
3. water temperature: models and USGS observations (and, optionally, air
   temperature);
4. step-by-step relative error, `(sim − obs) / obs · 100`, per hour.

A table gives the mean relative error, the mean absolute relative error and
other scores. *Show forcings* plots the NASA POWER forcings, and *Show heat
budget* plots the heat-flux terms. A CSV download has the time series.

## Minimal run from Python

```python
import pandas as pd
from forcings import fetch_nasa_power, HourlyForcing
from models import get_model, simulate
from usgs import site_info, instantaneous_values

g = site_info("02232400")
t0 = pd.Timestamp("2026-06-01", tz="UTC") - pd.Timedelta(hours=g.utc_offset_h)
t1 = t0 + pd.Timedelta(days=30)
meteo = fetch_nasa_power(g.lat, g.lon, t0, t1)
forcing = HourlyForcing(meteo.iloc[:-1], q=2.0)          # steady Q [m^3/s]

res = simulate(get_model("RNSTM"), forcing,
               reach=(1.4, 0.15, 0.1), y0=(30.0,))       # c_depth, f_depth, d_min
print(res["value"][:5], list(res["fluxes"]))
obs = instantaneous_values(g.site_no, t0, t1)["water_temp"]
```

## Conventions

- **Time** is in **minutes** inside the solver. Forcing records are hourly and
  piecewise constant, as in ASYNCH. Data are handled in UTC and plotted in the
  gauge's local standard time (no daylight saving).
- **Streamflow** is in m³/s, **depth** in m, **heat fluxes** in W/m² (positive
  into the water).
- **USGS values** are converted to SI and averaged to hourly values: the value
  at hour `h` is the mean of the readings within ±30 min.

## Data sources

- [NASA POWER](https://power.larc.nasa.gov/) hourly API: radiation, air
  temperature, humidity, wind, pressure, cloud cover, precipitation. Radiation and cloud
  products lag about three months.
- [USGS Water Services](https://waterservices.usgs.gov/) (instantaneous
  values, site information) and the
  [USGS Water Data OGC API](https://api.waterdata.usgs.gov/) (channel
  measurements).

## Reference

García-Múnera, V., Velásquez Giron, N., Molina, R. D., Weber, L. J., &
Krajewski, W. F. (2026). RNSTM: A Network-Link Approach to Streamflow
Temperature Modeling. *JAWRA Journal of the American Water Resources
Association*, 62(4), e70140.
<https://doi.org/10.1111/1752-1688.70140>
