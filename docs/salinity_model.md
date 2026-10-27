# Salinity model (SaltCSTR)

`SaltCSTR` in [`models.py`](../models.py) simulates salinity in one reach as
a fully mixed store of water and salt (a CSTR). It follows the lumped
in-stream salt routing of Davidson & Salbe (the IQQM salinity routing
upgrade) and is a port of `salt_segment.py`. It runs on any gauge: it needs
only the streamflow, the reach geometry and the NASA POWER forcings that the
temperature model already uses.

## Equations

The reach holds a volume `V` and a salt mass `M = c V`. Its salt balance is
Davidson's eq. 1:

```
dM/dt = Q_in c_in + L − Q_out M / V
```

For a single reach, the storage follows the hydraulic geometry at the current
streamflow, and it is held constant within a step (`dV/dt = 0`). The outflow
is then `Q_out = Q_in − Q_ev`, and the state is the concentration `c`:

```
V dc/dt = Q (c_bf − c) + Q_qf (c_qf − c) + Q_ev c + L
```

| Term | Formula | Meaning |
|---|---|---|
| `V` | `w d L_reach` | reach volume [m³] |
| `A_s` | `w L_reach` | water surface [m²] |
| `w`, `d` | `c_width Q^f_width`, `max(c_depth Q^f_depth, d_min)` | width and depth from the reach ratings [m] |
| `Q` | steady (or observed) streamflow | upstream inflow at the baseflow end member `c_bf` [m³/s] |
| `Q_qf` | `k_rain P A_s` | rain-driven quickflow at the quickflow end member `c_qf` [m³/s] |
| `Q_ev` | `E A_s`, `E = f(W) (e_s(T_a) − e_a)` | evaporation: removes water, not salt [m³/s] |
| `c_in` | `(Q c_bf + Q_qf c_qf) / (Q + Q_qf)` | mixed inflow concentration (`mix_end_members`) [mg/L] |
| `L` | parameter | external load: groundwater, tidal, point sources [g/s] |

The four terms of the balance are the budget the app plots, each in mg/L
per day:
- **upstream inflow**, `Q (c_bf − c) / V`, pulls `c` toward `c_bf`;
- **rain dilution**, `Q_qf (c_qf − c) / V`, pulls it toward `c_qf`;
- **evaporation**, `Q_ev c / V`, concentrates the salt;
- **external load**, `L / V`.

Their sum is `dc/dt`. With constant forcing, the reach settles at

```
c* = (Q c_bf + Q_qf c_qf + L) / Q_out
```

after a few residence times `V / Q_out`.

## Parameters

| Index | Name | Default | Unit | Meaning |
|---|---|---|---|---|
| 0 | `L_reach` | 1000 | m | reach length; with the ratings it sets `V` and `A_s` |
| 1 | `c_bf` | 500 (gauge) | mg/L | baseflow end member, the upstream concentration |
| 2 | `c_qf` | 20 | mg/L | quickflow end member, the rain-driven inflow |
| 3 | `k_rain` | 1 | – | rain-driven inflow per unit water surface, as in RNSTM |
| 4 | `load` | 0 | g/s | net external salt load (negative = withdrawal) |
| 5 | `a_w` | 1.505e-9 | m/(s·mbar) | wind-function intercept of evaporation (as RNSTM) |
| 6 | `b_w` | 1.6e-9 | 1/mbar | wind-function slope of evaporation (as RNSTM) |

`c_bf` is site-specific. Whenever the gauge or the window changes, the app
sets it to the initial concentration (the gauge's first observation), so a
run starts near equilibrium. The width rating (`c_width`, `f_width`) is
fitted to the USGS field measurements together with the depth rating.

## Differences from `salt_segment.py`

- **Storage.** `salt_segment.py` integrates `dV/dt = Q_in − Q_out` from given
  inflow and outflow series. Here the volume follows the hydraulic geometry
  at the current Q, and the outflow closes the water balance. With observed Q,
  `V` changes between records but not within them.
- **State.** The state is the concentration `c`, not the mass `M`. With `V`
  constant within a step the two are equivalent. The solver's continuous
  solution conserves mass, so the step-average outflow concentration of
  Davidson's eq. 4 is not needed.
- **Dry reaches.** Dry reaches cannot occur: the minimum depth `d_min` keeps
  `V > 0`. The `V_MIN` store of `salt_segment.py` is therefore not needed.
- **Inflows from the forcings.** The end members come from the forcings: the
  streamflow is baseflow and the rain makes the quickflow, rather than taking
  separate baseflow and quickflow series.
- **Evaporation is new.** It uses RNSTM's mass-transfer formula evaluated at
  the air temperature, because this model has no water temperature.

`mix_end_members` and `ec25_from_measured` are kept as functions.
`ec25_from_measured` is not used by the app, since USGS specific conductance
is already corrected to 25 °C.

## Observations

The model works in mg/L of total dissolved solids (TDS). USGS gauges report
specific conductance at 25 °C (parameter `00095`, µS/cm), which the app
converts with

```
TDS = 0.65 × SC
```

The factor depends on the ions in the water and typically ranges from 0.55
to 0.75. It is `SALINITY.obs_factor` in `models.py`. Few Florida stream
gauges report salinity in ppt (`00480`) without specific conductance, so the
app uses specific conductance only.

## Example: St. Johns River near Cocoa, June 2026

With the defaults and `c_bf` taken from the gauge (704 mg/L):
- **Fit:** the reach stays within about 2% of 705 mg/L, and the mean
  absolute relative error against the gauge is 7.8%.
- **Reach:** about 55 m wide and 1.5 m deep, with a residence time of about
  14 h.
- **Budget:** evaporation (about 6 mm/day) concentrates by about +3 mg/L/day
  and rain dilutes by about −2 mg/L/day on average.
- **Misses:** the gauge shows a fresh pulse on 1–5 June (TDS down to about
  190 mg/L) during a period of flow reversal. A steady-flow reach with a
  fixed upstream concentration cannot reproduce it.
