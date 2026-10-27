# Adding a model

Models live in [`models.py`](../models.py). Once a model is registered, the
app lists it under its constituent in the *Variable* selector. Up to three
models of the same constituent can be selected and compared.

## 1. Pick or add the constituent

Each model simulates one `Constituent`. The constituent says how the
variable is labelled and how a USGS gauge observes it. Salinity, for example:

```python
SALINITY = Constituent("salinity", "Salinity (TDS)", "mg/L",
                       usgs_pcode="00095", obs_column="spec_cond",
                       rel_error_floor=1.0, obs_factor=0.65,
                       obs_note="TDS = 0.65 × USGS specific conductance "
                                "at 25 °C [µS/cm]")
CONSTITUENTS = {c.key: c for c in (TEMPERATURE, SALINITY)}
```

- `obs_factor` converts the gauge's unit to the model's.
- `obs_note` is shown in the app wherever the observations are compared.

A new constituent also needs:
- its USGS code in `usgs.PCODES` (`"00095": ("spec_cond", 1.0)`), which makes
  `instantaneous_values` return the observations;
- an entry in the app's `GAUGE_FILTERS` / `FILTER_OF`, for the gauge finder;
- an entry in `FALLBACK_Y0`, the initial state when the gauge has no
  observation at the start.

## 2. Write the flux and RHS functions

```python
def fluxes_my_model(t, y, params, global_params, forcing):
    k, c_in = global_params
    f = forcing(t)                    # forcings.ForcingSample
    depth = reach_depth(f.q, params)  # shared hydraulic geometry
    width = reach_width(f.q, params)
    ...
    return {"gain": ..., "loss": ..., "rate_day": ...}


def model_my_model(t, y, params, global_params, forcing):
    f = fluxes_my_model(t, y, params, global_params, forcing)
    return [f["rate_day"] / 1440.0]   # per minute
```

- `t` is in minutes, and `forcing(t)` returns the hourly record covering `t`.
- `params` is the reach: `(c_depth, f_depth, d_min, c_width, f_width)`.
- Return every internal flux you want to plot from `fluxes`. The app's budget
  plot and `flux_timeseries` read them from there.

A model that needs a forcing that is not in `forcings.FIELDS` yet needs a new
field in `ForcingSample` and in `fetch_nasa_power` (or a new source function).

## 3. Register it

```python
register_model(
    "MyModel", SALINITY, state_names=("c",),
    params=(Param("k", "1/day", 0.1, "Decay rate."),
            Param("c_in", "mg/L", 300.0, "Inflow concentration.",
                  follows_initial=True)),
    fluxes=fluxes_my_model,
    description="One line shown in the app.",
    budget=(("gain", "Inflow", +1), ("loss", "Decay", -1)),
    budget_unit="mg/L/day", budget_title="Rate of change of c",
    equations=((r"\frac{dc}{dt}", r"\dots"),),
    notation="c: concentration [mg/L]; ...",
    forcings=("t_air",),              # checked for gaps before running
    summary=lambda f: f"Mean gain {f['gain'].mean():.2f} mg/L/day.",
)(model_my_model)
```

- `register_model` is a decorator, so `@register_model(...)` above
  `def model_my_model` works too.
- `Param(..., advanced=True)` puts a parameter in the *Advanced parameters*
  pop-over.
- `follows_initial=True` makes the app set it to the initial state (the
  gauge's first observation) whenever the gauge or the window changes. Use it
  for site-specific values such as an inflow concentration.
- `summary` is the sentence shown under the budget plot.
