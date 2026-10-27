# Adding a model

Models live in [`models.py`](../models.py). Once a model is registered, the
app lists it next to the existing ones. Up to three models can be selected
and compared.

## 1. Pick or add the constituent

Each model simulates one `Constituent`. The constituent says how the
variable is labelled and how a USGS gauge observes it:

```python
SALINITY = Constituent("salinity", "Specific conductance", "µS/cm",
                       usgs_pcode="00095", obs_column="spec_cond",
                       rel_error_floor=1.0)
CONSTITUENTS = {c.key: c for c in (TEMPERATURE, SALINITY)}
```

A new constituent also needs its USGS code in `usgs.PCODES`
(`"00095": ("spec_cond", 1.0)`). That makes `instantaneous_values` return
the observations.

## 2. Write the flux and RHS functions

```python
def fluxes_my_model(t, y, params, global_params, forcing):
    k, c_in = global_params
    f = forcing(t)                    # forcings.ForcingSample
    depth = reach_depth(f.q, params)  # shared hydraulic geometry
    ...
    return {"gain": ..., "loss": ..., "rate_day": ...}


def model_my_model(t, y, params, global_params, forcing):
    f = fluxes_my_model(t, y, params, global_params, forcing)
    return [f["rate_day"] / 1440.0]   # per minute
```

- `t` is in minutes and `forcing(t)` returns the hourly record covering `t`.
- `params` is the reach: `(c_depth, f_depth, d_min)`.
- Return every internal flux you want to plot from `fluxes`. The app's budget
  plot and `flux_timeseries` read them from there.

A model that needs a forcing that is not in `forcings.FIELDS` yet needs a new
field in `ForcingSample` and in `fetch_nasa_power` (or a new source function).

## 3. Register it

```python
register_model(
    "MyModel", SALINITY, state_names=("C",),
    params=(Param("k", "1/day", 0.1, "Decay rate."),
            Param("c_in", "µS/cm", 300.0, "Inflow concentration.")),
    fluxes=fluxes_my_model,
    description="One line shown in the app.",
    budget=(("gain", "Inflow", +1), ("loss", "Decay", -1)),
    budget_unit="µS/cm/day",
    equations=((r"\frac{dC}{dt}", r"\dots"),),
    forcings=("t_air",),              # checked for gaps before running
)(model_my_model)
```

`register_model` is a decorator, so `@register_model(...)` above
`def model_my_model` works too.

## Notes for non-temperature constituents

The app currently plots `CONSTITUENT = TEMPERATURE` (top of `app.py`). Two
pieces of the app are temperature-specific and should become per-constituent
when the second constituent arrives:

- the constituent selector, which would replace that constant;
- the "Air temperature on the plot" toggle and the heat-budget title.
