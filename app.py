"""Stream temperature explorer -- simulate a reach at a USGS gauge for 30 days.

Run with::

    streamlit run app.py

Pick a USGS gauge and a start date; the app pulls hourly NASA POWER forcings
at the gauge, holds the streamflow at a user-defined steady state and runs up
to three models of ``models.MODELS``. Where the gauge reports the simulated
constituent (water temperature), it is plotted as observations and used for
step-by-step relative errors and summary scores.
"""

import math
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from forcings import (FIELD_LABELS, POWER_PARAMS, HourlyForcing,
                      fetch_nasa_power, missing_report)
from metrics import stepwise, summary
from models import MODELS, REACH_PARAMS, TEMPERATURE, reach_depth, simulate
from usgs import (STATES, channel_measurements, find_gauges,
                  fit_depth_rating, instantaneous_values, site_info)

DEFAULT_SITE = "02232400"           # St. Johns River near Cocoa, FL
DEFAULT_STATE = "FL"
DEFAULT_START = date(2026, 6, 1)
DEFAULT_DAYS = 30
MAX_MODELS = 3
MODEL_COLORS = ("#d62728", "#1f77b4", "#2ca02c")
OBS_COLOR = "#374151"
RATING_YEARS = 10                   # fit the depth rating to +/- this window
CONSTITUENT = TEMPERATURE           # the only constituent modelled so far

st.set_page_config(page_title="Stream temperature explorer", layout="wide")


# ----------------------------------------------------------------------------
# Data (cached in memory here and on disk by http_cache)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Reading the USGS site…")
def load_site(site_no):
    return site_info(site_no)


@st.cache_data(show_spinner="Searching USGS gauges…")
def load_gauge_list(state):
    return find_gauges(state, CONSTITUENT.usgs_pcode)


@st.cache_data(show_spinner="Downloading USGS observations…")
def load_observations(site_no, t0, t1):
    return instantaneous_values(site_no, t0, t1)


@st.cache_data(show_spinner="Downloading USGS field measurements…")
def load_measurements(site_no):
    return channel_measurements(site_no)


@st.cache_data(show_spinner="Downloading NASA POWER forcings…")
def load_meteo(lat, lon, t0, t1, power_params):
    # power_params is only part of the cache key: adding a POWER variable
    # must not serve a cached frame without it.
    return fetch_nasa_power(lat, lon, t0, t1)


@st.cache_data(show_spinner=False)
def run_model(name, meteo, q, reach, gp, y0):
    forcing = HourlyForcing(meteo, q)
    return simulate(MODELS[name], forcing, reach, y0, global_params=gp)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _step(v):
    """A sensible +/- step for a number_input, based on the value's magnitude."""
    v = abs(v)
    return 10.0 ** (math.floor(math.log10(v)) - 1) if v > 0 else 0.01


def _reset_model(name):
    for k in [k for k in st.session_state if k.startswith(f"{name}::")]:
        del st.session_state[k]


def _toggle_equations(name):
    key = f"eqs::{name}"
    st.session_state[key] = not st.session_state.get(key, False)


def _use_gauge(site_no):
    st.session_state["site_no"] = site_no


def _apply_rating(fit):
    st.session_state["reach::c_depth"] = round(fit["c_depth"], 4)
    st.session_state["reach::f_depth"] = round(fit["f_depth"], 4)


def _set_default(key, value, ctx_key, ctx):
    """Reset widget ``key`` to ``value`` whenever ``ctx`` changes."""
    if st.session_state.get(ctx_key) != ctx:
        st.session_state[ctx_key] = ctx
        st.session_state[key] = value


def _aligned(lines):
    return r"\begin{aligned}" + r" \\[6pt] ".join(lines) + r"\end{aligned}"


def model_inputs(model):
    """Render the parameter inputs of one model; return them as a tuple."""
    gp = list(model.default_params)
    basic = [(i, p) for i, p in enumerate(model.params) if not p.advanced]
    cols = st.columns(2)
    for k, (i, p) in enumerate(basic):
        with cols[k % 2]:
            gp[i] = st.number_input(
                f"{p.name} [{p.unit}]", value=float(p.default),
                step=_step(p.default), format="%g", help=p.help,
                key=f"{model.name}::gp{i}")
    advanced = [(i, p) for i, p in enumerate(model.params) if p.advanced]
    if advanced:
        with st.popover("Physical constants", icon=":material/tune:"):
            for i, p in advanced:
                gp[i] = st.number_input(
                    f"{p.name} [{p.unit}]", value=float(p.default),
                    step=_step(p.default), format="%g", help=p.help,
                    key=f"{model.name}::gp{i}")
    st.button("Reset to defaults", key=f"{model.name}::reset",
              on_click=_reset_model, args=(model.name,))
    return tuple(gp)


def equations_panel(model):
    with st.container(border=True):
        st.markdown(f"#### {model.name} equations")
        st.latex(_aligned(f"{l} &= {r}" for l, r in model.equations))
        st.caption("Definitions")
        st.latex(_aligned(f"{l} &= {r}" for l, r in model.definitions))
        st.caption("T_w, T_a: water and air temperature [°C]; R_sw, R_lw: "
                   "incoming short- and long-wave radiation [W/m²]; RH: "
                   "relative humidity [0–1]; C: cloud cover [0–1]; W₂: wind "
                   "at 2 m [m/s]; p_a: pressure [mbar]; P: precipitation "
                   "[m/s]; Q: streamflow [m³/s]; H terms in W/m².")
        st.dataframe(pd.DataFrame(
            [(p.name, p.unit, p.default, p.help) for p in model.params],
            columns=["Parameter", "Unit", "Default", "Meaning"]),
            hide_index=True, width="stretch")


def _local(index, offset_h):
    """UTC index -> naive local-standard-time index (for the plots)."""
    return (index + pd.Timedelta(hours=offset_h)).tz_localize(None)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def make_figure(tl, q_model, depth_model, obs, depth_from_obs_q, meas_in_window,
                results, steps, meteo_l, show_air):
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.035,
        row_heights=[0.18, 0.22, 0.38, 0.22],
        specs=[[{}], [{"secondary_y": True}], [{}], [{}]])

    # 1. Streamflow
    if "discharge" in obs:
        fig.add_trace(go.Scatter(
            x=obs.index, y=obs["discharge"], name="USGS Q",
            legendgroup="obs", legendgrouptitle_text="USGS gauge",
            line=dict(color=OBS_COLOR, width=1.2),
            hovertemplate="%{y:.3g} m³/s<extra>USGS Q</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=tl, y=q_model, name="Model Q", legendgroup="model",
        legendgrouptitle_text="Reach", line=dict(color="#7c3aed", width=2.2),
        hovertemplate="%{y:.3g} m³/s<extra>Model Q</extra>"), row=1, col=1)

    # 2. Depth (+ gage height on the right axis)
    fig.add_trace(go.Scatter(
        x=tl, y=depth_model, name="Model depth", legendgroup="model",
        line=dict(color="#7c3aed", width=2.2),
        hovertemplate="%{y:.3g} m<extra>Model depth</extra>"), row=2, col=1)
    if depth_from_obs_q is not None:
        fig.add_trace(go.Scatter(
            x=obs.index, y=depth_from_obs_q, name="Rating depth of USGS Q",
            legendgroup="model", line=dict(color="#7c3aed", width=1.3, dash="dot"),
            hovertemplate="%{y:.3g} m<extra>Rating depth of USGS Q</extra>"),
            row=2, col=1)
    if "mean_depth" in obs:
        fig.add_trace(go.Scatter(
            x=obs.index, y=obs["mean_depth"], name="USGS mean depth",
            legendgroup="obs", line=dict(color=OBS_COLOR, width=1.5),
            hovertemplate="%{y:.3g} m<extra>USGS mean depth</extra>"),
            row=2, col=1)
    if meas_in_window is not None and len(meas_in_window):
        fig.add_trace(go.Scatter(
            x=meas_in_window["time"], y=meas_in_window["depth"],
            name="USGS measured depth (A/W)", legendgroup="obs", mode="markers",
            marker=dict(color=OBS_COLOR, size=9, symbol="diamond"),
            hovertemplate="%{y:.3g} m<extra>field measurement</extra>"),
            row=2, col=1)
    if "gage_height" in obs:
        fig.add_trace(go.Scatter(
            x=obs.index, y=obs["gage_height"], name="USGS gage height",
            legendgroup="obs", line=dict(color="#0891b2", width=1.2, dash="dash"),
            hovertemplate="%{y:.3g} m<extra>gage height</extra>"),
            row=2, col=1, secondary_y=True)

    # 3. Constituent
    col = CONSTITUENT.obs_column
    if show_air:
        fig.add_trace(go.Scatter(
            x=meteo_l.index, y=meteo_l["t_air"], name="Air temperature",
            legendgroup="forcing", legendgrouptitle_text="Forcing",
            line=dict(color="#9ca3af", width=1),
            hovertemplate="%{y:.1f} °C<extra>air</extra>"), row=3, col=1)
    if col in obs:
        fig.add_trace(go.Scatter(
            x=obs.index, y=obs[col], name=f"USGS {CONSTITUENT.label.lower()}",
            legendgroup="obs", mode="markers",
            marker=dict(color=OBS_COLOR, size=3.5),
            hovertemplate="%{y:.2f} " + CONSTITUENT.unit + "<extra>observed</extra>"),
            row=3, col=1)
    for (name, res), color in zip(results, MODEL_COLORS):
        fig.add_trace(go.Scatter(
            x=tl, y=res["value"], name=name, legendgroup=name,
            line=dict(color=color, width=2.2),
            hovertemplate="%{y:.2f} " + CONSTITUENT.unit + f"<extra>{name}</extra>"),
            row=3, col=1)

    # 4. Step-by-step relative error
    for (name, _), color in zip(results, MODEL_COLORS):
        if name in steps:
            fig.add_trace(go.Scatter(
                x=steps[name].index, y=steps[name]["rel_error"],
                name=f"{name} · rel. error", legendgroup=name, showlegend=False,
                line=dict(color=color, width=1.4),
                hovertemplate="%{y:.2f} %<extra>" + name + "</extra>"),
                row=4, col=1)
    fig.add_hline(y=0, line=dict(color="#9ca3af", width=1), row=4, col=1)

    fig.update_yaxes(title_text="Q [m³/s]", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title_text="Depth [m]", rangemode="tozero", row=2, col=1)
    fig.update_yaxes(title_text="Gage height [m]", showgrid=False,
                     row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text=f"{CONSTITUENT.label} [{CONSTITUENT.unit}]",
                     row=3, col=1)
    fig.update_yaxes(title_text="Rel. error [%]", row=4, col=1)
    fig.update_layout(height=860, hovermode="x unified",
                      margin=dict(l=60, r=20, t=10, b=30),
                      legend=dict(groupclick="toggleitem"))
    return fig


def forcing_figure(meteo_l, results, tl):
    fig = make_subplots(rows=6, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03)
    line = lambda c, w=1.4: dict(color=c, width=w)
    hover = lambda u: "%{y:.4g} " + u + "<extra>%{fullData.name}</extra>"
    fig.add_trace(go.Scatter(x=meteo_l.index, y=meteo_l["sw_down"],
                             name="Shortwave ↓", line=line("#f59e0b"),
                             hovertemplate=hover("W/m²")), row=1, col=1)
    fig.add_trace(go.Scatter(x=meteo_l.index, y=meteo_l["lw_down"],
                             name="Longwave ↓ (POWER)", line=line("#b45309"),
                             hovertemplate=hover("W/m²")), row=1, col=1)
    # Long-wave computed by models that use the emissivity formula
    for (name, res), color in zip(results, MODEL_COLORS):
        if np.isfinite(res["fluxes"]["eps_a"]).any():
            fig.add_trace(go.Scatter(
                x=tl, y=res["fluxes"]["H_atm"] / 0.97,
                name=f"Longwave ↓ ({name})", line=dict(color=color, width=1, dash="dot"),
                hovertemplate=hover("W/m²")), row=1, col=1)
            break
    fig.add_trace(go.Scatter(x=meteo_l.index, y=meteo_l["t_air"],
                             name="Air temperature", line=line("#dc2626"),
                             hovertemplate=hover("°C")), row=2, col=1)
    fig.add_trace(go.Scatter(x=meteo_l.index, y=meteo_l["rh"],
                             name="Relative humidity", line=line("#0284c7"),
                             hovertemplate=hover("%")), row=3, col=1)
    fig.add_trace(go.Scatter(x=meteo_l.index, y=meteo_l["cloud"],
                             name="Cloud cover", line=line("#6b7280"),
                             hovertemplate=hover("%")), row=3, col=1)
    fig.add_trace(go.Scatter(x=meteo_l.index, y=meteo_l["wind2"],
                             name="Wind at 2 m", line=line("#059669"),
                             hovertemplate=hover("m/s")), row=4, col=1)
    fig.add_trace(go.Scatter(x=meteo_l.index, y=meteo_l["pressure"] / 100.0,
                             name="Pressure", line=line("#7c3aed"),
                             hovertemplate=hover("hPa")), row=5, col=1)
    fig.add_trace(go.Bar(x=meteo_l.index + pd.Timedelta(minutes=30),
                         y=meteo_l["rain"], width=3.6e6, name="Precipitation",
                         marker_color="#2563eb",
                         hovertemplate=hover("mm/h")), row=6, col=1)
    for row, title in enumerate(("Radiation [W/m²]", "T air [°C]", "RH, cloud [%]",
                                 "Wind [m/s]", "Pressure [hPa]", "Rain [mm/h]"),
                                start=1):
        fig.update_yaxes(title_text=title, row=row, col=1)
    fig.update_yaxes(range=[0, 105], row=3, col=1)
    fig.update_layout(height=860, hovermode="x unified", bargap=0,
                      margin=dict(l=60, r=20, t=10, b=30))
    return fig


def budget_figure(model, res, tl):
    fig = make_subplots(rows=1, cols=2, column_widths=[0.78, 0.22],
                        horizontal_spacing=0.06)
    colors = ("#f59e0b", "#b45309", "#0284c7", "#059669", "#9333ea")
    means = []
    for (key, label, sign), color in zip(model.budget, colors):
        v = sign * res["fluxes"][key]
        means.append((key, float(np.mean(v)), color))
        fig.add_trace(go.Scatter(x=tl, y=v, name=label, line=dict(color=color, width=1.4),
                                 hovertemplate="%{y:.0f} W/m²<extra>" + label + "</extra>"),
                      row=1, col=1)
    net = sum(sign * res["fluxes"][key] for key, _, sign in model.budget)
    means.append(("Net", float(np.mean(net)), "#111827"))
    fig.add_trace(go.Scatter(x=tl, y=net, name="Net", line=dict(color="#111827", width=2),
                             hovertemplate="%{y:.0f} W/m²<extra>Net</extra>"),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=[m[0] for m in means], y=[m[1] for m in means],
                         marker_color=[m[2] for m in means], showlegend=False,
                         hovertemplate="%{y:.1f} W/m²<extra>mean</extra>"),
                  row=1, col=2)
    fig.update_yaxes(title_text=f"Heat flux into the water [{model.budget_unit}]",
                     row=1, col=1)
    fig.update_yaxes(title_text="Mean over the run", row=1, col=2)
    fig.update_layout(height=380, hovermode="x unified",
                      margin=dict(l=60, r=20, t=10, b=30))
    return fig


def rating_figure(meas, fit, reach, q_now):
    fig = go.Figure()
    used = fit["used"] if fit else meas.iloc[:0]
    others = meas.drop(used.index, errors="ignore")
    fig.add_trace(go.Scatter(x=others["discharge"], y=others["depth"], mode="markers",
                             name="other measurements",
                             marker=dict(color="#d1d5db", size=5)))
    fig.add_trace(go.Scatter(x=used["discharge"], y=used["depth"], mode="markers",
                             name="used in the fit", marker=dict(color=OBS_COLOR, size=6),
                             text=used["time"].dt.strftime("%Y-%m-%d"),
                             hovertemplate="Q %{x:.3g} m³/s, d %{y:.3g} m<br>%{text}"
                                           "<extra></extra>"))
    qq = np.geomspace(max(meas["discharge"].min(), 1e-3), meas["discharge"].max(), 60)
    fig.add_trace(go.Scatter(x=qq, y=[reach_depth(q, reach) for q in qq],
                             name="current d = c·Q^f", line=dict(color="#7c3aed", width=2)))
    fig.add_trace(go.Scatter(x=[q_now], y=[reach_depth(q_now, reach)], mode="markers",
                             name="steady Q", marker=dict(color="#7c3aed", size=12,
                                                          symbol="star")))
    fig.update_xaxes(type="log", title_text="Q [m³/s]")
    fig.update_yaxes(type="log", title_text="Mean depth A/W [m]")
    fig.update_layout(height=330, margin=dict(l=60, r=20, t=10, b=40))
    return fig


# ----------------------------------------------------------------------------
# Sidebar: gauge, simulation window, reach
# ----------------------------------------------------------------------------
st.session_state.setdefault("site_no", DEFAULT_SITE)
st.session_state.setdefault("use::RNSTM", True)

with st.sidebar:
    st.header("USGS gauge")
    site_no = st.text_input("Site number", key="site_no").strip()
    with st.expander("Find a gauge with water temperature"):
        state = st.selectbox("State", STATES, index=STATES.index(DEFAULT_STATE))
        try:
            gauges = load_gauge_list(state)
        except Exception as exc:
            gauges = None
            st.error(f"Gauge search failed: {exc}")
        if gauges is not None and len(gauges):
            gauges = gauges.sort_values("name")
            labels = {r.site_no: f"{r.site_no} · {r.name} "
                                 f"({r.begin[:4]}–{r.end[:4]}"
                                 f"{', Q' if r.has_q else ''})"
                      for r in gauges.itertuples()}
            pick = st.selectbox(f"{len(gauges)} active gauges", list(labels),
                                format_func=labels.get)
            st.button("Use this gauge", on_click=_use_gauge, args=(pick,),
                      icon=":material/check:")
        elif gauges is not None:
            st.caption("No active gauges with water temperature in this state.")

    st.header("Simulation")
    start = st.date_input("Start date (local standard time)", value=DEFAULT_START,
                          min_value=date(2008, 1, 1), max_value=date.today())
    days = st.number_input("Length [days]", min_value=1, max_value=60,
                           value=DEFAULT_DAYS, step=1)

try:
    gauge = load_site(site_no)
except Exception as exc:
    st.error(f"Could not read USGS site {site_no!r}: {exc}")
    st.stop()

t0 = pd.Timestamp(start, tz="UTC") - pd.Timedelta(hours=gauge.utc_offset_h)
t1 = t0 + pd.Timedelta(days=int(days))
try:
    obs_utc = load_observations(gauge.site_no, t0, t1)
except Exception as exc:
    st.warning(f"USGS observations unavailable: {exc}")
    obs_utc = pd.DataFrame(index=pd.date_range(t0, t1, freq="1h"))
try:
    meas = load_measurements(gauge.site_no)
except Exception as exc:
    st.warning(f"USGS field measurements unavailable: {exc}")
    meas = pd.DataFrame(columns=["time", "discharge", "width", "area",
                                 "velocity", "depth"])
try:
    meteo_utc = load_meteo(gauge.lat, gauge.lon, t0, t1, tuple(POWER_PARAMS))
except Exception as exc:
    st.error(f"NASA POWER request failed: {exc}")
    st.stop()

fit = fit_depth_rating(meas, since=pd.Timestamp(start) - pd.DateOffset(years=RATING_YEARS))

# Gauge-dependent defaults: reset when the gauge or the window changes.
q_obs = obs_utc["discharge"] if "discharge" in obs_utc else None
q_start = (float(q_obs.iloc[:24].median()) if q_obs is not None
           and q_obs.iloc[:24].notna().any() else 10.0)
q_start = round(max(q_start, 0.01), 3)
col = CONSTITUENT.obs_column
first_obs = (obs_utc[col].iloc[:6].dropna() if col in obs_utc else pd.Series(dtype=float))
T0_default = round(float(first_obs.iloc[0]) if len(first_obs)
                   else float(meteo_utc["t_air"].iloc[:24].mean()), 2)
ctx = (gauge.site_no, str(start), int(days))
_set_default("q_steady", q_start, "ctx::q", ctx)
_set_default("T0", T0_default, "ctx::T0", ctx)
if st.session_state.get("ctx::rating") != gauge.site_no:
    st.session_state["ctx::rating"] = gauge.site_no
    for p in REACH_PARAMS:
        st.session_state[f"reach::{p.name}"] = p.default
    if fit:
        _apply_rating(fit)

with st.sidebar:
    st.header("Streamflow")
    q_steady = st.number_input("Steady streamflow Q [m³/s]", min_value=0.001,
                               step=_step(q_start), format="%g", key="q_steady",
                               help="Held constant over the run. Defaults to the "
                                    "gauge's median Q over the first day.")
    use_obs_q = st.toggle("Drive with the observed USGS Q instead",
                          value=False, disabled=q_obs is None,
                          help="Uses the gauge's hourly discharge as the "
                               "streamflow forcing (gaps filled, floored at "
                               "0.001 m³/s).")

    st.header("Initial state")
    T0 = st.number_input(f"{CONSTITUENT.label} at start [{CONSTITUENT.unit}]",
                         step=0.1, format="%g", key="T0",
                         help="Defaults to the gauge's first observation, or the "
                              "first-day mean air temperature.")

    st.header("Channel depth")
    st.latex(r"d = \max(c_{depth}\,Q^{f_{depth}},\ d_{min})")
    c1, c2 = st.columns(2)
    reach = []
    for k, p in enumerate(REACH_PARAMS):
        with (c1 if k % 2 == 0 else c2):
            reach.append(st.number_input(f"{p.name} [{p.unit}]", min_value=0.0,
                                         step=_step(p.default), format="%g",
                                         help=p.help, key=f"reach::{p.name}"))
    reach = tuple(reach)
    if fit:
        st.caption(f"USGS field-measurement fit: d = {fit['c_depth']:.3g}·"
                   f"Q^{fit['f_depth']:.3g} (R² {fit['r2']:.2f}, "
                   f"{fit['n']} visits).")
        st.button("Use the USGS fit", on_click=_apply_rating, args=(fit,),
                  icon=":material/straighten:")
    else:
        st.caption("No USGS channel measurements to fit the depth rating; "
                   "set c_depth and f_depth by hand.")


# ----------------------------------------------------------------------------
# Forcing and observation series for the run
# ----------------------------------------------------------------------------
records = meteo_utc.iloc[:-1]                    # one record per hour of the run
if use_obs_q and q_obs is not None:
    q_series = q_obs.reindex(records.index).interpolate(limit_direction="both")
    q_forcing = np.clip(q_series.fillna(q_steady).to_numpy(float), 0.001, None)
    q_negative = int((q_obs < 0).sum())
else:
    q_forcing, q_negative = float(q_steady), 0

off = gauge.utc_offset_h
obs = obs_utc.set_axis(_local(obs_utc.index, off))
meteo_l = meteo_utc.set_axis(_local(meteo_utc.index, off))
sim_index = pd.date_range(t0, t1, freq="1h")
tl = _local(sim_index, off)
q_model = (np.r_[q_forcing, q_forcing[-1]] if np.ndim(q_forcing)
           else np.full(len(tl), q_forcing))
depth_model = np.array([reach_depth(q, reach) for q in q_model])
depth_from_obs_q = (obs["discharge"].apply(lambda q: reach_depth(q, reach))
                    .where(obs["discharge"] > 0)            # NaN or reversed flow
                    if "discharge" in obs else None)
meas_in_window = None
if len(meas):
    mw = meas[(meas["time"] >= t0) & (meas["time"] <= t1)]
    meas_in_window = mw.assign(time=_local(pd.DatetimeIndex(mw["time"]), off))


# ----------------------------------------------------------------------------
# Main: gauge summary, models | results
# ----------------------------------------------------------------------------
st.title("Stream temperature explorer")
area = (f" · {gauge.drainage_area_km2:,.0f} km²"
        if np.isfinite(gauge.drainage_area_km2) else "")
st.markdown(
    f"**{gauge.name}** · [USGS {gauge.site_no}]"
    f"(https://waterdata.usgs.gov/monitoring-location/USGS-{gauge.site_no}/)"
    f" · {gauge.lat:.4f}, {gauge.lon:.4f}{area} · times in {gauge.tz} "
    f"(UTC{off:+d}) · forcings: NASA POWER hourly")

have = {"Q": "discharge" in obs, "gage height": "gage_height" in obs,
        "mean depth": "mean_depth" in obs, CONSTITUENT.label.lower(): col in obs}
st.caption("Gauge data in this window: " + " · ".join(
    f"{'✅' if ok else '—'} {k}" for k, ok in have.items())
    + (f" · {len(meas_in_window)} field measurement(s)"
       if meas_in_window is not None and len(meas_in_window) else ""))
if q_negative:
    st.caption(f"⚠️ The gauge reports negative Q in {q_negative} h of the window "
               "(flow reversal); the forcing floors it at 0.001 m³/s.")

models = list(MODELS.values())
n_selected = sum(bool(st.session_state.get(f"use::{m.name}")) for m in models)
left, right = st.columns([1, 2.4], gap="large")

selected = []
with left:
    st.subheader(f"Models ({n_selected}/{MAX_MODELS})")
    for model in models:
        key = f"use::{model.name}"
        c_box, c_eq = st.columns([3, 2], vertical_alignment="center")
        on = c_box.checkbox(model.name, key=key, help=model.description,
                            disabled=n_selected >= MAX_MODELS
                                     and not st.session_state.get(key))
        shown = st.session_state.get(f"eqs::{model.name}", False)
        c_eq.button("Equations", key=f"eqs_btn::{model.name}",
                    icon=":material/functions:",
                    type="primary" if shown else "secondary",
                    on_click=_toggle_equations, args=(model.name,))
        if on:
            with st.expander(f"{model.name} parameters", expanded=True,
                             key=f"{model.name}::expander"):
                st.caption(model.description)
                gp = model_inputs(model)
            selected.append((model, gp))
    st.map(pd.DataFrame({"lat": [gauge.lat], "lon": [gauge.lon]}), zoom=8,
           height=220, size=600)

with right:
    results, steps, scores = [], {}, []
    gaps = missing_report(records, sorted({f for m, _ in selected for f in m.forcings}))
    if gaps:
        st.error("NASA POWER has no data for part of this window: " + "; ".join(
            f"{FIELD_LABELS[f][0]} ({n} h, from {_local(pd.DatetimeIndex([t]), off)[0]:%Y-%m-%d %H:%M})"
            for f, (n, t) in gaps.items())
            + ". Its radiation and cloud products lag about three months; "
              "pick an earlier start date.")
    elif not selected:
        st.info("Select up to three models on the left to simulate them.")
    else:
        for model, gp in selected:
            try:
                res = run_model(model.name, records, q_forcing, reach, gp, (float(T0),))
                results.append((model.name, res))
            except Exception as exc:                     # solver blow-ups etc.
                st.error(f"{model.name} failed: {exc}")

    if col in obs:
        for name, res in results:
            sim = pd.Series(res["value"], index=tl)
            steps[name] = stepwise(sim, obs[col], CONSTITUENT.rel_error_floor)
            scores.append({"Model": name} | summary(steps[name]))

    c1, c2, c3 = st.columns(3)
    show_forcings = c1.toggle("Show forcings", value=False)
    show_budget = c2.toggle("Show heat budget", value=False, disabled=not results)
    show_air = c3.toggle("Air temperature on the plot", value=False)
    st.plotly_chart(make_figure(tl, q_model, depth_model, obs, depth_from_obs_q,
                                meas_in_window, results, steps, meteo_l, show_air),
                    width="stretch")
    st.caption("Model depth uses the steady (or forcing) Q; the dotted line is "
               "the same rating applied to the observed Q. Gage height (right "
               "axis) is the stage above the gauge datum, not the water depth.")

    if scores:
        table = pd.DataFrame(scores).set_index("Model").rename(columns={
            "n": "n [h]", "MRE": "Mean rel. error [%]",
            "MARE": "Mean |rel. error| [%]", "bias": f"Bias [{CONSTITUENT.unit}]",
            "MAE": f"MAE [{CONSTITUENT.unit}]", "RMSE": f"RMSE [{CONSTITUENT.unit}]"})
        st.dataframe(table.style.format("{:.3g}").format("{:.0f}", subset=["n [h]"]),
                     width="stretch")
        st.caption("Relative error per step: (sim − obs) / obs · 100, hourly, "
                   f"skipping |obs| < {CONSTITUENT.rel_error_floor} "
                   f"{CONSTITUENT.unit}. NSE and KGE: 1 is perfect.")
    elif results:
        st.info(f"This gauge reports no {CONSTITUENT.label.lower()} in the "
                "window, so the errors are not computed.")

    if results:
        out = pd.DataFrame(index=pd.Index(tl, name=f"time_{gauge.tz}"))
        out["q_model_m3s"], out["depth_model_m"] = q_model, depth_model
        if col in obs:
            out[f"{col}_obs"] = obs[col].reindex(tl)
        for name, res in results:
            out[f"{name}_{CONSTITUENT.key}"] = res["value"]
            if name in steps:
                out[f"{name}_rel_error_pct"] = steps[name]["rel_error"].to_numpy()
        st.download_button("Download results (CSV)", out.to_csv().encode(),
                           file_name=f"{gauge.site_no}_{start:%Y%m%d}_{CONSTITUENT.key}.csv",
                           mime="text/csv", icon=":material/download:")

if show_forcings:
    st.subheader("Forcings · NASA POWER hourly")
    st.plotly_chart(forcing_figure(meteo_l, results, tl), width="stretch")
    st.caption("Values are hourly means, held constant over each hour by the "
               "models. The dotted long-wave line is the one the model computes "
               "from air temperature, humidity and cloud cover.")

if show_budget and results:
    st.subheader("Heat budget")
    names = [n for n, _ in results]
    pick = st.radio("Model", names, horizontal=True) if len(names) > 1 else names[0]
    res = dict(results)[pick]
    st.plotly_chart(budget_figure(MODELS[pick], res, tl), width="stretch")
    f = res["fluxes"]
    st.caption(f"Positive = heat into the water. Mean evaporation "
               f"{np.mean(f['evap_mm_d']):.2f} mm/day; mean warming rate "
               f"{np.mean(f['dTdt_day']):+.3f} °C/day over a depth of "
               f"{np.mean(f['depth']):.2f} m.")

if len(meas):
    with st.expander("Depth rating · USGS field measurements"):
        st.plotly_chart(rating_figure(meas, fit, reach, float(q_model[0])),
                        width="stretch")
        st.caption(f"Mean depth = area / width of each USGS discharge "
                   f"measurement. The fit uses visits from "
                   f"{start.year - RATING_YEARS} on (all of them if fewer than "
                   "10).")

for model in models:
    if st.session_state.get(f"eqs::{model.name}"):
        equations_panel(model)

st.divider()
st.markdown("##### Reference")
st.markdown(
    "García-Múnera, V., Velásquez Giron, N., Molina, R. D., Weber, L. J., & "
    "Krajewski, W. F. (2026). RNSTM: A Network-Link Approach to Streamflow "
    "Temperature Modeling. *JAWRA Journal of the American Water Resources "
    "Association*, 62(4), e70140. "
    "[https://doi.org/10.1111/1752-1688.70140]"
    "(https://doi.org/10.1111/1752-1688.70140)")
