"""Single-reach stream water-quality models.

Every model is registered in :data:`MODELS` by name, so the app (and any
script) can pick and compare them interchangeably::

    from models import get_model, simulate
    model = get_model("RNSTM")
    res = simulate(model, forcing, reach=(1.2, 0.2, 0.1, 5.0, 0.5), y0=(28.0,))

A model simulates one **constituent** (water temperature, salinity; dissolved
oxygen, ... later). The constituent carries what the app needs to plot it and
to compare it with a USGS gauge (unit, parameter code, unit conversion).

Adding a new model
------------------
Each model is built from a *flux function* and a *RHS function*:

* ``fluxes(t, y, params, global_params, forcing) -> dict``
      returns every internal flux/rate. This is the single source of truth.
* ``dydt(t, y, params, global_params, forcing)``
      assembles dy/dt from those fluxes; this is what ``solve_ivp`` integrates.

Register the pair with :func:`register_model`. See docs/adding_a_model.md.

RHS contract
------------
* ``t``              : time [min] since the start of the run
* ``y``              : state vector, length ``dim``
* ``params``         : per-reach params (:data:`REACH_PARAMS`)
                       -> [c_depth, f_depth, d_min, c_width, f_width]
* ``global_params``  : the model's parameter vector (``model.params``)
* ``forcing``        : callable ``forcing(t) -> forcings.ForcingSample``
"""

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Constituent:
    """A simulated water-quality variable and how a gauge observes it."""
    key: str
    label: str
    unit: str
    usgs_pcode: str          # USGS parameter code of the observation
    obs_column: str          # its column in usgs.instantaneous_values()
    rel_error_floor: float   # |obs| below this is left out of relative errors
    obs_factor: float = 1.0  # observation [gauge unit] * obs_factor -> unit
    obs_note: str = ""       # how the observation is converted, for the app


TEMPERATURE = Constituent("temperature", "Water temperature", "°C",
                          usgs_pcode="00010", obs_column="water_temp",
                          rel_error_floor=0.5)
# Salinity as total dissolved solids, the units of the salt routing scheme.
# USGS reports specific conductance at 25 degC (00095); TDS ~ 0.65 SC is the
# usual conversion (0.55-0.75 depending on the ions).
SALINITY = Constituent("salinity", "Salinity (TDS)", "mg/L",
                       usgs_pcode="00095", obs_column="spec_cond",
                       rel_error_floor=1.0, obs_factor=0.65,
                       obs_note="TDS = 0.65 × USGS specific conductance "
                                "at 25 °C [µS/cm]")
CONSTITUENTS = {c.key: c for c in (TEMPERATURE, SALINITY)}


@dataclass(frozen=True)
class Param:
    name: str
    unit: str
    default: float
    help: str = ""
    advanced: bool = False   # physical constants: hidden by default in the app
    # The app sets it to the initial state (the gauge's first observation)
    # whenever the gauge or the window changes, e.g. an inflow concentration.
    follows_initial: bool = False


# Per-reach parameters, shared by every model: hydraulic geometry
# d = max(c_depth * Q**f_depth, d_min) (RNSTM local params 3, 4 in asynch) and
# w = c_width * Q**f_width (used by the salinity model for the reach volume).
REACH_PARAMS = (
    Param("c_depth", "m/(m³/s)^f", 0.30,
          "Depth coefficient of d = c_depth·Q^f_depth."),
    Param("f_depth", "-", 0.40,
          "Depth exponent of d = c_depth·Q^f_depth (0.3–0.5 is typical)."),
    Param("d_min", "m", 0.10, "Minimum depth (asynch uses 0.1 m)."),
    Param("c_width", "m/(m³/s)^f", 5.0,
          "Width coefficient of w = c_width·Q^f_width."),
    Param("f_width", "-", 0.50,
          "Width exponent of w = c_width·Q^f_width (about 0.5 is typical)."),
)
W_MIN = 0.1            # minimum reach width [m]


@dataclass(frozen=True)
class WQModel:
    """Metadata + right-hand side + flux extractor of one model."""
    name: str
    constituent: Constituent
    dydt: Callable
    fluxes: Callable
    state_names: Tuple[str, ...]
    params: Tuple[Param, ...]
    description: str = ""
    # Budget terms for the app's budget plot: ((flux key, label, sign), ...);
    # sign +1 = gain, -1 = loss of the state.
    budget: Tuple[Tuple[str, str, int], ...] = ()
    budget_unit: str = ""
    budget_title: str = ""
    # Equations, for display: ((LaTeX lhs, LaTeX rhs), ...)
    equations: Tuple[Tuple[str, str], ...] = ()
    definitions: Tuple[Tuple[str, str], ...] = ()
    notation: str = ""                    # symbols and units of the equations
    # Forcing fields the model reads (the app checks they have no gaps).
    forcings: Tuple[str, ...] = ()
    # summary(fluxes) -> one sentence about a run, under the budget plot
    summary: Optional[Callable] = None

    @property
    def dim(self):
        return len(self.state_names)

    @property
    def default_params(self):
        return tuple(p.default for p in self.params)


MODELS = {}


def register_model(name, constituent, state_names, params, fluxes,
                   description="", budget=(), budget_unit="", budget_title="",
                   equations=(), definitions=(), notation="", forcings=(),
                   summary=None):
    """Decorator: register a dydt function as a named model."""
    def deco(fn):
        MODELS[name] = WQModel(
            name=name, constituent=constituent, dydt=fn, fluxes=fluxes,
            state_names=tuple(state_names), params=tuple(params),
            description=description, budget=tuple(budget),
            budget_unit=budget_unit, budget_title=budget_title,
            equations=tuple(equations), definitions=tuple(definitions),
            notation=notation, forcings=tuple(forcings), summary=summary)
        return fn
    return deco


def get_model(name):
    """Look up a registered model by name (raises if unknown)."""
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; available: {list(MODELS)}")
    return MODELS[name]


def reach_depth(q, params):
    """Hydraulic-geometry depth [m] for streamflow ``q`` [m^3/s]."""
    c_depth, f_depth, d_min = params[:3]
    return max(c_depth * max(q, 0.0) ** f_depth, d_min)


def reach_width(q, params):
    """Hydraulic-geometry width [m] for streamflow ``q`` [m^3/s]."""
    c_width, f_width = params[3:5]
    return max(c_width * max(q, 0.0) ** f_width, W_MIN)


def flux_timeseries(model, t, y, params, global_params, forcing):
    """Evaluate a model's fluxes at every time of a solution.

    ``t`` has shape (N,) [min] and ``y`` shape (dim, N). Returns one array
    of shape (N,) per flux key.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    y = np.atleast_2d(np.asarray(y, dtype=float))
    out = {}
    for k in range(t.size):
        f = model.fluxes(t[k], y[:, k], params, global_params, forcing)
        for key, val in f.items():
            if key not in out:
                out[key] = np.empty(t.size)
            out[key][k] = val
    return out


def simulate(model, forcing, reach, y0, global_params=None, t_end=None,
             t_eval=None):
    """Run one single-reach simulation.

    Parameters
    ----------
    model : WQModel
    forcing : forcings.HourlyForcing
    reach : (c_depth, f_depth, d_min, c_width, f_width), see
        :data:`REACH_PARAMS`
    y0 : initial state, length ``model.dim``
    global_params : the model's parameters; defaults to its defaults
    t_end : simulation length [min]; defaults to the end of the forcing
    t_eval : times [min] at which to store the solution (default: hourly)

    Returns
    -------
    dict with ``t`` [min], ``y`` (dim, N), ``fluxes`` (dict of arrays) and
    ``value`` (the constituent, i.e. the first state, shape (N,)).
    """
    gp = tuple(float(v) for v in
               (model.default_params if global_params is None else global_params))
    params = tuple(float(v) for v in reach)
    t_end = forcing.t_end if t_end is None else float(t_end)
    if t_eval is None:
        t_eval = np.arange(0.0, t_end + 1e-9, 60.0)

    # Forcings are piecewise constant over each record, so max_step keeps the
    # solver from stepping across several records at once.
    sol = solve_ivp(model.dydt, (0.0, t_end), np.asarray(y0, dtype=float),
                    args=(params, gp, forcing), method="LSODA", rtol=1e-6,
                    atol=1e-6, max_step=forcing.dt / 2, t_eval=t_eval)
    if not sol.success:
        raise RuntimeError(f"{model.name}: {sol.message}")
    f = flux_timeseries(model, sol.t, sol.y, params, gp, forcing)
    return {"t": sol.t, "y": sol.y, "fluxes": f, "value": sol.y[0]}


# ----------------------------------------------------------------------------
# Water temperature -- RNSTM, River Network Streamflow Temperature Model
# (Garcia-Munera et al., 2026, JAWRA, doi:10.1111/1752-1688.70140)
# ----------------------------------------------------------------------------
# Air-water interface heat budget of RNSTM, ported from its asynch
# implementation (src/models/equations.c, Temperature_Model and its helpers)
# for a single reach (no upstream links, so the mixing temperature M of the
# C code equals the reach temperature). Differences with the C code, all
# documented in docs/temperature_model.md:
#   * solar_radiation() uses the incoming radiation (the C code multiplies an
#     uninitialised variable);
#   * the air saturation vapour pressure of the long-wave term uses the air
#     temperature only (the C code mixes T_air and T_water);
#   * evaporation uses the air vapour pressure e_a = RH * e_s(T_air) (the C
#     code uses RH * e_s(T_water));
#   * calib_a is in degC/day (degC/min in the C code);
#   * the water cannot cool below 0 degC (no ice model);
#   * rainfall mixes into the reach as a heat flux H_r at the rain
#     temperature T_rain = max(T_air - dT_rain, 0): the ODE form of the
#     paper's advective step (Eqs. 14-16), with k_rain * P standing for the
#     rain-driven overland flow q_pL per unit water surface. The paper and the
#     coupled asynch model (dist_v2_temperature) use T_r = T_air - 1.5 only
#     inside the mixed temperature T_c at which the other fluxes are evaluated
#     (Eqs. 17-19), which by itself warms rather than cools the water.
def _e_sat(T):
    """Saturation vapour pressure [mbar] at temperature T [degC]."""
    return 6.1275 * math.exp(17.27 * T / (237.3 + T))


def _make_temperature_fluxes(longwave_from_forcing):
    def fluxes(t, y, params, global_params, forcing):
        (sigma, rho, c_w, SF, a_w, b_w, calib_m, calib_a,
         dT_rain, k_rain) = global_params
        f = forcing(t)
        T_w = y[0]

        T_a = f.t_air
        rh = min(f.rh, 100.0) * 0.01             # %    -> [0, 1]
        cloud = f.cloud * 0.01                   # %    -> [0, 1]
        p_a = f.pressure * 0.01                  # Pa   -> mbar
        e_a = rh * _e_sat(T_a)                   # air vapour pressure [mbar]
        e_w = _e_sat(T_w)                        # saturated at the surface

        # Solar (albedo 0.03) attenuated by shade
        H_s = 0.97 * f.sw_down * (1.0 - SF)
        # Long-wave: atmospheric (absorbed, emissivity of water 0.97) - back
        if longwave_from_forcing:
            eps_a = math.nan
            H_atm = 0.97 * f.lw_down
        else:
            eps_a = (1.72 * ((e_a * 0.1) / (273.2 + T_a)) ** (1.0 / 7.0)
                     * (1.0 + 0.22 * cloud ** 2))
            H_atm = 0.97 * sigma * eps_a * (T_a + 273.2) ** 4
        H_back = 0.97 * sigma * (T_w + 273.2) ** 4
        H_l = H_atm - H_back
        # Evaporation (mass transfer) and convection (Bowen ratio)
        L_e = 1000.0 * (2501.4 + 1.83 * T_w)     # [J/kg], as in the C code
        f_w = a_w + b_w * f.wind2                # wind function [m/(s mbar)]
        evap = f_w * (e_w - e_a)                 # [m/s]
        H_e = rho * L_e * evap
        H_c = 0.61 * (p_a / 1000.0) * rho * L_e * f_w * (T_w - T_a)
        # Rain mixing: k_rain * P per unit water surface enters at T_rain
        T_rain = max(T_a - dT_rain, 0.0)
        rain = k_rain * f.rain * (1e-3 / 3600.0)  # mm/h -> m/s
        H_r = rho * c_w * rain * (T_rain - T_w)

        H_net = H_s + H_l - H_e - H_c + H_r
        depth = reach_depth(f.q, params)
        dTdt = (H_net * 60.0 / (rho * c_w * depth) * calib_m
                + calib_a / 1440.0)              # [degC/min]
        return {
            "H_s": H_s, "H_atm": H_atm, "H_back": H_back, "H_l": H_l,
            "H_e": H_e, "H_c": H_c, "H_r": H_r, "H_net": H_net,
            "eps_a": eps_a, "T_rain": T_rain,
            "evap_mm_d": evap * 1000.0 * 86400.0,
            "depth": depth, "dTdt_day": dTdt * 1440.0,
        }
    return fluxes


def _temperature_dydt(fluxes):
    def dydt(t, y, params, global_params, forcing):
        rate = fluxes(t, y, params, global_params, forcing)["dTdt_day"] / 1440.0
        if y[0] <= 0.0 and rate < 0.0:           # freezing point
            rate = 0.0
        return [rate]
    return dydt


TEMPERATURE_PARAMS = (
    Param("sigma", "W/(m²K⁴)", 5.67e-8, "Stefan-Boltzmann constant.",
          advanced=True),
    Param("rho", "kg/m³", 997.0, "Water density.", advanced=True),
    Param("c_w", "J/(kg°C)", 4182.0, "Specific heat of water.", advanced=True),
    Param("SF", "-", 0.10,
          "Shade factor: fraction of the solar radiation blocked by "
          "riparian vegetation and topography (0 = open channel)."),
    Param("a_w", "m/(s·mbar)", 1.505e-9,
          "Wind-function intercept, f(W) = a_w + b_w·W, of evaporation and "
          "convection."),
    Param("b_w", "1/mbar", 1.6e-9, "Wind-function slope, f(W) = a_w + b_w·W."),
    Param("calib_m", "-", 1.0,
          "Multiplier of the heat-budget warming rate (asynch calib_m)."),
    Param("calib_a", "°C/day", 0.0,
          "Additive warming rate (asynch calib_a, there in °C/min)."),
    Param("dT_rain", "°C", 1.5,
          "How much colder than the air the rain is: T_rain = max(T_air − "
          "dT_rain, 0). RNSTM uses 1.5 °C (Byers et al., 1949)."),
    Param("k_rain", "-", 1.0,
          "Rain-driven inflow per unit of water surface, as a multiple of "
          "the rainfall: k_rain·P ≈ q_pL / (channel surface area). 1 = only "
          "rain falling on the water; > 1 adds the overland flow from the "
          "hillslope (≈ runoff coefficient × A_h / channel area)."),
)

TEMPERATURE_BUDGET = (
    ("H_s", "Solar H_s", +1),
    ("H_l", "Net long-wave H_l", +1),
    ("H_e", "Evaporation H_e", -1),
    ("H_c", "Convection H_c", -1),
    ("H_r", "Rain H_r", +1),
)

TEMPERATURE_EQUATIONS = (
    (r"\frac{dT_w}{dt}",
     r"m\,\frac{H_s + H_l - H_e - H_c + H_r}{\rho\,c_w\,d} + a"),
    (r"H_s", r"0.97\,(1 - SF)\,R_{sw}"),
    (r"H_l", r"0.97\,\sigma\,\varepsilon_a\,(T_a + 273.2)^4"
             r" - 0.97\,\sigma\,(T_w + 273.2)^4"),
    (r"H_e", r"\rho\,L_e\,f(W)\,\left(e_s(T_w) - e_a\right)"),
    (r"H_c", r"0.61\,\frac{p_a}{1000}\,\rho\,L_e\,f(W)\,(T_w - T_a)"),
    (r"H_r", r"\rho\,c_w\,k_{rain}\,P\,(T_{rain} - T_w)"),
    (r"d", r"\max\left(c_{depth}\,Q^{f_{depth}},\ d_{min}\right)"),
)

TEMPERATURE_DEFINITIONS = (
    (r"e_s(T)", r"6.1275\,\exp\left(\frac{17.27\,T}{237.3 + T}\right)"
                r"\ \text{[mbar]},\quad e_a = RH\,e_s(T_a)"),
    (r"\varepsilon_a", r"1.72\left(\frac{0.1\,e_a}{273.2 + T_a}\right)^{1/7}"
                       r"\left(1 + 0.22\,C^2\right)"),
    (r"f(W)", r"a_w + b_w\,W_2,\qquad L_e = 1000\,(2501.4 + 1.83\,T_w)"),
    (r"T_{rain}", r"\max\left(T_a - \Delta T_{rain},\ 0\right)"),
    (r"m,\ a", r"\text{calib\_m},\ \text{calib\_a}"),
)

TEMPERATURE_NOTATION = (
    "T_w, T_a: water and air temperature [°C]; R_sw, R_lw: incoming short- "
    "and long-wave radiation [W/m²]; RH: relative humidity [0–1]; C: cloud "
    "cover [0–1]; W₂: wind at 2 m [m/s]; p_a: pressure [mbar]; P: "
    "precipitation [m/s]; Q: streamflow [m³/s]; H terms in W/m².")


def _temperature_summary(f):
    return (f"Positive = heat into the water. Mean evaporation "
            f"{np.mean(f['evap_mm_d']):.2f} mm/day; mean warming rate "
            f"{np.mean(f['dTdt_day']):+.3f} °C/day over a depth of "
            f"{np.mean(f['depth']):.2f} m.")


_fluxes_rnstm = _make_temperature_fluxes(longwave_from_forcing=False)
_fluxes_rnstm_lw = _make_temperature_fluxes(longwave_from_forcing=True)

register_model(
    "RNSTM", TEMPERATURE, state_names=("T_w",),
    params=TEMPERATURE_PARAMS, fluxes=_fluxes_rnstm,
    description="River Network Streamflow Temperature Model heat budget for "
                "one reach: solar, net long-wave (atmospheric emissivity "
                "from air temperature, humidity and cloud cover), evaporation, "
                "convection and rain mixing, over a depth d = c_depth·Q^f_depth.",
    budget=TEMPERATURE_BUDGET, budget_unit="W/m²",
    budget_title="Heat flux into the water", notation=TEMPERATURE_NOTATION,
    summary=_temperature_summary,
    equations=TEMPERATURE_EQUATIONS, definitions=TEMPERATURE_DEFINITIONS,
    forcings=("sw_down", "t_air", "rh", "cloud", "wind2", "pressure", "rain"),
)(_temperature_dydt(_fluxes_rnstm))

register_model(
    "RNSTM_LWobs", TEMPERATURE, state_names=("T_w",),
    params=TEMPERATURE_PARAMS, fluxes=_fluxes_rnstm_lw,
    description="RNSTM with the incoming long-wave radiation taken from "
                "the forcing data (NASA POWER ALLSKY_SFC_LW_DWN) instead of "
                "the emissivity formula; cloud cover is then not used.",
    budget=TEMPERATURE_BUDGET, budget_unit="W/m²",
    budget_title="Heat flux into the water", notation=TEMPERATURE_NOTATION,
    summary=_temperature_summary,
    equations=(TEMPERATURE_EQUATIONS[:2]
               + ((r"H_l", r"0.97\,R_{lw} - 0.97\,\sigma\,(T_w + 273.2)^4"),)
               + TEMPERATURE_EQUATIONS[3:]),
    definitions=(TEMPERATURE_DEFINITIONS[0],) + TEMPERATURE_DEFINITIONS[2:],
    forcings=("sw_down", "lw_down", "t_air", "rh", "wind2", "pressure",
              "rain"),
)(_temperature_dydt(_fluxes_rnstm_lw))


# ----------------------------------------------------------------------------
# Salinity -- lumped salt routing of one reach (Davidson & Salbe, IQQM
# salinity routing upgrade; port of salt_segment.py)
# ----------------------------------------------------------------------------
# The reach is a fully mixed store (CSTR) of volume V and salt mass M = c V:
#     dM/dt = Q_in c_in + L - Q_out M / V          (Davidson eq. 1)
# For a single reach the storage follows the hydraulic geometry, V = w d L_reach
# at the current Q, and is held constant within a step (dV/dt = 0), so the
# outflow is Q_out = Q_in - Q_ev and the state is the concentration c = M / V:
#     V dc/dt = Q (c_bf - c) + Q_qf (c_qf - c) + Q_ev c + L
# Inflows are two end members (mix_end_members): the upstream flow Q at the
# baseflow concentration c_bf and the rain-driven quickflow Q_qf = k_rain P A_s
# at c_qf. Evaporation Q_ev = E A_s removes water but not salt; it uses the
# mass-transfer formula of RNSTM at the air temperature, since this model has
# no water temperature. Dry reaches cannot occur: V >= d_min w L_reach > 0.
def mix_end_members(q_bf, c_bf, q_qf, c_qf):
    """Two-end-member inflow concentration (baseflow + quickflow/lateral)."""
    q = q_bf + q_qf
    if q <= 0:
        return 0.0
    return (q_bf * c_bf + q_qf * c_qf) / q


def ec25_from_measured(ec_measured, temp_c, alpha=0.02):
    """Temperature-compensate electrical conductivity to 25 degC.

    USGS specific conductance (00095) is already at 25 degC; use this for
    uncompensated EC records before comparing them with the model.
    """
    return ec_measured / (1.0 + alpha * (temp_c - 25.0))


def fluxes_salt(t, y, params, global_params, forcing):
    L_reach, c_bf, c_qf, k_rain, load, a_w, b_w = global_params
    f = forcing(t)
    c = y[0]

    q = max(f.q, 0.0)
    depth = reach_depth(q, params)
    width = reach_width(q, params)
    area = width * L_reach                       # water surface A_s [m^2]
    V = area * depth                             # reach volume [m^3]

    q_qf = k_rain * f.rain * (1e-3 / 3600.0) * area       # mm/h -> m3/s
    rh = min(f.rh, 100.0) * 0.01
    evap = max((a_w + b_w * f.wind2) * _e_sat(f.t_air) * (1.0 - rh), 0.0)
    q_ev = evap * area                           # [m^3/s]
    q_in = q + q_qf
    q_out = max(q_in - q_ev, 0.0)
    c_in = mix_end_members(q, c_bf, q_qf, c_qf)

    per_day = 86400.0 / V                        # (g/s) / V -> mg/L per day
    adv = q * (c_bf - c) * per_day
    rain = q_qf * (c_qf - c) * per_day
    evap_term = (q_in - q_out) * c * per_day
    load_term = load * per_day
    return {
        "adv": adv, "rain": rain, "evap": evap_term, "load": load_term,
        "dcdt_day": adv + rain + evap_term + load_term,
        "c_in": c_in, "q_qf": q_qf, "q_out": q_out,
        "evap_mm_d": evap * 1000.0 * 86400.0,
        "depth": depth, "width": width, "volume": V,
        "tau_h": V / max(q_out, 1e-9) / 3600.0,
    }


def model_salt(t, y, params, global_params, forcing):
    rate = fluxes_salt(t, y, params, global_params, forcing)["dcdt_day"] / 1440.0
    if y[0] <= 0.0 and rate < 0.0:               # no negative concentrations
        rate = 0.0
    return [rate]


SALT_PARAMS = (
    Param("L_reach", "m", 1000.0,
          "Reach length. With the width and depth it sets the volume "
          "V = w·d·L_reach and the water surface A_s = w·L_reach."),
    Param("c_bf", "mg/L", 500.0,
          "Baseflow end member: concentration of the upstream inflow Q. "
          "The app sets it to the initial concentration for each gauge.",
          follows_initial=True),
    Param("c_qf", "mg/L", 20.0,
          "Quickflow end member: concentration of the rain-driven inflow."),
    Param("k_rain", "-", 1.0,
          "Rain-driven inflow per unit of water surface, as a multiple of "
          "the rainfall (as in RNSTM). 1 = only rain falling on the water."),
    Param("load", "g/s", 0.0,
          "Net external salt load L (groundwater, tidal, point sources); "
          "negative for a withdrawal."),
    Param("a_w", "m/(s·mbar)", 1.505e-9,
          "Wind-function intercept of evaporation, as in RNSTM.", advanced=True),
    Param("b_w", "1/mbar", 1.6e-9,
          "Wind-function slope of evaporation, as in RNSTM.", advanced=True),
)


def _salt_summary(f):
    return (f"Positive = concentration rising. Mean residence time "
            f"{np.mean(f['tau_h']):.1f} h in a reach of "
            f"{np.mean(f['volume']):,.0f} m³ ({np.mean(f['width']):.1f} m wide, "
            f"{np.mean(f['depth']):.2f} m deep); mean inflow concentration "
            f"{np.mean(f['c_in']):.0f} mg/L; mean evaporation "
            f"{np.mean(f['evap_mm_d']):.2f} mm/day.")


register_model(
    "SaltCSTR", SALINITY, state_names=("c",),
    params=SALT_PARAMS, fluxes=fluxes_salt,
    description="Lumped salt routing (Davidson & Salbe, IQQM): the reach is a "
                "fully mixed store fed by the upstream flow at the baseflow "
                "concentration and by rain-driven quickflow at the quickflow "
                "concentration, concentrated by evaporation, plus an external "
                "load.",
    budget=(("adv", "Upstream inflow", +1), ("rain", "Rain dilution", +1),
            ("evap", "Evaporation", +1), ("load", "External load", +1)),
    budget_unit="mg/L/day", budget_title="Rate of change of c",
    equations=(
        (r"\frac{dc}{dt}",
         r"\frac{Q\,(c_{bf} - c) + Q_{qf}\,(c_{qf} - c) + Q_{ev}\,c + L}{V}"),
        (r"V", r"w\,d\,L_{reach},\qquad A_s = w\,L_{reach}"),
        (r"Q_{qf}", r"k_{rain}\,P\,A_s"),
        (r"Q_{ev}", r"E\,A_s,\qquad E = f(W)\,\left(e_s(T_a) - e_a\right)"),
        (r"c_{in}", r"\frac{Q\,c_{bf} + Q_{qf}\,c_{qf}}{Q + Q_{qf}}"),
        (r"w,\ d", r"c_{width}\,Q^{f_{width}},\quad "
                   r"\max\left(c_{depth}\,Q^{f_{depth}},\ d_{min}\right)"),
    ),
    definitions=(
        (r"e_s(T)", r"6.1275\,\exp\left(\frac{17.27\,T}{237.3 + T}\right)"
                    r"\ \text{[mbar]},\quad e_a = RH\,e_s(T_a)"),
        (r"f(W)", r"a_w + b_w\,W_2"),
    ),
    notation="c: salt concentration as TDS [mg/L]; Q: upstream streamflow "
             "[m³/s]; Q_qf, Q_ev: rain-driven inflow and evaporation [m³/s]; "
             "P: precipitation [m/s]; E: evaporation rate [m/s]; L: external "
             "load [g/s]; w, d, L_reach: reach width, depth and length [m]; "
             "T_a: air temperature [°C]; RH: relative humidity [0–1]; W₂: "
             "wind at 2 m [m/s].",
    forcings=("t_air", "rh", "wind2", "rain"),
    summary=_salt_summary,
)(model_salt)
