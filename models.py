"""Single-reach stream water-quality models.

Every model is registered in :data:`MODELS` by name, so the app (and any
script) can pick and compare them interchangeably::

    from models import get_model, simulate
    model = get_model("RNSTM")
    res = simulate(model, forcing, reach=(1.2, 0.2, 0.1), y0=(28.0,))

A model simulates one **constituent** (water temperature now; salinity,
dissolved oxygen, ... later). The constituent carries what the app needs to
plot it and to compare it with a USGS gauge (unit, parameter code).

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
* ``params``         : per-reach params (:data:`REACH_PARAMS`), as asynch's
                       local params -> [c_depth, f_depth, d_min]
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


TEMPERATURE = Constituent("temperature", "Water temperature", "°C",
                          usgs_pcode="00010", obs_column="water_temp",
                          rel_error_floor=0.5)
CONSTITUENTS = {c.key: c for c in (TEMPERATURE,)}


@dataclass(frozen=True)
class Param:
    name: str
    unit: str
    default: float
    help: str = ""
    advanced: bool = False   # physical constants: hidden by default in the app


# Per-reach parameters, shared by every model: hydraulic-geometry depth
# d = max(c_depth * Q**f_depth, d_min) (RNSTM local params 3, 4 in asynch).
REACH_PARAMS = (
    Param("c_depth", "m/(m³/s)^f", 0.30,
          "Depth coefficient of d = c_depth·Q^f_depth."),
    Param("f_depth", "-", 0.40,
          "Depth exponent of d = c_depth·Q^f_depth (0.3–0.5 is typical)."),
    Param("d_min", "m", 0.10, "Minimum depth (asynch uses 0.1 m)."),
)


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
    # Equations, for display: ((LaTeX lhs, LaTeX rhs), ...)
    equations: Tuple[Tuple[str, str], ...] = ()
    definitions: Tuple[Tuple[str, str], ...] = ()
    # Forcing fields the model reads (the app checks they have no gaps).
    forcings: Tuple[str, ...] = ()

    @property
    def dim(self):
        return len(self.state_names)

    @property
    def default_params(self):
        return tuple(p.default for p in self.params)


MODELS = {}


def register_model(name, constituent, state_names, params, fluxes,
                   description="", budget=(), budget_unit="", equations=(),
                   definitions=(), forcings=()):
    """Decorator: register a dydt function as a named model."""
    def deco(fn):
        MODELS[name] = WQModel(
            name=name, constituent=constituent, dydt=fn, fluxes=fluxes,
            state_names=tuple(state_names), params=tuple(params),
            description=description, budget=tuple(budget),
            budget_unit=budget_unit, equations=tuple(equations),
            definitions=tuple(definitions), forcings=tuple(forcings))
        return fn
    return deco


def get_model(name):
    """Look up a registered model by name (raises if unknown)."""
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; available: {list(MODELS)}")
    return MODELS[name]


def reach_depth(q, params):
    """Hydraulic-geometry depth [m] for streamflow ``q`` [m^3/s]."""
    c_depth, f_depth, d_min = params
    return max(c_depth * max(q, 0.0) ** f_depth, d_min)


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
    reach : (c_depth, f_depth, d_min), see :data:`REACH_PARAMS`
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
    equations=(TEMPERATURE_EQUATIONS[:2]
               + ((r"H_l", r"0.97\,R_{lw} - 0.97\,\sigma\,(T_w + 273.2)^4"),)
               + TEMPERATURE_EQUATIONS[3:]),
    definitions=(TEMPERATURE_DEFINITIONS[0],) + TEMPERATURE_DEFINITIONS[2:],
    forcings=("sw_down", "lw_down", "t_air", "rh", "wind2", "pressure",
              "rain"),
)(_temperature_dydt(_fluxes_rnstm_lw))
