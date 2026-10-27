# RNSTM: River Network Streamflow Temperature Model

`RNSTM` in [`models.py`](../models.py) is the River Network Streamflow
Temperature Model (García-Múnera et al., 2026), applied to a single reach.
It is ported from the model's asynch implementation (`Temperature_Model` in
`src/models/equations.c`). `RNSTM_LWobs` is the same model, except that it
takes the incoming long-wave radiation from the forcing data.

The port solves the four air–water interface terms plus the cooling by rain
(see [Rainfall](#rainfall)). The published RNSTM also includes sub-surface
heat transfer, which neither the asynch `Temperature_Model` nor this port has
yet.

## Equations

The state is the water temperature `T_w` [°C]. Its rate is

```
dT_w/dt = calib_m · (H_s + H_l − H_e − H_c + H_r) / (ρ c_w d) + calib_a
```

| Term | Formula | Meaning |
|---|---|---|
| `H_s` | `0.97 (1 − SF) R_sw` | absorbed solar radiation (albedo 0.03, shade factor `SF`) |
| `H_l` | `0.97 σ ε_a (T_a + 273.2)⁴ − 0.97 σ (T_w + 273.2)⁴` | net long-wave: atmospheric minus back radiation |
| `ε_a` | `1.72 (0.1 e_a / (273.2 + T_a))^(1/7) (1 + 0.22 C²)` | atmospheric emissivity (Brutsaert with clouds) |
| `H_e` | `ρ L_e f(W) (e_s(T_w) − e_a)` | evaporation (mass transfer) |
| `H_c` | `0.61 (p_a / 1000) ρ L_e f(W) (T_w − T_a)` | convection (Bowen ratio) |
| `H_r` | `ρ c_w k_rain P (T_rain − T_w)` | rain mixing into the reach, `P` in m/s |
| `T_rain` | `max(T_a − ΔT_rain, 0)` | rain temperature [°C] |
| `f(W)` | `a_w + b_w W₂` | wind function [m/(s·mbar)] |
| `e_s(T)` | `6.1275 exp(17.27 T / (237.3 + T))` | saturation vapour pressure [mbar]; `e_a = RH · e_s(T_a)` |
| `L_e` | `1000 (2501.4 + 1.83 T_w)` | latent heat as written in the C code [J/kg] |
| `d` | `max(c_depth Q^f_depth, d_min)` | reach depth from hydraulic geometry [m] |

All `H` terms are in W/m², with positive values warming the water. Time is in
minutes inside the solver, as in asynch.

`RNSTM_LWobs` replaces the atmospheric term with `0.97 R_lw`, where
`R_lw` is NASA POWER `ALLSKY_SFC_LW_DWN`. It then does not use cloud cover.

## Parameters

| Index | Name | Default | Unit | Source of the default |
|---|---|---|---|---|
| 0 | `sigma` | 5.67e-8 | W/(m²K⁴) | physical constant |
| 1 | `rho` | 997 | kg/m³ | physical constant |
| 2 | `c_w` | 4182 | J/(kg °C) | physical constant |
| 3 | `SF` | 0.10 | – | assumed: mostly open channel |
| 4 | `a_w` | 1.505e-9 | m/(s·mbar) | Heat Source / HFLUX mass-transfer coefficients |
| 5 | `b_w` | 1.6e-9 | 1/mbar | idem |
| 6 | `calib_m` | 1 | – | no correction |
| 7 | `calib_a` | 0 | °C/day | no correction |
| 8 | `dT_rain` | 1.5 | °C | RNSTM paper, Eq. 14 (also hardcoded in asynch) |
| 9 | `k_rain` | 1 | – | only rain falling on the water surface (see [Rainfall](#rainfall)) |

Indices 0–7 match asynch's `global_params[0..7]`; 8 and 9 are new. The reach parameters
`c_depth`, `f_depth` (asynch local `params[3]`, `params[4]`) and `d_min`
belong to the reach, not the model. The app fits `c_depth` and `f_depth` to
the gauge's USGS field measurements, using mean depth = area / width.

The asynch repository has no reference values for these global parameters.
The defaults above are literature-based. Check that `a_w` and `b_w` match the
units your asynch `.gbl` files assume.

## Rainfall

### In the paper

RNSTM (García-Múnera et al., 2026, Section 2.2.1) treats rain as part of an
advective mixing step. Rain reaches a link as the overland flow from its
hillslope, `q_pL` [m³/s], at the rainfall temperature

```
T_r = T_a − 1.5 °C                     (Byers et al. 1949; van Beek et al. 2012)
```

Together with the subsurface flow `q_sL` at `T_s`, the snowmelt `q_snow` at
0 °C and the upstream links `q_i` at `T_c,i`, it is mixed with the channel
flow `q`:

```
M_e = q T_c + q_pL T_r + q_sL T_s + q_snow T_snow                    (14)
M_u = M_e + Σ q_i T_c,i                                               (15)
T_c = M_u / q_total,   q_total = q + q_pL + q_sL + q_snow + Σ q_i    (16)
```

The surface fluxes `H_l`, `H_e` and `H_c` are then evaluated at `T_c`
(Eq. 17), and `dT_w/dt = H_T / (60 ρ c_p d)` (Eqs. 18–19). In the paper's
experiments, including the rain temperature lowered the Cedar River outlet
temperature by about 1.5 °C after a large event. Small basins (< 100 km²)
were the most sensitive.

### In this port

A single reach has no hillslope model to provide `q_pL`. The port writes the
mixing step as a term of the ODE. Inflow at `T_rain` replaces part of the
reach water, so

```
dT_w/dt |rain = (q_pL / V) (T_rain − T_w),       V = channel area × d
```

Per unit of water surface, `q_pL / channel area = k_rain · P`, which gives the
heat flux

```
H_r = ρ c_w k_rain P (T_rain − T_w)        [W/m²]
T_rain = max(T_a − ΔT_rain, 0)
```

added to the budget with the other terms.

- `ΔT_rain` (`dT_rain`) is the paper's 1.5 °C by default. The floor at 0 °C
  comes from the asynch code: colder precipitation is snow, which the paper
  mixes in at 0 °C.
- `k_rain` stands in for `q_pL`. With 1, only rain falling on the water
  surface counts. A hillslope that sheds a fraction `r_c` of the rain as
  overland flow gives `k_rain ≈ r_c · A_h / (W L)` (hillslope area over
  channel area). For example, `r_c = 0.2`, `A_h = 0.1 km²` (the paper's
  hillslope size) and a 10 m × 500 m link give `k_rain ≈ 4`.
- `P` is NASA POWER `PRECTOTCORR`.

The rain water is not added to the streamflow: Q stays at its steady value
(or at the observed USGS Q).

As a sense of scale, at St. Johns River near Cocoa in June 2026 (depth about
1.3 m), `k_rain = 1` cools by at most about 40 W/m² in the heaviest hour, and
`k_rain = 10` by about 350 W/m².

### Why the port does not evaluate the fluxes at `T_c` only

Eq. 19 and the coupled asynch model (`dist_v2_temperature`, model type 603 on
the `temp_model` branch) have no `(T_c − T_w)` term. Rain enters only through
the temperature at which `H_l`, `H_e` and `H_c` are evaluated. Those losses
grow with temperature: at Cocoa in June 2026, the net surface flux drops by
about 54 W/m² per °C (17–128 W/m² depending on the hour). Evaluating them at
a colder `T_c` therefore *raises* the net heat input, and rain would warm
the stream. The 1.5 °C cooling reported in the paper needs the Step-1
update `T_w ← T_c` that the paper describes in words. In an ODE solver,
that update is the mixing term above. The C code may need it as well.

The asynch code also mixes in the active subsurface flows (`q_sLa`, `q_sLt`)
at `T_r`, while Eq. 14 uses the subsurface temperature `T_s` for `q_sL`.

## Differences from the C code

Porting the model surfaced three bugs in `equations.c`. They are fixed in
the Python version and should probably be fixed in asynch too. The fixed
versions match the paper's Eq. 4 (solar), Eqs. 6–8 (vapour pressures from the
air temperature) and Eq. 9 (`e_s` at the water surface minus `e_a`).

1. **`solar_radiation` ignores its input.** It contains
   `double hs = 0.97*hs*(1 -sf);`, which multiplies the uninitialised local
   `hs` instead of the argument `hsi`. `H_s` is therefore undefined behaviour
   (in practice garbage or 0). Fix: `double hs = 0.97*hsi*(1 - sf);`.
2. **The long-wave vapour pressure mixes temperatures.** It contains
   `es = 6.1275*exp(17.27*tair/(237.3 + twater))`, but both temperatures
   should be the air temperature: `17.27*tair/(237.3 + tair)`.
3. **Evaporation uses the water temperature for the air vapour pressure.**
   `ea_ = (rh/100)*es_` with `es_` evaluated at `twater` makes
   `es_ − ea_ = e_s(T_w)(1 − RH)`. The mass-transfer formula needs
   `e_s(T_w) − RH·e_s(T_a)`. Fix: compute `ea_` from `tair`.

Other differences:

- **`calib_a` is in °C/day** here (°C/min in C) so that it has a usable
  magnitude in the app. Divide by 1440 to get the C value.
- **Freezing floor.** The water does not cool below 0 °C, since there is no
  ice model. The C code has no such guard.
- **The densities hardcoded in C are parameters here.** The C helpers
  hardcode ρ = 997 instead of using `global_params[1]`. Python uses the
  parameter everywhere; with the default it gives the same result.
- **Single reach, no mixing.** The C code mixes the reach temperature with the
  parent links' temperatures, weighted by flow (`M`). A single reach has no
  parents, so `M = T_w`.
- **The RH comment is wrong in C.** It says RH is in [0, 1], but the code
  (`rh/100`, cap at 100) expects percent. NASA POWER `RH2M` is in percent,
  which matches the code.

## Forcings

NASA POWER hourly point data, in UTC. Each value is the mean over the hour
starting at its time stamp, and is held constant over that hour.

| Field | POWER parameter | Unit given to the model |
|---|---|---|
| `sw_down` | `ALLSKY_SFC_SW_DWN` | W/m² |
| `lw_down` | `ALLSKY_SFC_LW_DWN` | W/m² |
| `t_air` | `T2M` | °C |
| `rh` | `RH2M` | % |
| `cloud` | `CLOUD_AMT` | % |
| `wind2` | `WS2M` | m/s |
| `pressure` | `PS` | Pa (POWER gives kPa) |
| `rain` | `PRECTOTCORR` | mm/h |

POWER meteorology is available up to about a week before today. Radiation
and cloud cover arrive about three months later. The app reports any gaps
and refuses to run over them.

## Reference

García-Múnera, V., Velásquez Giron, N., Molina, R. D., Weber, L. J., &
Krajewski, W. F. (2026). RNSTM: A Network-Link Approach to Streamflow
Temperature Modeling. *JAWRA Journal of the American Water Resources
Association*, 62(4), e70140.
<https://doi.org/10.1111/1752-1688.70140>
