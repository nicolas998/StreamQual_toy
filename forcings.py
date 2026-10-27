"""Meteorological forcings for the stream models.

* :func:`fetch_nasa_power` -- hourly point data from NASA POWER
  (https://power.larc.nasa.gov), returned in the units the models expect.
* :class:`HourlyForcing`   -- piecewise-constant forcing over the simulation,
  sampled by the models as ``forcing(t)`` with ``t`` in minutes (as in
  ASYNCH and the HLM playground).

Forcing variables (:data:`FIELDS`)
----------------------------------
    sw_down   incoming shortwave (solar) radiation          [W/m^2]
    lw_down   incoming longwave radiation                   [W/m^2]
    t_air     air temperature at 2 m                        [degC]
    rh        relative humidity at 2 m                      [%]
    cloud     cloud cover                                   [%]
    wind2     wind speed at 2 m                             [m/s]
    pressure  surface pressure                              [Pa]
    rain      precipitation                                 [mm/h]
    q         streamflow through the reach                  [m^3/s]

The units of ``rh``, ``cloud`` and ``pressure`` match the forcings of the
asynch implementation of RNSTM, which converts them internally (``* 0.01``).

NASA POWER's radiation and cloud products lag the present by a few months;
hours it has no data for come back as NaN (see :func:`missing_report`).
"""

from typing import NamedTuple

import numpy as np
import pandas as pd

from http_cache import get_json

POWER_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
# NASA POWER parameter -> (forcing field, factor to model units)
POWER_PARAMS = {
    "ALLSKY_SFC_SW_DWN": ("sw_down", 1.0),     # Wh/m^2 per hour = W/m^2
    "ALLSKY_SFC_LW_DWN": ("lw_down", 1.0),
    "T2M": ("t_air", 1.0),
    "RH2M": ("rh", 1.0),
    "CLOUD_AMT": ("cloud", 1.0),
    "WS2M": ("wind2", 1.0),
    "PS": ("pressure", 1000.0),                # kPa -> Pa
    "PRECTOTCORR": ("rain", 1.0),              # mm/h
}
MAX_GAP_H = 3          # gaps up to this many hours are interpolated


class ForcingSample(NamedTuple):
    sw_down: float
    lw_down: float
    t_air: float
    rh: float
    cloud: float
    wind2: float
    pressure: float
    rain: float
    q: float


FIELDS = ForcingSample._fields
METEO_FIELDS = FIELDS[:-1]

# For plots: field -> (label, unit)
FIELD_LABELS = {
    "sw_down": ("Shortwave ↓", "W/m²"),
    "lw_down": ("Longwave ↓", "W/m²"),
    "t_air": ("Air temperature", "°C"),
    "rh": ("Relative humidity", "%"),
    "cloud": ("Cloud cover", "%"),
    "wind2": ("Wind at 2 m", "m/s"),
    "pressure": ("Pressure", "Pa"),
    "rain": ("Precipitation", "mm/h"),
    "q": ("Streamflow", "m³/s"),
}


def fetch_nasa_power(lat, lon, start_utc, end_utc):
    """Hourly NASA POWER forcings for one point.

    Parameters
    ----------
    lat, lon : float, decimal degrees
    start_utc, end_utc : pandas.Timestamp (tz-aware, UTC), inclusive

    Returns
    -------
    DataFrame on an hourly UTC index with the columns of
    :data:`METEO_FIELDS` in model units. A row stamped hour ``h`` holds the
    mean over ``[h, h + 1 h)``. Gaps of up to :data:`MAX_GAP_H` hours are
    interpolated; longer ones stay NaN.
    """
    d = get_json(POWER_URL, {
        "parameters": ",".join(POWER_PARAMS), "community": "RE",
        "latitude": round(lat, 4), "longitude": round(lon, 4),
        "start": start_utc.strftime("%Y%m%d"),
        "end": end_utc.strftime("%Y%m%d"),
        "format": "JSON", "time-standard": "UTC",
    })
    fill = d.get("header", {}).get("fill_value", -999.0)
    hours = pd.date_range(start_utc, end_utc, freq="1h")
    out = pd.DataFrame(index=hours)
    for param, (field, factor) in POWER_PARAMS.items():
        raw = pd.Series(d["properties"]["parameter"][param], dtype=float)
        raw.index = pd.to_datetime(raw.index, format="%Y%m%d%H", utc=True)
        raw = raw.where(raw != fill) * factor
        out[field] = raw.reindex(hours)
    return out.interpolate(limit=MAX_GAP_H, limit_area="inside")[list(METEO_FIELDS)]


def missing_report(meteo, fields=METEO_FIELDS):
    """``{field: (n_missing_hours, first_missing_time)}`` for fields with NaNs.

    A field that is not a column of ``meteo`` counts as missing everywhere.
    """
    rep = {}
    for f in fields:
        if f not in meteo:
            rep[f] = (len(meteo), meteo.index[0])
            continue
        bad = meteo[f].isna()
        if bad.any():
            rep[f] = (int(bad.sum()), bad.idxmax())
    return rep


class HourlyForcing:
    """Piecewise-constant forcing driven by hourly records.

    Record ``i`` holds over ``[i dt, (i + 1) dt)`` minutes from the start of
    the run, as in ASYNCH. Times before the first record use record 0; times
    at/after the last record hold the final value.

    Parameters
    ----------
    meteo : DataFrame
        One row per record with (at least) the columns of
        :data:`METEO_FIELDS`, e.g. from :func:`fetch_nasa_power`.
    q : float or 1-D array_like
        Streamflow [m^3/s]: a scalar for a steady state, or one value per
        record.
    dt : float
        Record length in MINUTES (60.0 for hourly records).
    """

    def __init__(self, meteo, q, dt=60.0):
        n = len(meteo)
        q = np.full(n, float(q)) if np.ndim(q) == 0 else np.asarray(q, float)
        if q.size != n:
            raise ValueError("q array must match the number of records")
        cols = [meteo[f].to_numpy(float) for f in METEO_FIELDS] + [q]
        self.records = [ForcingSample(*map(float, row)) for row in zip(*cols)]
        self.index = meteo.index
        self.n = n
        self.dt = float(dt)

    def __call__(self, t):
        i = int(t // self.dt)                 # which record contains time t
        i = 0 if i < 0 else (self.n - 1 if i >= self.n else i)
        return self.records[i]

    @property
    def t_end(self):
        """End of the forcing series [min]."""
        return self.n * self.dt

    def frame(self):
        """The records as a DataFrame (one column per field)."""
        return pd.DataFrame(self.records, index=self.index)
