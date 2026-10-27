"""USGS gauge data: site metadata, instantaneous values and field measurements.

* :func:`site_info`            -- name, location, drainage area, time zone.
* :func:`find_gauges`          -- stream gauges of a state that report water
                                  temperature and/or specific conductance,
                                  with their periods of record.
* :func:`instantaneous_values` -- discharge, gage height, mean depth, water
                                  temperature and specific conductance,
                                  converted to SI and averaged to hourly
                                  values on a UTC index.
* :func:`channel_measurements` -- USGS field measurements (Q, width, area),
                                  giving the mean depth ``d = area / width``.
* :func:`fit_hydraulic_geometry` -- fit ``d = c_depth * Q**f_depth`` and
                                  ``w = c_width * Q**f_width`` to them.

Instantaneous values and site metadata come from the NWIS Water Services
(waterservices.usgs.gov); channel measurements from the Water Data OGC API
(api.waterdata.usgs.gov).
"""

import io
from dataclasses import dataclass

import numpy as np
import pandas as pd

from http_cache import get_json, get_text

IV_URL = "https://waterservices.usgs.gov/nwis/iv/"
SITE_URL = "https://waterservices.usgs.gov/nwis/site/"
OGC_URL = "https://api.waterdata.usgs.gov/ogcapi/v0/collections"

FT = 0.3048
# USGS parameter code -> (column name, factor to SI)
PCODES = {
    "00060": ("discharge", FT ** 3),    # ft^3/s -> m^3/s
    "00065": ("gage_height", FT),       # ft     -> m
    "00064": ("mean_depth", FT),        # ft     -> m
    "00010": ("water_temp", 1.0),       # degC
    "00095": ("spec_cond", 1.0),        # uS/cm at 25 degC
}
# Standard-time UTC offsets [h] of the USGS tz_cd codes.
TZ_OFFSETS = {"AST": -4, "EST": -5, "CST": -6, "MST": -7, "PST": -8,
              "AKST": -9, "HST": -10, "UTC": 0, "GMT": 0}
SQMI_TO_KM2 = 2.589988

STATES = ("AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA "
          "MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA PR RI SC SD TN "
          "TX UT VT VA WA WV WI WY").split()


@dataclass(frozen=True)
class Gauge:
    site_no: str
    name: str
    lat: float
    lon: float
    drainage_area_km2: float     # NaN if unknown
    tz: str                      # standard-time abbreviation, e.g. "CST"
    utc_offset_h: int            # standard-time offset, e.g. -6


def _read_rdb(text):
    """NWIS tab-separated RDB -> DataFrame (drops the column-format row)."""
    df = pd.read_csv(io.StringIO(text), sep="\t", comment="#", dtype=str)
    return df.iloc[1:].reset_index(drop=True)


def site_info(site_no):
    df = _read_rdb(get_text(SITE_URL, dict(format="rdb", sites=site_no,
                                           siteOutput="expanded")))
    if df.empty:
        raise ValueError(f"USGS site {site_no} not found")
    row = df.iloc[0]
    lat, lon = float(row["dec_lat_va"]), float(row["dec_long_va"])
    tz = str(row.get("tz_cd") or "")
    offset = TZ_OFFSETS.get(tz, round(lon / 15.0))
    area = pd.to_numeric(row.get("drain_area_va"), errors="coerce")
    return Gauge(site_no=row["site_no"], name=row["station_nm"], lat=lat,
                 lon=lon, drainage_area_km2=float(area) * SQMI_TO_KM2,
                 tz=tz or f"UTC{offset:+d}", utc_offset_h=int(offset))


def find_gauges(state_cd, pcodes=("00010", "00095")):
    """Active stream gauges in a state with instantaneous data for any of
    ``pcodes``.

    Returns one row per site with ``params`` (the set of every parameter code
    the site reports), ``periods`` (``{pcode: (begin, end)}`` dates of the
    requested codes it has), and ``has_q`` / ``has_gh`` for discharge and
    gage height. Filter with e.g. ``df[df.params.map({"00095"}.issubset)]``.
    """
    text = get_text(SITE_URL, dict(
        format="rdb", stateCd=state_cd.lower(), parameterCd=",".join(pcodes),
        hasDataTypeCd="iv", siteStatus="active", siteType="ST",
        seriesCatalogOutput="true", outputDataTypeCd="iv"))
    cat = _read_rdb(text)
    if cat.empty:
        return pd.DataFrame(columns=["site_no", "name", "lat", "lon", "params",
                                     "periods", "has_q", "has_gh"])
    rows = []
    for site_no, g in cat.groupby("site_no", sort=True):
        params = frozenset(g["parm_cd"].dropna())
        periods = {code: (g.loc[g["parm_cd"] == code, "begin_date"].min(),
                          g.loc[g["parm_cd"] == code, "end_date"].max())
                   for code in pcodes if code in params}
        if not periods:
            continue
        rows.append({
            "site_no": site_no,
            "name": g["station_nm"].iloc[0],
            "lat": float(g["dec_lat_va"].iloc[0]),
            "lon": float(g["dec_long_va"].iloc[0]),
            "params": params,
            "periods": periods,
            "has_q": "00060" in params,
            "has_gh": "00065" in params,
        })
    return pd.DataFrame(rows)


def _series_to_frame(ts):
    """One NWIS ``timeSeries`` entry -> Series [SI units] on a UTC index.

    A parameter can have several sensors (``values`` entries); the one with
    the most valid readings is used.
    """
    nodata = ts["variable"].get("noDataValue")
    best = None
    for block in ts["values"]:
        vals = block["value"]
        if not vals:
            continue
        s = pd.Series(pd.to_numeric([v["value"] for v in vals], errors="coerce"),
                      index=pd.to_datetime([v["dateTime"] for v in vals], utc=True))
        if nodata is not None:
            s = s.where(s != nodata)
        s = s.dropna()
        if best is None or s.size > best.size:
            best = s
    return best


def instantaneous_values(site_no, start_utc, end_utc, pcodes=tuple(PCODES)):
    """USGS instantaneous values, averaged to hourly values in SI units.

    Parameters
    ----------
    site_no : str
    start_utc, end_utc : pandas.Timestamp (tz-aware, UTC)
    pcodes : USGS parameter codes to request (keys of :data:`PCODES`).

    Returns
    -------
    DataFrame on an hourly UTC index from ``start_utc`` to ``end_utc`` with a
    column per available parameter (``discharge`` [m^3/s], ``gage_height``
    [m], ``mean_depth`` [m], ``water_temp`` [degC]). The value at hour ``h``
    is the mean of the readings within +/-30 min of ``h``. Parameters the
    gauge does not report are absent.
    """
    d = get_json(IV_URL, dict(
        format="json", sites=site_no, parameterCd=",".join(pcodes),
        startDT=(start_utc - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        endDT=(end_utc + pd.Timedelta(days=1)).strftime("%Y-%m-%d")))
    hours = pd.date_range(start_utc, end_utc, freq="1h")
    out = pd.DataFrame(index=hours)
    for ts in d["value"]["timeSeries"]:
        code = ts["variable"]["variableCode"][0]["value"]
        if code not in PCODES:
            continue
        s = _series_to_frame(ts)
        if s is None or s.empty:
            continue
        name, factor = PCODES[code]
        hourly = (s * factor).groupby(s.index.round("1h")).mean()
        out[name] = hourly.reindex(hours)
    return out.dropna(axis=1, how="all")


def channel_measurements(site_no):
    """USGS field measurements of the channel: Q, width, area and depth.

    A visit that measured several channels (e.g. a main and a side channel)
    becomes one row with the summed discharge, width and area.

    Returns a DataFrame with ``time`` (UTC), ``discharge`` [m^3/s],
    ``width`` [m], ``area`` [m^2], ``velocity`` [m/s] (Q / area) and
    ``depth`` [m] (mean depth, area / width), sorted by time. Empty if there
    are none.
    """
    feats, url = [], f"{OGC_URL}/channel-measurements/items"
    params = dict(monitoring_location_id=f"USGS-{site_no}", f="json",
                  limit=10000, skipGeometry="true")
    while url:
        d = get_json(url, params)
        feats += [f["properties"] for f in d.get("features", [])]
        url = next((l["href"] for l in d.get("links", [])
                    if l.get("rel") == "next"), None)
        params = None                    # the next link carries the query
    cols = ["time", "discharge", "width", "area", "velocity", "depth"]
    if not feats:
        return pd.DataFrame(columns=cols)
    raw = pd.DataFrame(feats)
    num = lambda c: pd.to_numeric(raw.get(c), errors="coerce")
    df = pd.DataFrame({
        "visit": raw.get("field_visit_id", raw.get("id")).fillna(raw["id"]),
        "time": pd.to_datetime(raw["time"], utc=True, errors="coerce"),
        "discharge": num("channel_flow") * FT ** 3,
        "width": num("channel_width") * FT,
        "area": num("channel_area") * FT ** 2,
    })
    ok = (df["discharge"] > 0) & (df["width"] > 0) & (df["area"] > 0)
    df = (df[ok].groupby("visit")
          .agg(time=("time", "min"), discharge=("discharge", "sum"),
               width=("width", "sum"), area=("area", "sum")))
    df["velocity"] = df["discharge"] / df["area"]
    df["depth"] = df["area"] / df["width"]
    return df.sort_values("time").reset_index(drop=True)[cols]


def _power_fit(x, y):
    """Fit y = c * x**f in log space; return (c, f, r2)."""
    lx, ly = np.log(x), np.log(y)
    f, logc = np.polyfit(lx, ly, 1)
    resid = ly - (logc + f * lx)
    r2 = 1.0 - resid.var() / ly.var() if ly.var() > 0 else np.nan
    return float(np.exp(logc)), float(f), float(r2)


def fit_hydraulic_geometry(meas, since=None, min_points=10):
    """Least-squares fits of ``depth = c_depth * Q**f_depth`` and
    ``width = c_width * Q**f_width`` in log space.

    Parameters
    ----------
    meas : DataFrame from :func:`channel_measurements`
    since : str or Timestamp, optional
        Use only measurements from this date on (channels change), unless
        that leaves fewer than ``min_points``.

    Returns
    -------
    dict with ``c_depth``, ``f_depth``, ``r2`` (of the depth fit),
    ``c_width``, ``f_width``, ``r2_width``, ``n`` and the ``used``
    measurements, or None if there are fewer than 3 valid measurements.
    """
    m = meas.dropna(subset=["discharge", "depth", "width"])
    if since is not None:
        recent = m[m["time"] >= pd.Timestamp(since, tz="UTC")]
        if len(recent) >= min_points:
            m = recent
    if len(m) < 3:
        return None
    q = m["discharge"].to_numpy()
    c_depth, f_depth, r2 = _power_fit(q, m["depth"].to_numpy())
    c_width, f_width, r2_width = _power_fit(q, m["width"].to_numpy())
    return {"c_depth": c_depth, "f_depth": f_depth, "r2": r2,
            "c_width": c_width, "f_width": f_width, "r2_width": r2_width,
            "n": int(len(m)), "used": m}
