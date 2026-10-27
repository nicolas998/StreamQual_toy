"""Simulated-vs-observed comparison: step-by-step errors and summary scores.

Relative error at each step::

    RE_i = (sim_i - obs_i) / obs_i * 100            [%]

It is left out (NaN) where ``|obs_i|`` is below the constituent's
``rel_error_floor`` (e.g. water near 0 degC), where it is meaningless.
"""

import numpy as np
import pandas as pd


def stepwise(sim, obs, rel_floor=0.0):
    """Per-step comparison on the common index of ``sim`` and ``obs``.

    Parameters
    ----------
    sim, obs : pandas.Series on the same (time) index; obs may have NaNs.
    rel_floor : float, |obs| below this gives no relative error.

    Returns
    -------
    DataFrame with ``sim``, ``obs``, ``error`` (sim - obs) and
    ``rel_error`` [%].
    """
    df = pd.DataFrame({"sim": sim, "obs": obs.reindex(sim.index)})
    df["error"] = df["sim"] - df["obs"]
    usable = df["obs"].abs() >= max(rel_floor, 1e-12)
    df["rel_error"] = (df["error"] / df["obs"] * 100.0).where(usable)
    return df


def summary(steps):
    """Summary scores of a :func:`stepwise` table (NaN rows are skipped).

    Returns a dict: ``n``, mean relative error ``MRE`` [%], mean absolute
    relative error ``MARE`` [%], ``bias``, ``MAE``, ``RMSE`` (in the units of
    the constituent), Nash-Sutcliffe ``NSE`` and Kling-Gupta ``KGE``.
    """
    d = steps.dropna(subset=["sim", "obs"])
    n = len(d)
    out = {"n": n}
    if n == 0:
        return out | dict.fromkeys(
            ["MRE", "MARE", "bias", "MAE", "RMSE", "NSE", "KGE"], np.nan)
    s, o, e = d["sim"].to_numpy(), d["obs"].to_numpy(), d["error"].to_numpy()
    re = d["rel_error"].dropna().to_numpy()
    out["MRE"] = float(re.mean()) if re.size else np.nan
    out["MARE"] = float(np.abs(re).mean()) if re.size else np.nan
    out["bias"] = float(e.mean())
    out["MAE"] = float(np.abs(e).mean())
    out["RMSE"] = float(np.sqrt((e ** 2).mean()))
    var_o = ((o - o.mean()) ** 2).sum()
    out["NSE"] = float(1.0 - (e ** 2).sum() / var_o) if var_o > 0 else np.nan
    if n > 1 and o.std() > 0 and s.std() > 0 and o.mean() != 0:
        r = np.corrcoef(s, o)[0, 1]
        alpha, beta = s.std() / o.std(), s.mean() / o.mean()
        out["KGE"] = float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2
                                         + (beta - 1) ** 2))
    else:
        out["KGE"] = np.nan
    return out
