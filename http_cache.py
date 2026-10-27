"""Tiny on-disk cache for GET requests to the public data services.

Every response body is stored under ``.cache/`` keyed by the URL and its
query parameters, so re-running the app (or a script) with the same gauge and
dates does not hit NASA POWER / USGS again. Delete the folder to refresh.
"""

import hashlib
import json
from pathlib import Path

import requests

CACHE_DIR = Path(__file__).with_name(".cache")
USER_AGENT = "StreamQual-toy (stream water-quality playground)"


def get_text(url, params=None, timeout=120, refresh=False):
    """GET ``url`` with ``params`` and return the response body as text.

    Successful responses are cached on disk; errors raise
    ``requests.HTTPError`` and are never cached.
    """
    params = dict(params or {})
    key = hashlib.sha1(
        (url + json.dumps(params, sort_keys=True, default=str)).encode()
    ).hexdigest()
    path = CACHE_DIR / f"{key}.txt"
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8")

    r = requests.get(url, params=params, timeout=timeout,
                     headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    CACHE_DIR.mkdir(exist_ok=True)
    path.write_text(r.text, encoding="utf-8")
    return r.text


def get_json(url, params=None, timeout=120, refresh=False):
    return json.loads(get_text(url, params, timeout, refresh))
