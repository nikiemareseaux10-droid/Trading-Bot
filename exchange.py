"""Connexion aux bourses crypto via CCXT."""
import os

import pandas as pd


def make_exchange(cfg, live=False):
    import ccxt  # importé ici pour que le reste du code se teste sans CCXT

    params = {"enableRateLimit": True}
    if live:
        params.update(apiKey=os.getenv("API_KEY"), secret=os.getenv("API_SECRET"))
    ex = getattr(ccxt, cfg["exchange"])(params)
    if os.getenv("TESTNET") == "1":  # compte de démonstration du courtier
        ex.set_sandbox_mode(True)
    return ex


def to_df(rows):
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.drop_duplicates("ts").set_index("ts").sort_index()


def fetch_recent(ex, symbol, timeframe, limit=500):
    return to_df(ex.fetch_ohlcv(symbol, timeframe, limit=limit))


def fetch_history(ex, symbol, timeframe, days):
    since = ex.milliseconds() - days * 86400 * 1000
    tf_ms = ex.parse_timeframe(timeframe) * 1000
    rows = []
    while since < ex.milliseconds():
        batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + tf_ms
    return to_df(rows)
