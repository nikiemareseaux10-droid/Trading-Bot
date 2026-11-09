"""Stratégie de suivi de tendance : EMA 50/200 + RSI + volume. Stops basés sur l'ATR."""
import pandas as pd


def add_indicators(df, p):
    d = df.copy()
    c = d["close"]
    d["ema_fast"] = c.ewm(span=p["ema_fast"], adjust=False).mean()
    d["ema_slow"] = c.ewm(span=p["ema_slow"], adjust=False).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / p["rsi_period"], adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / p["rsi_period"], adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + gain / loss.clip(lower=1e-12))

    prev = c.shift()
    tr = pd.concat(
        [d["high"] - d["low"], (d["high"] - prev).abs(), (d["low"] - prev).abs()], axis=1
    ).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / p["atr_period"], adjust=False).mean()
    d["vol_ma"] = d["volume"].rolling(p["vol_period"]).mean()

    # 3 signaux : tendance haussière, RSI dans la zone, volume au-dessus de sa moyenne
    trend = d["ema_fast"] > d["ema_slow"]
    rsi_ok = d["rsi"].between(p["rsi_min"], p["rsi_max"])
    vol_ok = d["volume"] > d["vol_ma"]
    d["signals"] = trend.astype(int) + rsi_ok.astype(int) + vol_ok.astype(int)
    # Entrée : tendance obligatoire + au moins `min_signals` signaux concordants
    d["entry"] = trend & (d["signals"] >= p["min_signals"])
    d.iloc[: p["ema_slow"], d.columns.get_loc("entry")] = False  # période de chauffe
    return d
