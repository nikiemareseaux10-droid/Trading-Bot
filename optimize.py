"""Teste plusieurs réglages de la stratégie, sans se tromper soi-même.

- Les 70 % de données les plus anciennes servent à CHOISIR les meilleurs réglages.
- Les 30 % les plus récentes, jamais utilisées pour choisir, servent à VÉRIFIER.
Usage : python optimize.py
"""
import copy
import itertools
import json
import os
import time

import numpy as np
import pandas as pd

import exchange as xc
from backtest import simulate

TRAIN_SHARE = 0.7
MIN_TRAIN_TRADES = 30
GRID = {
    "tf": ["1h", "4h"],
    "ema": [(50, 200), (20, 100)],
    "sltp": [(2, 3), (2, 4), (3, 6), (1.5, 4)],
    "min_signals": [2, 3],
}


def load_data(cfg):
    """Télécharge l'historique en 1h (gardé 24 h en cache dans le dossier)."""
    ex, raw = None, {}
    days = int(cfg["backtest_years"] * 365)
    for sym in cfg["symbols"]:
        path = f"cache_{cfg['exchange']}_{sym.replace('/', '_')}_1h.csv"
        if os.path.exists(path) and time.time() - os.path.getmtime(path) < 86400:
            df = pd.read_csv(path, index_col=0)
            df.index = pd.to_datetime(df.index, utc=True)
        else:
            ex = ex or xc.make_exchange(cfg)
            df = xc.fetch_history(ex, sym, "1h", days)
            df.to_csv(path)
        raw[sym] = df
    return raw


def resample(df, tf):
    if tf == "1h":
        return df
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return df.resample(tf).agg(agg).dropna()


def make_cfg(base, tf, ema, sltp, ms):
    c = copy.deepcopy(base)
    c["timeframe"] = tf
    c["strategy"].update(ema_fast=ema[0], ema_slow=ema[1], sl_atr=sltp[0], tp_atr=sltp[1], min_signals=ms)
    c["risk"].update(max_drawdown_pct=100, max_daily_loss_pct=100)  # évalue la stratégie seule
    return c


def run_combo(cfg, raw):
    start = cfg["starting_balance"]
    curves, trades = [], []
    for df in raw.values():
        tr, cv = simulate(resample(df, cfg["timeframe"]), cfg, start / len(raw))
        trades += tr
        curves.append(cv)
    return pd.concat(curves, axis=1).ffill().bfill().sum(axis=1), trades


def seg(curve, trades, a, b):
    c = curve[(curve.index >= a) & (curve.index <= b)]
    if len(c) < 2:
        return None
    pnl = np.array([t["pnl"] for t in trades if a <= t["exit_time"] <= b])
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl <= 0].sum()
    pf = wins / losses if losses > 0 else (99.0 if wins > 0 else 0.0)
    return {"ret": float((c.iloc[-1] / c.iloc[0] - 1) * 100),
            "dd": float((c / c.cummax() - 1).min() * 100),
            "n": int(len(pnl)), "pf": float(min(pf, 99.0))}


def fmt(s):
    return "n/a" if not s else f"{s['ret']:+.1f} % | PF {s['pf']:.2f} | {s['n']} trades"


def show(title, r):
    print(f"{title} {r['label']}")
    print(f"  Appris : {fmt(r['train'])}")
    print(f"  Test   : {fmt(r['test'])}\n")


def main():
    with open("config.json") as f:
        base = json.load(f)
    raw = load_data(base)
    t0 = min(d.index[0] for d in raw.values())
    t1 = max(d.index[-1] for d in raw.values())
    cut = t0 + (t1 - t0) * TRAIN_SHARE

    s = base["strategy"]
    current = (base["timeframe"], (s["ema_fast"], s["ema_slow"]), (s["sl_atr"], s["tp_atr"]), s["min_signals"])
    combos = list(itertools.product(GRID["tf"], GRID["ema"], GRID["sltp"], GRID["min_signals"]))
    print(f"{len(combos)} réglages + réglage actuel à tester (quelques minutes)...")

    rows = []
    for i, (tf, ema, sltp, ms) in enumerate([current] + combos):
        cfg = make_cfg(base, tf, ema, sltp, ms)
        curve, trades = run_combo(cfg, raw)
        rows.append({"label": f"{tf} EMA{ema[0]}/{ema[1]} SL{sltp[0]} TP{sltp[1]} sig{ms}",
                     "cfg": cfg, "train": seg(curve, trades, t0, cut), "test": seg(curve, trades, cut, t1)})
        print(f"  {i}/{len(combos)}", end="\r")
    print()

    bh = np.mean([(d.loc[d.index >= cut, "close"].iloc[-1] / d.loc[d.index >= cut, "close"].iloc[0] - 1) * 100
                  for d in raw.values()])
    print(f"Appris : {t0.date()} -> {cut.date()} | Test : {cut.date()} -> {t1.date()}")
    print(f"Référence (acheter et garder) sur le test : {bh:+.1f} %\n")

    show("RÉGLAGE ACTUEL", rows[0])
    cands = [r for r in rows[1:] if r["train"] and r["train"]["n"] >= MIN_TRAIN_TRADES]
    cands.sort(key=lambda r: r["train"]["pf"], reverse=True)
    for k, r in enumerate(cands[:3], 1):
        show(f"MEILLEUR #{k} (choisi sur l'appris)", r)

    with open("optimize_result.json", "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "cfg"} for r in rows], f, indent=2)

    best = cands[0] if cands else None
    t = best["test"] if best else None
    if t and t["ret"] > 0 and t["pf"] >= 1.2 and t["n"] >= 15:
        keep = copy.deepcopy(best["cfg"])
        keep["risk"] = base["risk"]  # remet les protections normales
        with open("config_best.json", "w") as f:
            json.dump(keep, f, indent=2)
        print("Candidat prometteur enregistré dans config_best.json.")
        print("Ce n'est PAS une preuve : plusieurs dizaines de réglages ont été essayés,")
        print("le hasard peut expliquer ce résultat. Validez-le en mode paper.")
    else:
        print("VERDICT : aucun réglage ne tient sur la période de test.")
        print("Forcer les paramètres jusqu'à trouver un gain serait se tromper soi-même.")
        print("Mieux vaut revoir l'idée de la stratégie. Rien n'a été enregistré.")


if __name__ == "__main__":
    main()
