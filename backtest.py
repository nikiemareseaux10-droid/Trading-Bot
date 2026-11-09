"""Backtest : ordre exécuté à l'ouverture de la bougie suivante, frais, slippage, SL testé avant TP."""
import numpy as np
import pandas as pd

from risk import RiskManager
from strategy import add_indicators


def simulate(df, cfg, balance):
    s, r = cfg["strategy"], cfg["risk"]
    d = add_indicators(df, s)
    fee, slip = r["fee_pct"] / 100, r["slippage_pct"] / 100
    rm = RiskManager(r, balance)
    t = d.index
    o, h, l, c = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
    atr, ent = d["atr"].to_numpy(), d["entry"].to_numpy()

    cash, pos, pending, last_exit = balance, None, None, -10**9
    trades, curve = [], []
    for i in range(len(d)):
        if pending is not None and pos is None:  # 1) entrée à l'ouverture
            px = o[i] * (1 + slip)
            sl, tp = px - s["sl_atr"] * pending, px + s["tp_atr"] * pending
            qty = min(rm.size(cash, px, sl), cash / (px * (1 + fee)))
            if qty > 0:
                cost = qty * px * (1 + fee)
                cash -= cost
                pos = {"qty": qty, "entry": px, "sl": sl, "tp": tp, "cost": cost}
        pending = None

        if pos:  # 2) sorties (stop-loss prioritaire, gap pris en compte)
            ex = None
            if l[i] <= pos["sl"]:
                ex = min(pos["sl"], o[i])
            elif h[i] >= pos["tp"]:
                ex = max(pos["tp"], o[i])
            if ex is not None:
                proceeds = pos["qty"] * ex * (1 - slip) * (1 - fee)
                cash += proceeds
                trades.append({"exit_time": t[i], "pnl": proceeds - pos["cost"]})
                pos, last_exit = None, i

        equity = cash + (pos["qty"] * c[i] if pos else 0.0)  # 3) valorisation
        curve.append(equity)
        rm.update(t[i], equity)

        if (pos is None and ent[i] and rm.ok() and i < len(d) - 1
                and i - last_exit >= s["cooldown_bars"]):  # 4) signal -> ordre pour la bougie suivante
            pending = atr[i]
    return trades, pd.Series(curve, index=t)


def run_backtest(data, cfg):
    start = cfg["starting_balance"]
    trades, curves = [], []
    for df in data.values():
        tr, cv = simulate(df, cfg, start / len(data))
        trades += tr
        curves.append(cv)
    curve = pd.concat(curves, axis=1).ffill().bfill().sum(axis=1)

    pnl = np.array([x["pnl"] for x in trades])
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl <= 0].sum()
    step = curve.index.to_series().diff().median().total_seconds()
    rets = curve.pct_change().dropna()
    sharpe = rets.mean() / rets.std() * np.sqrt(365 * 86400 / step) if rets.std() > 0 else 0.0
    return {
        "days": max((curve.index[-1] - curve.index[0]).days, 1),
        "return_pct": round(float((curve.iloc[-1] / start - 1) * 100), 2),
        "max_drawdown_pct": round(float((curve / curve.cummax() - 1).min() * 100), 2),
        "trades": len(trades),
        "win_rate_pct": round(float((pnl > 0).mean() * 100), 1) if len(pnl) else 0.0,
        "profit_factor": round(float(wins / losses), 2) if losses > 0 else None,
        "sharpe": round(float(sharpe), 2),
        "final_balance": round(float(curve.iloc[-1]), 2),
    }
