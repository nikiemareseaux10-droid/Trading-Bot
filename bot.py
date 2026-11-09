"""Bot de trading (bêta) : python bot.py backtest | paper | live"""
import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

import exchange as xc
from backtest import run_backtest
from risk import RiskManager, check_deviation
from strategy import add_indicators

log = logging.getLogger("bot")


def notify(msg):
    log.info(msg)
    token, chat = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if token and chat:
        try:
            import requests
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat, "text": msg}, timeout=10)
        except Exception as e:
            log.warning("Telegram : %s", e)


def journal(symbol, side, qty, price, pnl, reason):
    new = not os.path.exists("journal.csv")
    with open("journal.csv", "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "symbol", "side", "qty", "price", "pnl", "reason"])
        w.writerow([datetime.now(timezone.utc).isoformat(), symbol, side, qty, price, pnl, reason])


def load_state(path, balance):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"cash": balance, "start_balance": balance, "start": datetime.now(timezone.utc).isoformat(),
            "positions": {}, "last_bar": {}, "last_exit": {}, "trades": 0}


def save_state(path, st):
    with open(path, "w") as f:
        json.dump(st, f, indent=2)


def quote_of(cfg):
    return cfg["symbols"][0].split("/")[1]


def refresh_cash(ex, cfg, st):
    st["cash"] = float(ex.fetch_balance()[quote_of(cfg)]["free"])


def equity_of(st, prices):
    return st["cash"] + sum(p["qty"] * prices.get(s, p["entry"]) for s, p in st["positions"].items())


def open_position(ex, cfg, st, rm, sym, price, atr, equity, live):
    s, r = cfg["strategy"], cfg["risk"]
    fee, slip = r["fee_pct"] / 100, r["slippage_pct"] / 100
    px = price * (1 + slip)
    sl, tp = px - s["sl_atr"] * atr, px + s["tp_atr"] * atr
    qty = min(rm.size(equity, px, sl), st["cash"] / (px * (1 + fee)))
    if live:
        qty = float(ex.amount_to_precision(sym, qty))
    if qty <= 0:
        return
    if live:
        order = ex.create_market_buy_order(sym, qty)
        px = float(order.get("average") or px)
        qty = float(order.get("filled") or qty)
        refresh_cash(ex, cfg, st)
    else:
        st["cash"] -= qty * px * (1 + fee)
    st["positions"][sym] = {"qty": qty, "entry": px, "sl": sl, "tp": tp}
    journal(sym, "BUY", qty, px, "", "signal")
    notify(f"ACHAT {sym} qté={qty:.6g} prix={px:.2f} SL={sl:.2f} TP={tp:.2f}")


def close_position(ex, cfg, st, sym, price, reason, live):
    r = cfg["risk"]
    fee, slip = r["fee_pct"] / 100, r["slippage_pct"] / 100
    p = st["positions"][sym]
    qty = p["qty"]
    if live:
        free = float(ex.fetch_balance()[sym.split("/")[0]]["free"])  # les frais réduisent le solde
        qty = float(ex.amount_to_precision(sym, min(qty, free)))
        order = ex.create_market_sell_order(sym, qty)
        px = float(order.get("average") or price)
        pnl = (px - p["entry"]) * qty  # approximation : frais non inclus
        refresh_cash(ex, cfg, st)
    else:
        px = price * (1 - slip)
        proceeds = qty * px * (1 - fee)
        st["cash"] += proceeds
        pnl = proceeds - qty * p["entry"] * (1 + fee)
    del st["positions"][sym]  # supprimé seulement si l'ordre a réussi
    st["trades"] += 1
    st["last_exit"][sym] = time.time()
    journal(sym, "SELL", qty, px, round(pnl, 4), reason)
    notify(f"VENTE {sym} ({reason}) prix={px:.2f} résultat={pnl:+.2f}")


def step(ex, cfg, st, rm, live):
    """Un cycle : sorties SL/TP, contrôles de risque, entrées, contrôle ±11 %. False = arrêter le bot."""
    s, r, m = cfg["strategy"], cfg["risk"], cfg["monitor"]
    tf_s = ex.parse_timeframe(cfg["timeframe"])
    now = datetime.now(timezone.utc)
    if live:
        refresh_cash(ex, cfg, st)
    prices, data = {}, {}
    for sym in cfg["symbols"]:
        prices[sym] = float(ex.fetch_ticker(sym)["last"])
        bars = xc.fetch_recent(ex, sym, cfg["timeframe"], 500).iloc[:-1]  # sans la bougie en cours
        data[sym] = add_indicators(bars, s)

    for sym, p in list(st["positions"].items()):  # stops gérés par le bot (pas par la bourse)
        if prices[sym] <= p["sl"]:
            close_position(ex, cfg, st, sym, prices[sym], "SL", live)
        elif prices[sym] >= p["tp"]:
            close_position(ex, cfg, st, sym, prices[sym], "TP", live)

    before = rm.halted
    rm.update(now, equity_of(st, prices))
    if rm.halted and rm.halted != before:
        notify(f"PAUSE ({rm.halted}) : perte maximale atteinte.")
    stop = ("ARRÊT D'URGENCE (fichier STOP)" if os.path.exists("STOP")
            else "ARRÊT : drawdown maximal atteint" if rm.halted == "drawdown" else None)
    if stop:
        for sym in list(st["positions"]):
            close_position(ex, cfg, st, sym, prices[sym], "ARRET", live)
        notify(stop)
        return False

    equity = equity_of(st, prices)
    for sym in cfg["symbols"]:  # entrées, une seule fois par bougie fermée
        bar = str(data[sym].index[-1])
        if st["last_bar"].get(sym) == bar:
            continue
        st["last_bar"][sym] = bar
        last = data[sym].iloc[-1]
        cooled = time.time() - st["last_exit"].get(sym, 0) >= s["cooldown_bars"] * tf_s
        if (rm.ok() and bool(last["entry"]) and sym not in st["positions"] and cooled
                and len(st["positions"]) < r["max_open_positions"]):
            open_position(ex, cfg, st, rm, sym, prices[sym], float(last["atr"]), equity, live)

    if rm.ok() and os.path.exists("backtest_result.json"):  # contrôle de l'écart ±11 %
        with open("backtest_result.json") as f:
            bt = json.load(f)
        days = (now - datetime.fromisoformat(st["start"])).total_seconds() / 86400
        actual = (equity_of(st, prices) / st["start_balance"] - 1) * 100
        ok, gap = check_deviation(actual, bt["return_pct"], bt["days"], days, st["trades"], m)
        if not ok:
            rm.halted = "deviation"
            notify(f"PAUSE : écart de {gap:+.1f} points, hors de ±{m['tolerance_pct']} %. "
                   "Décision humaine requise (voir README).")
    return True


def run_loop(cfg, live):
    ex = xc.make_exchange(cfg, live)
    ex.load_markets()
    mode = "live" if live else "paper"
    path = f"state_{mode}.json"
    balance = cfg["starting_balance"]
    if live and not os.path.exists(path):
        balance = float(ex.fetch_balance()[quote_of(cfg)]["total"])
    st = load_state(path, balance)
    rm = RiskManager(cfg["risk"], st["start_balance"])
    notify(f"Bot démarré en mode {mode.upper()} sur {', '.join(cfg['symbols'])}")
    try:
        while True:
            try:
                if not step(ex, cfg, st, rm, live):
                    break
            except Exception:
                log.exception("Erreur, nouvel essai au prochain cycle")
            save_state(path, st)
            time.sleep(cfg["poll_seconds"])
    except KeyboardInterrupt:
        notify("Arrêt manuel : les positions ouvertes ne sont PLUS surveillées (stops gérés par le bot).")
    save_state(path, st)


def cmd_backtest(cfg):
    ex = xc.make_exchange(cfg)
    days = int(cfg["backtest_years"] * 365)
    data = {s: xc.fetch_history(ex, s, cfg["timeframe"], days) for s in cfg["symbols"]}
    res = run_backtest(data, cfg)
    with open("backtest_result.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print("Un backtest ne garantit rien. Étape suivante : python bot.py paper")


def main():
    ap = argparse.ArgumentParser(description="Bot de trading (bêta)")
    ap.add_argument("command", choices=["backtest", "paper", "live"])
    ap.add_argument("--config", default="config.json")
    a = ap.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()])
    with open(a.config) as f:
        cfg = json.load(f)
    if a.command == "backtest":
        cmd_backtest(cfg)
        return
    live = a.command == "live"
    if live:
        if os.getenv("CONFIRM_LIVE") != "OUI":
            sys.exit("Mode réel bloqué : mettez CONFIRM_LIVE=OUI dans .env après avoir lu le README.")
        if not os.path.exists("backtest_result.json"):
            sys.exit("Lancez d'abord : python bot.py backtest")
        if not (os.getenv("API_KEY") and os.getenv("API_SECRET")):
            sys.exit("Clés API manquantes dans .env (sans droit de retrait).")
    run_loop(cfg, live)


if __name__ == "__main__":
    main()
