"""Gestion du risque : taille de position, pertes maximales, contrôle de l'écart ±11 %."""


class RiskManager:
    def __init__(self, cfg, balance):
        self.cfg = cfg
        self.peak = balance
        self.day = None
        self.day_start = balance
        self.halted = None  # None, "daily", "drawdown" ou "deviation"

    def update(self, ts, equity):
        if ts.date() != self.day:  # nouveau jour : la pause journalière est levée
            self.day, self.day_start = ts.date(), equity
            if self.halted == "daily":
                self.halted = None
        self.peak = max(self.peak, equity)
        daily_loss = (self.day_start - equity) / self.day_start * 100
        if self.halted is None and daily_loss >= self.cfg["max_daily_loss_pct"]:
            self.halted = "daily"
        if (self.peak - equity) / self.peak * 100 >= self.cfg["max_drawdown_pct"]:
            self.halted = "drawdown"

    def ok(self):
        return self.halted is None

    def size(self, equity, entry, sl):
        """Quantité telle que la perte au stop-loss = risk_per_trade_pct du capital."""
        per_unit = entry - sl
        if per_unit <= 0:
            return 0.0
        qty = equity * self.cfg["risk_per_trade_pct"] / 100 / per_unit
        return min(qty, equity * self.cfg["max_alloc_pct"] / 100 / entry)


def check_deviation(actual_pct, bt_return_pct, bt_days, live_days, trades, mon):
    """Compare le résultat réel au résultat attendu du backtest (mis à l'échelle de la durée).
    Retourne (ok, écart en points de %). Pas de verdict avant min_days et min_trades."""
    if live_days < mon["min_days"] or trades < mon["min_trades"]:
        return True, 0.0
    gap = actual_pct - bt_return_pct * live_days / bt_days
    return abs(gap) <= mon["tolerance_pct"], gap
