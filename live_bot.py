"""Bot en vivo (demo) para la estrategia return-first v8 DIARIA (min 28 dias).

Reutiliza SIN CAMBIOS la lógica de backtester_batch.py (indicadores, señal,
stops, break-even), así lo que corre en vivo es lo mismo que se backtesteó.

Ejecución: cron cada hora ->  python live_bot.py
- Cada hora: revisa stop duro y break-even con el precio actual.
- Una vez al día (primera ejecución tras las 00:00 UTC): arma break-even con
  el cierre de ayer, salida por régimen, ejecuta la decisión pendiente y genera
  la nueva, igual que el backtest diario (decisión día t, ejecución día t+1).

Estado en state.json (sobrevive reinicios). Logs en logs/live_log.csv y
logs/bot.log. Cartera de papel interna = fuente de verdad. Si hay claves de
Binance Spot Testnet en el entorno, las órdenes de BTC se replican allí.

Requisitos: pip install ccxt yfinance pandas numpy requests
Variables opcionales: BINANCE_TESTNET_KEY, BINANCE_TESTNET_SECRET
"""

import csv
import json
import logging
import os
from datetime import datetime, timezone

import pandas as pd

import config
import backtester_batch as bt

BASE = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE, "state.json")
LOG_DIR = os.path.join(BASE, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(filename=os.path.join(LOG_DIR, "bot.log"), level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bot")

SOURCES = {"BTC/USDT": ("ccxt", "BTC/USDT"), "S&P 500": ("yahoo", "^GSPC")}
COST = config.TRANSACTION_COST_PCT


# ---------------------------------------------------------------- datos
def exchange(auth=False):
    import ccxt
    if auth:
        ex = ccxt.binance({"apiKey": os.environ["BINANCE_TESTNET_KEY"],
                           "secret": os.environ["BINANCE_TESTNET_SECRET"]})
        ex.set_sandbox_mode(True)
        return ex
    return ccxt.binance()


def fetch_daily(symbol):
    """Velas diarias CERRADAS (sin la vela en curso), ~500 días."""
    kind, ticker = SOURCES[symbol]
    if kind == "ccxt":
        rows = exchange().fetch_ohlcv(ticker, "1d", limit=600)
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["ts"], unit="ms").dt.normalize()
        df = df.iloc[:-1]  # vela de hoy aún abierta
    else:
        import yfinance as yf
        df = yf.Ticker(ticker).history(period="3y", interval="1d", auto_adjust=False).reset_index()
        df = df.rename(columns=str.lower)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        df = df[df["date"] < pd.Timestamp.now().normalize()]  # sólo sesiones cerradas
    return df[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["close"])


def market_data():
    md = {}
    for s in config.ALLOWED_ASSETS:
        df = bt.add_indicators(fetch_daily(s))
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        md[s] = df.set_index("date", drop=False)
    return md


def live_price(symbol, md):
    if SOURCES[symbol][0] == "ccxt":
        return float(exchange().fetch_ticker(SOURCES[symbol][1])["last"])
    try:
        import yfinance as yf
        return float(yf.Ticker(SOURCES[symbol][1]).fast_info["last_price"])
    except Exception:
        return float(md[symbol]["close"].iloc[-1])


# ---------------------------------------------------------------- estado
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"cash": config.INITIAL_CAPITAL, "positions": {}, "cooldown_until": {},
            "pending": None, "last_week": ""}


def save_state(st):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=2, default=str)
    os.replace(tmp, STATE_FILE)


def equity(st, prices):
    return st["cash"] + sum(p["quantity"] * prices.get(s, p["avg_price"]) for s, p in st["positions"].items())


def write_log(row):
    path = os.path.join(LOG_DIR, "live_log.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------- órdenes
def mirror_testnet(side, symbol, quantity):
    if symbol != "BTC/USDT" or not os.environ.get("BINANCE_TESTNET_KEY"):
        return ""
    try:
        o = exchange(auth=True).create_market_order("BTC/USDT", side, round(quantity, 5))
        return f"testnet id {o.get('id')}"
    except Exception as e:
        log.error("testnet %s %s: %s", side, symbol, e)
        return f"testnet error: {e}"


def sell(st, symbol, price, reason, prices):
    p = st["positions"].pop(symbol)
    st["cash"] += p["quantity"] * price * (1 - COST)
    ret = 100 * (price / p["avg_price"] - 1)
    ext = mirror_testnet("sell", symbol, p["quantity"])
    prices[symbol] = price
    write_log({"utc": now_iso(), "action": "sell", "symbol": symbol, "price": price,
               "reason": reason, "trade_return_pct": round(ret, 2),
               "equity": round(equity(st, prices), 2), "external": ext})
    log.info("VENTA %s @ %.2f (%s) %+.2f%%", symbol, price, reason, ret)


def buy(st, symbol, price, reason, prices):
    total = equity(st, prices)
    spend = min(total * bt.position_weight(symbol), st["cash"] / (1 + COST))
    if spend < 1:
        return
    qty = spend / price
    st["cash"] -= spend * (1 + COST)
    st["positions"][symbol] = {"quantity": qty, "avg_price": price, "highest_price": price,
                               "entry_date": today(), "breakeven_armed": False,
                               "breakeven_pending": False,
                               "weight": bt.position_weight(symbol),
                               "stop_level": price * (1 - bt.hard_stop_pct(symbol))}
    ext = mirror_testnet("buy", symbol, qty)
    write_log({"utc": now_iso(), "action": "buy", "symbol": symbol, "price": price,
               "reason": reason, "trade_return_pct": "", "equity": round(equity(st, prices), 2),
               "external": ext})
    log.info("COMPRA %s @ %.2f (%s)", symbol, price, reason)


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")


# ---------------------------------------------------------------- ciclo
def arm_breakeven(st, md):
    """Igual que el backtester: si un CIERRE diario supera el trigger, el
    break-even queda pendiente y se arma en la vela siguiente."""
    trig = getattr(config, "BREAKEVEN_TRIGGER_PCT", None)
    if not (getattr(config, "USE_BREAKEVEN_PROTECTION", False) and trig):
        return
    for s, p in st["positions"].items():
        if p.get("breakeven_pending"):
            p["breakeven_armed"], p["breakeven_pending"] = True, False
        last = md[s].iloc[-1]
        if last["date"] >= p["entry_date"] and float(last["close"]) >= p["avg_price"] * (1 + trig):
            p["breakeven_pending"] = True


def check_stops(st, prices):
    """Stop duro / break-even con el precio actual (cada hora)."""
    for s in list(st["positions"]):
        p, px = st["positions"][s], prices[s]
        p["highest_price"] = max(p["highest_price"], px)
        p["stop_level"] = bt.stop_level(p, s)
        if px <= p["stop_level"]:
            reason = "break-even" if p["breakeven_armed"] else "stop-loss"
            sell(st, s, px, reason, prices)
            weeks = bt.cooldown_weeks_after_stop(s) if reason == "stop-loss" else 0
            if weeks > 0:
                st["cooldown_until"][s] = bt.add_weeks(today(), weeks)


def today():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def daily(st, md, prices):
    date = today()
    # 1. salida por régimen
    for s in list(st["positions"]):
        exit_now, reason = bt.should_exit_position(bt.get_row(md, s, date))
        if exit_now:
            sell(st, s, prices[s], reason, prices)
    # 2. ejecutar decisión de la semana anterior
    pend = st.get("pending")
    if pend and pend["action"] == "buy" and pend["symbol"] not in st["positions"]:
        for s in list(st["positions"]):
            sell(st, s, prices[s], f"rotacion a {pend['symbol']}", prices)
        buy(st, pend["symbol"], prices[pend["symbol"]], pend["reason"], prices)
    # 3. nueva decisión determinista
    cur = next(iter(st["positions"]), "")
    d = bt.decide(md, date, cur, st["cooldown_until"], st["positions"].get(cur, {}))
    st["pending"] = {"action": d["action"], "symbol": d["symbol"], "reason": d["reason"]}
    st["last_week"] = date
    write_log({"utc": now_iso(), "action": "decision", "symbol": d["symbol"], "price": "",
               "reason": f"{d['action']}: {d['reason']}", "trade_return_pct": "",
               "equity": round(equity(st, prices), 2), "external": ""})


def main():
    st = load_state()
    md = market_data()
    prices = {s: live_price(s, md) for s in config.ALLOWED_ASSETS}
    day = today()
    if st.get("last_day") != day:          # una vez por día UTC, con la vela de ayer cerrada
        arm_breakeven(st, md)
    check_stops(st, prices)
    if st.get("last_day") != day:
        daily(st, md, prices)
        st["last_day"] = day
    save_state(st)
    log.info("ok equity=%.2f cash=%.2f pos=%s", equity(st, prices), st["cash"], list(st["positions"]))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("fallo en el ciclo")
        raise
