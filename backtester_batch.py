"""Backtester return-first v8 — decisiones diarias y mínimo de 28 días.


Características
---------------
- Señales evaluadas diariamente; una señal se ejecuta en la siguiente fecha.
- El mínimo de permanencia usa días naturales, no número de ciclos de decisión.
- Una sola posición activa: BTC/USDT o S&P 500; resto en cash.
- Selección, tamaño, rotación y riesgo son deterministas.
- Stops se evalúan con OHLC diario, con protección break-even diferida una vela.
- El LLM, si se activa, solo redacta una explicación y no puede decidir.


Requiere
--------
config.py y tools.py con load_csv(symbol) que devuelva:
    date, open, high, low, close[, volume]
"""


from __future__ import annotations


import os
import re
import sys
import time
from typing import Any


import numpy as np
import pandas as pd
import requests


import config
from tools import load_csv



ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
API_URL = "https://api.anthropic.com/v1/messages"
API_HEADERS = {
    "x-api-key": ANTHROPIC_API_KEY,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
}



class BacktestAPIError(RuntimeError):
    pass



def cfg(name: str, default: Any = None) -> Any:
    return getattr(config, name, default)



def fmt(value: Any, digits: int = 2) -> str:
    return "n/a" if value is None or pd.isna(value) else str(round(float(value), digits))



# ---------------------------------------------------------------------------
# Narrativa opcional: no participa en decisiones, sizing o ejecución
# ---------------------------------------------------------------------------



def call_claude(prompt: str, max_tokens: int) -> str:
    if not ANTHROPIC_API_KEY:
        raise BacktestAPIError("ANTHROPIC_API_KEY no configurada")
    payload = {
        "model": config.CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    response = requests.post(API_URL, headers=API_HEADERS, json=payload, timeout=30)
    response.raise_for_status()
    content = response.json().get("content", [])
    if not content or "text" not in content[0]:
        raise BacktestAPIError("Respuesta API sin texto")
    return content[0]["text"].strip()



def narrate(action: str, symbol: str, date: str, row: dict[str, Any] | None, fallback: str) -> str:
    if not cfg("USE_LLM_NARRATIVE", False) or row is None:
        return fallback
    prompt = (
        f"FECHA: {date}\nOPERACION YA DECIDIDA POR REGLAS: {action} {symbol}\n"
        f"DATOS: tendencia={row.get('trend')}, score={fmt(row.get('score'))}, "
        f"ADX={fmt(row.get('adx_14'))}, DI+={fmt(row.get('plus_di_14'))}, "
        f"DI-={fmt(row.get('minus_di_14'))}, ER20={fmt(row.get('er_20'))}, "
        f"RSI={fmt(row.get('rsi_14'), 1)}, ret20={fmt(row.get('ret_20d'))}%, "
        f"ret60={fmt(row.get('ret_60d'))}%.\n"
        "Describe el motivo en maximo diez palabras, sin inventar datos ni "
        "sugerir otra operacion. Devuelve solo texto plano."
    )
    for attempt in range(2):
        try:
            text = call_claude(prompt, config.MAX_TOKENS)
            return re.sub(r"\s+", " ", text).strip(' "')[:120] or fallback
        except (requests.RequestException, BacktestAPIError):
            if attempt == 0:
                time.sleep(1)
    return fallback



# ---------------------------------------------------------------------------
# Indicadores y datos
# ---------------------------------------------------------------------------



def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()



def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")


    df["sma_10"] = close.rolling(10).mean()
    df["sma_50"] = close.rolling(50).mean()
    df["sma_200"] = close.rolling(200).mean()


    delta = close.diff()
    avg_gain = wilder_smooth(delta.clip(lower=0.0), config.RSI_PERIOD)
    avg_loss = wilder_smooth((-delta).clip(lower=0.0), config.RSI_PERIOD)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    df["rsi_14"] = 100 - 100 / (1 + rs)
    df.loc[(avg_loss == 0) & (avg_gain > 0), "rsi_14"] = 100.0


    previous_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = wilder_smooth(true_range, config.ADX_PERIOD)
    df["atr_pct"] = 100 * df["atr_14"] / close.replace(0.0, np.nan)


    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    plus_di = 100 * wilder_smooth(plus_dm, config.ADX_PERIOD) / df["atr_14"].replace(0.0, np.nan)
    minus_di = 100 * wilder_smooth(minus_dm, config.ADX_PERIOD) / df["atr_14"].replace(0.0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    df["plus_di_14"] = plus_di
    df["minus_di_14"] = minus_di
    df["adx_14"] = wilder_smooth(dx, config.ADX_PERIOD)
    df["adx_slope_5"] = df["adx_14"].diff(5)


    directional_change = (close - close.shift(config.ER_PERIOD)).abs()
    path_length = close.diff().abs().rolling(config.ER_PERIOD).sum()
    df["er_20"] = directional_change / path_length.replace(0.0, np.nan)


    for period in (5, 20, 60):
        df[f"ret_{period}d"] = close.pct_change(period) * 100
    df["momentum_score"] = (
        config.MOMENTUM_5D_WEIGHT * df["ret_5d"]
        + config.MOMENTUM_20D_WEIGHT * df["ret_20d"]
        + config.MOMENTUM_60D_WEIGHT * df["ret_60d"]
    )
    df["momentum_adj_atr"] = df["momentum_score"] / df["atr_pct"].clip(lower=0.10)


    bb_mid = close.rolling(config.BB_PERIOD).mean()
    bb_std = close.rolling(config.BB_PERIOD).std()
    df["bb_upper"] = bb_mid + config.BB_STDDEV * bb_std
    df["bb_lower"] = bb_mid - config.BB_STDDEV * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_mid.replace(0.0, np.nan)
    df["bb_pct_b"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0.0, np.nan)


    if "volume" in df.columns:
        volume = pd.to_numeric(df["volume"], errors="coerce")
        df["vol_ma_20"] = volume.rolling(20).mean()
        df["rel_volume"] = volume / df["vol_ma_20"].replace(0.0, np.nan)
    else:
        df["rel_volume"] = np.nan


    df["trend_entry_level"] = df["sma_200"] + cfg("ENTRY_TREND_BUFFER_ATR", 0.0) * df["atr_14"]
    df["trend_exit_level"] = df["sma_200"] - cfg("EXIT_TREND_BUFFER_ATR", 0.0) * df["atr_14"]
    df["trend"] = np.select(
        [df["sma_200"].isna(), close > df["trend_entry_level"]],
        ["indefinida", "alcista"],
        default="bajista",
    )
    return df



def load_all_data(symbols: list[str], warmup_start: str, end: str, verbose: bool = True) -> dict[str, pd.DataFrame]:
    market_data: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            df = load_csv(symbol).copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df[(df["date"] >= pd.Timestamp(warmup_start)) & (df["date"] <= pd.Timestamp(end))].dropna(subset=["date"])
            if df.empty:
                if verbose:
                    print(f"  Aviso: sin datos para {symbol}; se omite")
                continue
            df = add_indicators(df)
            df["date"] = df["date"].dt.strftime("%Y-%m-%d")
            market_data[symbol] = df.set_index("date", drop=False)
            if verbose:
                print(f"  OK {symbol}: {len(df)} sesiones cargadas")
        except Exception as exc:
            if verbose:
                print(f"  Error cargando {symbol}: {exc}")
    return market_data



# ---------------------------------------------------------------------------
# Mercado, elegibilidad, señal y sizing
# ---------------------------------------------------------------------------



def get_row(market_data: dict[str, pd.DataFrame], symbol: str, date: str) -> pd.Series | None:
    df = market_data.get(symbol)
    if df is None:
        return None
    rows = df.loc[df.index <= date]
    return None if rows.empty else rows.iloc[-1]



def get_daily_window(market_data: dict[str, pd.DataFrame], symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = market_data.get(symbol)
    if df is None:
        return pd.DataFrame()
    return df.loc[(df.index > start_date) & (df.index <= end_date)].copy()



def get_price(market_data: dict[str, pd.DataFrame], symbol: str, date: str) -> float | None:
    row = get_row(market_data, symbol, date)
    return None if row is None else float(row["close"])



def is_cooldown(cooldown_until: dict[str, str], symbol: str, date: str) -> bool:
    return cooldown_until.get(symbol, "") >= date



def cooldown_weeks_after_stop(symbol: str) -> int:
    return int(cfg("COOLDOWN_WEEKS_AFTER_STOP_BY_ASSET", {}).get(symbol, cfg("COOLDOWN_WEEKS_AFTER_SL", 0)))



def is_volume_ok(row: pd.Series) -> bool:
    if not cfg("USE_VOLUME_FILTER", False):
        return True
    value = row.get("rel_volume", np.nan)
    return pd.notna(value) and value >= config.REL_VOLUME_MIN



def hard_stop_pct(symbol: str) -> float:
    return float(cfg("STOP_LOSS_PCT_BY_ASSET", {}).get(symbol, config.STOP_LOSS_PCT))



def position_weight(symbol: str) -> float:
    global_max = float(config.MAX_POSITION_SIZE_PCT)
    min_weight = float(cfg("MIN_POSITION_WEIGHT_BY_ASSET", {}).get(symbol, global_max))
    max_weight = float(cfg("MAX_POSITION_WEIGHT_BY_ASSET", {}).get(symbol, global_max))
    if not 0 <= min_weight <= max_weight <= 1:
        raise ValueError(f"Pesos invalidos para {symbol}: min={min_weight}, max={max_weight}")
    return min(global_max, max_weight)



def eligibility(row: pd.Series | None, cooldown: bool) -> tuple[bool, str]:
    if row is None:
        return False, "sin datos"
    required = ["sma_10", "sma_200", "adx_14", "plus_di_14", "minus_di_14", "er_20", "ret_20d", "ret_60d", "bb_pct_b", "rsi_14"]
    if any(pd.isna(row.get(key, np.nan)) for key in required):
        return False, "indicadores incompletos"
    if cooldown:
        return False, "cooldown"
    if row["trend"] != "alcista":
        return False, "tendencia no alcista"
    if row["close"] <= row["sma_10"]:
        return False, "precio bajo SMA10"
    if row["adx_14"] < config.ADX_MIN:
        return False, "ADX bajo"
    if row["plus_di_14"] <= row["minus_di_14"]:
        return False, "DI negativo"
    if row["er_20"] < config.ER_MIN:
        return False, "ER bajo"
    if row["ret_20d"] <= 0 or row["ret_60d"] <= 0:
        return False, "momentum medio negativo"
    if not is_volume_ok(row):
        return False, "volumen insuficiente"


    normal = row["rsi_14"] < config.RSI_MAX_NORMAL_ENTRY and row["bb_pct_b"] <= config.BB_PCT_B_MAX_ENTRY
    extended = (
        row["rsi_14"] >= config.RSI_MAX_NORMAL_ENTRY
        and row["adx_14"] >= config.ADX_MIN_EXTENDED_ENTRY
        and row["adx_slope_5"] > 0
        and row["er_20"] >= config.ER_MIN_EXTENDED_ENTRY
        and row["bb_pct_b"] <= config.BB_PCT_B_MAX_EXTENDED_ENTRY
    )
    if normal:
        return True, "entrada normal"
    if extended:
        return True, "continuacion tendencia"
    return False, "entrada extendida no confirmada"



def make_snapshot(market_data: dict[str, pd.DataFrame], date: str, current_position: str | None, cooldown_until: dict[str, str]):
    rows: list[dict[str, Any]] = []
    fields = [
        "close", "sma_10", "sma_50", "sma_200", "trend_entry_level", "trend_exit_level", "trend",
        "rsi_14", "atr_14", "atr_pct", "adx_14", "adx_slope_5", "plus_di_14", "minus_di_14",
        "er_20", "ret_5d", "ret_20d", "ret_60d", "bb_pct_b", "bb_width",
    ]
    for symbol in market_data:
        row = get_row(market_data, symbol, date)
        if row is None:
            continue
        cooldown = is_cooldown(cooldown_until, symbol, date)
        eligible_now, reason = eligibility(row, cooldown)
        item = {
            "symbol": symbol,
            "eligible": eligible_now,
            "eligibility_reason": reason,
            "cooldown": cooldown,
            "cooldown_until": cooldown_until.get(symbol, ""),
            "score": row.get("momentum_adj_atr"),
        }
        item.update({field: row.get(field, np.nan) for field in fields})
        rows.append(item)
    eligible_rows = [item for item in rows if item["eligible"] and pd.notna(item["score"])]
    best = max(eligible_rows, key=lambda item: item["score"], default=None)
    current = next((item for item in rows if item["symbol"] == current_position), None)
    return rows, current, best



def should_exit_position(row: pd.Series | None) -> tuple[bool, str]:
    if row is not None and row["close"] < row["trend_exit_level"]:
        return True, "salida por regimen"
    return False, ""



def held_days(position: dict[str, Any] | None, date: str) -> int:
    if not position or not position.get("entry_date"):
        return 0
    return max(0, (pd.Timestamp(date) - pd.Timestamp(position["entry_date"])).days)



def decide(market_data: dict[str, pd.DataFrame], date: str, current_position: str | None, cooldown_until: dict[str, str], position: dict[str, Any] | None) -> dict[str, Any]:
    rows, current, best = make_snapshot(market_data, date, current_position, cooldown_until)
    base = {
        "date": date,
        "snapshot": rows,
        "decision_source": "rule",
        "rotation_margin_used": np.nan,
        "current_score": np.nan,
        "best_score": np.nan,
        "score_gap": np.nan,
        "held_days": held_days(position, date),
    }


    if current_position:
        if best is None:
            return {**base, "action": "hold", "symbol": current_position, "reason": "sin candidato mejor, se mantiene tendencia"}
        if best["symbol"] == current_position:
            return {**base, "action": "hold", "symbol": current_position, "reason": "sigue siendo el mejor candidato"}
        minimum_days = int(cfg("ROTATION_MIN_HOLD_DAYS", 0))
        if base["held_days"] < minimum_days:
            return {**base, "action": "hold", "symbol": current_position, "reason": "periodo minimo de mantenimiento"}


        margin = float(config.ROTATION_SCORE_MARGIN)
        if current_position == "BTC/USDT":
            margin = max(margin, float(cfg("BTC_ROTATION_SCORE_MARGIN", margin)))
        current_score = current.get("score") if current else np.nan
        score_gap = best["score"] - current_score if pd.notna(current_score) else np.nan
        details = {"rotation_margin_used": margin, "current_score": current_score, "best_score": best["score"], "score_gap": score_gap}
        if pd.isna(current_score) or score_gap > margin:
            return {**base, **details, "action": "buy", "symbol": best["symbol"], "reason": f"rotacion a {best['symbol']}, gap {fmt(score_gap)} supera margen {fmt(margin)}"}
        return {**base, **details, "action": "hold", "symbol": current_position, "reason": "ventaja de rotacion insuficiente"}


    if best is None:
        return {**base, "action": "hold", "symbol": "", "reason": "sin activos elegibles"}
    return {**base, "action": "buy", "symbol": best["symbol"], "best_score": best["score"], "reason": f"mejor candidato elegible, {best['eligibility_reason']}, score {fmt(best['score'])}"}



# ---------------------------------------------------------------------------
# Cartera, stops, ejecución y log
# ---------------------------------------------------------------------------



def portfolio_value(portfolio: dict[str, Any], date: str, market_data: dict[str, pd.DataFrame]) -> float:
    total = float(portfolio["cash"])
    for symbol, position in portfolio["positions"].items():
        price = get_price(market_data, symbol, date) or position["avg_price"]
        position["last_price"] = price
        total += position["quantity"] * price
    return total



def add_weeks(date_str: str, weeks: int) -> str:
    return (pd.Timestamp(date_str) + pd.Timedelta(weeks=weeks)).strftime("%Y-%m-%d")



def snapshot_for_symbol(snapshot: list[dict[str, Any]], symbol: str) -> dict[str, Any]:
    row = next((item for item in snapshot if item["symbol"] == symbol), {})
    keys = [
        "adx_14", "adx_slope_5", "plus_di_14", "minus_di_14", "er_20", "ret_5d", "ret_20d", "ret_60d",
        "score", "atr_14", "atr_pct", "rsi_14", "sma_10", "sma_50", "sma_200", "trend_entry_level",
        "trend_exit_level", "bb_pct_b", "bb_width", "trend", "eligible", "eligibility_reason", "cooldown", "cooldown_until",
    ]
    event = {key: row.get(key, np.nan) for key in keys}
    event["entry_eligible"] = event.pop("eligible", False)
    event["momentum_adj_atr"] = event.pop("score", np.nan)
    return event



def log_event(trade_log: list[dict[str, Any]], date: str, symbol: str, action: str, reason: str, status: str, portfolio: dict[str, Any], market_data: dict[str, pd.DataFrame], snapshot: list[dict[str, Any]] | None = None, decision_source: str = "rule", position: dict[str, Any] | None = None, execution_price: float = np.nan, trade_return: float = np.nan, decision: dict[str, Any] | None = None, cooldown_weeks: int | float = np.nan) -> None:
    decision = decision or {}
    event = {
        "date": date,
        "symbol": symbol,
        "action": action,
        "reason": reason,
        "status": status,
        "decision_source": decision_source,
        "execution_price": execution_price,
        "trade_return_pct": trade_return,
        "portfolio_value": round(portfolio_value(portfolio, date, market_data), 2),
        "rotation_margin_used": decision.get("rotation_margin_used", np.nan),
        "current_score": decision.get("current_score", np.nan),
        "best_score": decision.get("best_score", np.nan),
        "score_gap": decision.get("score_gap", np.nan),
        "decision_held_days": decision.get("held_days", np.nan),
        "cooldown_weeks_applied": cooldown_weeks,
    }
    event.update(snapshot_for_symbol(snapshot or [], symbol))
    if position:
        event.update({
            "entry_date": position.get("entry_date", ""),
            "entry_price": position.get("avg_price", np.nan),
            "highest_price": position.get("highest_price", np.nan),
            "stop_level": position.get("stop_level", np.nan),
            "breakeven_armed": position.get("breakeven_armed", False),
            "position_weight": position.get("weight", np.nan),
            "held_days": held_days(position, date),
        })
    else:
        event.update({
            "entry_date": "",
            "entry_price": np.nan,
            "highest_price": np.nan,
            "stop_level": np.nan,
            "breakeven_armed": False,
            "position_weight": np.nan,
            "held_days": np.nan,
        })
    trade_log.append(event)



def stop_level(position: dict[str, Any], symbol: str) -> float:
    hard = position["avg_price"] * (1 - hard_stop_pct(symbol))
    if cfg("USE_BREAKEVEN_PROTECTION", False) and position.get("breakeven_armed", False):
        return max(hard, position["avg_price"] * (1 + float(cfg("BREAKEVEN_OFFSET_PCT", 0.0))))
    return hard



def intraperiod_exits(portfolio: dict[str, Any], previous_date: str, date: str, market_data: dict[str, pd.DataFrame]):
    """Evalúa stops con velas diarias entre dos cortes de decisión.


    Supuesto de gap: si la apertura está peor que el stop, se ejecuta a apertura;
    de lo contrario al nivel del stop. El break-even se arma con retardo de una vela:
    - Si el cierre cruza el trigger, se marca breakeven_pending = True.
    - En la siguiente vela, breakeven_armed = True.
    Esto evita que en una misma vela se usen high para activar y low para ejecutar.
    """
    exits = []
    trigger = cfg("BREAKEVEN_TRIGGER_PCT", None)
    for symbol in list(portfolio["positions"]):
        position = portfolio["positions"][symbol]
        bars = get_daily_window(market_data, symbol, previous_date, date)
        for i, (_, bar) in enumerate(bars.iterrows()):
            high = float(bar["high"])
            low = float(bar["low"])
            open_price = float(bar["open"]) if pd.notna(bar.get("open", np.nan)) else float(bar["close"])
            position["highest_price"] = max(position.get("highest_price", high), high)


            # Break-even con retardo de una vela
            if cfg("USE_BREAKEVEN_PROTECTION", False) and trigger:
                # Si ya estaba pendiente, se arma al inicio de esta vela
                if position.get("breakeven_pending", False):
                    position["breakeven_armed"] = True
                    position["breakeven_pending"] = False
                # Si el cierre cruza el trigger, marcar pendiente para la siguiente vela
                close_price = float(bar["close"])
                if close_price >= position["avg_price"] * (1 + float(trigger)):
                    position["breakeven_pending"] = True


            level = stop_level(position, symbol)
            position["stop_level"] = level
            if low <= level:
                exit_price = min(open_price, level)
                reason = "break-even" if position.get("breakeven_armed", False) else "stop-loss"
                exits.append((symbol, exit_price, reason, str(bar["date"]), position.copy()))
                break
    return exits



def close_position(portfolio: dict[str, Any], symbol: str, price: float) -> dict[str, Any]:
    position = portfolio["positions"].pop(symbol)
    portfolio["cash"] += position["quantity"] * price * (1 - config.TRANSACTION_COST_PCT)
    position["realized_return"] = price / position["avg_price"] - 1
    return position



def open_position(portfolio: dict[str, Any], symbol: str, price: float, date: str, market_data: dict[str, pd.DataFrame]) -> dict[str, Any] | None:
    weight = position_weight(symbol)
    total_value = portfolio_value(portfolio, date, market_data)
    max_spend = portfolio["cash"] / (1 + config.TRANSACTION_COST_PCT)
    spend = min(total_value * weight, max_spend)
    if spend < 1:
        return None
    quantity = spend / price
    portfolio["cash"] -= spend * (1 + config.TRANSACTION_COST_PCT)
    position = {
        "quantity": quantity,
        "avg_price": price,
        "last_price": price,
        "highest_price": price,
        "weight": weight,
        "entry_date": date,
        "breakeven_armed": False,
        "breakeven_pending": False,
        "stop_level": price * (1 - hard_stop_pct(symbol)),
    }
    portfolio["positions"][symbol] = position
    return position



# ---------------------------------------------------------------------------
# Backtest diario
# ---------------------------------------------------------------------------



def run_backtest(start: str | None = None, end: str | None = None, verbose: bool = True):
    start = start or config.BACKTEST_START
    end = end or config.BACKTEST_END
    if verbose:
        print("Cargando datos historicos...")
    market_data = load_all_data(config.ALLOWED_ASSETS, config.INDICATOR_WARMUP_START, end, verbose)
    if not market_data:
        return pd.DataFrame(), {"cash": float(config.INITIAL_CAPITAL), "positions": {}}, market_data


    dates = [stamp.strftime("%Y-%m-%d") for stamp in pd.date_range(start, end, freq=config.DECISION_FREQUENCY)]
    if verbose:
        print(f"Fechas a procesar: {len(dates)}\n")


    portfolio = {"cash": float(config.INITIAL_CAPITAL), "positions": {}}
    trade_log: list[dict[str, Any]] = []
    cooldown_until: dict[str, str] = {}
    pending: dict[str, Any] | None = None
    previous_date = start


    for index, date in enumerate(dates):
        # Reporte cada 26 días en modo diario o cada 26 semanas en modo semanal.
        if verbose and index and index % 26 == 0:
            print(f"\n[{date}] Valor: {portfolio_value(portfolio, date, market_data):,.2f} | Cash: {portfolio['cash']:,.2f} | Pos: {list(portfolio['positions']) or 'ninguna'}")


        # 1. Stops / break-even intraperiodo. Con frecuencia diaria se inspecciona
        # la vela del día actual una vez; con frecuencia semanal recorre la semana.
        for symbol, exit_price, reason, exit_date, _ in intraperiod_exits(portfolio, previous_date, date, market_data):
            if symbol not in portfolio["positions"]:
                continue
            closed = close_position(portfolio, symbol, exit_price)
            cooldown_weeks = cooldown_weeks_after_stop(symbol) if reason == "stop-loss" else 0
            if cooldown_weeks > 0:
                cooldown_until[symbol] = add_weeks(exit_date, cooldown_weeks)
            log_event(trade_log, exit_date, symbol, "sell", reason, f"auto {reason}", portfolio, market_data, decision_source="risk_rule", position=closed, execution_price=exit_price, trade_return=100 * closed["realized_return"], cooldown_weeks=cooldown_weeks)
            if verbose:
                suffix = f" | cooldown {cooldown_weeks} sem" if cooldown_weeks else ""
                print(f"  [{exit_date}] AUTO {reason} {symbol} @ {exit_price:,.2f} | retorno {100 * closed['realized_return']:+.1f}%{suffix}")


        # 2. Salida por pérdida de régimen.
        for symbol in list(portfolio["positions"]):
            row = get_row(market_data, symbol, date)
            price = get_price(market_data, symbol, date)
            exit_now, reason = should_exit_position(row)
            if exit_now and price is not None:
                closed = close_position(portfolio, symbol, price)
                log_event(trade_log, date, symbol, "sell", reason, f"venta @ {price:,.2f}", portfolio, market_data, decision_source="risk_rule", position=closed, execution_price=price, trade_return=100 * closed["realized_return"])
                if verbose:
                    print(f"  [{date}] Venta {symbol} @ {price:,.2f} — {reason} | retorno {100 * closed['realized_return']:+.1f}%")


        # 3. Ejecuta la decisión tomada en el ciclo previo.
        if pending and pending["action"] == "buy":
            symbol = pending["symbol"]
            snapshot = pending.get("snapshot", [])
            price = get_price(market_data, symbol, date)
            if symbol in market_data and symbol not in portfolio["positions"] and price is not None:
                for held_symbol in list(portfolio["positions"]):
                    held_price = get_price(market_data, held_symbol, date) or portfolio["positions"][held_symbol]["avg_price"]
                    closed = close_position(portfolio, held_symbol, held_price)
                    log_event(trade_log, date, held_symbol, "sell", f"rotacion a {symbol}", "venta rotacion", portfolio, market_data, snapshot, "execution", closed, held_price, 100 * closed["realized_return"], pending)
                    if verbose:
                        print(f"  [{date}] Venta {held_symbol} @ {held_price:,.2f} — rotacion a {symbol} | retorno {100 * closed['realized_return']:+.1f}%")


                position = open_position(portfolio, symbol, price, date, market_data)
                if position:
                    row = next((item for item in snapshot if item["symbol"] == symbol), None)
                    reason = narrate("buy", symbol, date, row, pending["reason"])
                    log_event(trade_log, date, symbol, "buy", reason, f"compra {position['quantity']:.6f} @ {price:,.2f}", portfolio, market_data, snapshot, pending["decision_source"], position, price, decision=pending)
                    if verbose:
                        print(f"  [{date}] Compra {symbol} @ {price:,.2f} | peso {position['weight']:.1%} — {reason}")
        elif pending:
            log_event(trade_log, date, pending.get("symbol", ""), "hold", pending["reason"], "hold", portfolio, market_data, pending.get("snapshot", []), pending["decision_source"], portfolio["positions"].get(pending.get("symbol", "")), decision=pending)


        current_position = next(iter(portfolio["positions"]), None)
        pending = decide(market_data, date, current_position, cooldown_until, portfolio["positions"].get(current_position))
        previous_date = date


    if dates:
        final_symbol = next(iter(portfolio["positions"]), "")
        final_position = portfolio["positions"].get(final_symbol)
        log_event(trade_log, dates[-1], final_symbol, "hold", "valoracion final", "mark_to_market", portfolio, market_data, position=final_position)
    return pd.DataFrame(trade_log), portfolio, market_data



# ---------------------------------------------------------------------------
# Métricas e informe
# ---------------------------------------------------------------------------



def equity_curve(log_df: pd.DataFrame) -> pd.Series:
    if log_df.empty:
        return pd.Series(dtype=float)
    frame = log_df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["portfolio_value"] = pd.to_numeric(frame["portfolio_value"], errors="coerce")
    return frame.dropna(subset=["date", "portfolio_value"]).sort_values("date").groupby("date")["portfolio_value"].last().sort_index()



def metrics(curve: pd.Series, initial: float | None = None) -> dict[str, float]:
    if curve.empty:
        return {}
    initial_value = float(initial if initial is not None else curve.iloc[0])
    final = float(curve.iloc[-1])
    years = max((curve.index[-1] - curve.index[0]).days / 365.25, 1 / 365.25)
    drawdowns = curve / curve.cummax() - 1
    max_drawdown = float(drawdowns.min())
    returns = curve.pct_change().dropna()
    periods_per_year = float(cfg("SHARPE_PERIODS_PER_YEAR", 365 if config.DECISION_FREQUENCY == "D" else 52))
    sharpe = float(returns.mean() / returns.std() * np.sqrt(periods_per_year)) if len(returns) > 1 and returns.std() > 0 else 0.0
    cagr = (final / initial_value) ** (1 / years) - 1
    return {
        "final": final,
        "total_return_pct": 100 * (final / initial_value - 1),
        "cagr_pct": 100 * cagr,
        "max_drawdown_pct": 100 * max_drawdown,
        "sharpe": sharpe,
        "calmar": cagr / abs(max_drawdown) if max_drawdown < 0 else np.nan,
    }



def buy_and_hold(market_data: dict[str, pd.DataFrame], symbol: str, start: str, end: str) -> pd.Series:
    df = market_data.get(symbol)
    if df is None:
        return pd.Series(dtype=float)
    window = df.loc[(df.index >= start) & (df.index <= end)]
    if window.empty:
        return pd.Series(dtype=float)
    quantity = config.INITIAL_CAPITAL / float(window.iloc[0]["close"])
    curve = quantity * window["close"].astype(float)
    curve.index = pd.to_datetime(window.index)
    return curve



def print_summary(log_df: pd.DataFrame, final_portfolio: dict[str, Any], market_data: dict[str, pd.DataFrame], start: str, end: str) -> None:
    if log_df.empty:
        print("Sin registros para resumir.")
        return
    curve = equity_curve(log_df)
    est = metrics(curve, config.INITIAL_CAPITAL)
    executed = log_df[log_df["action"].isin(["buy", "sell"])]
    sells = log_df[(log_df["action"] == "sell") & log_df["trade_return_pct"].notna()].copy()


    print("\n" + "=" * 72)
    print("RESUMEN DEL BACKTEST — return-first diario")
    print("=" * 72)
    print(f"Periodo:                  {start} a {end}")
    print(f"Capital inicial:          {config.INITIAL_CAPITAL:,.2f} USDT")
    print(f"Valor final:              {est['final']:,.2f} USDT")
    print(f"Retorno acumulado:        {est['total_return_pct']:+.2f}%")
    print(f"CAGR:                     {est['cagr_pct']:+.2f}%")
    print(f"Max drawdown:             {est['max_drawdown_pct']:.2f}%")
    print(f"Sharpe (diario anualiz.): {est['sharpe']:.2f}")
    print(f"Calmar:                   {est['calmar']:.2f}" if pd.notna(est["calmar"]) else "Calmar:                   n/a")
    print(f"Ordenes ejecutadas:       {len(executed)}")
    if not sells.empty:
        print(f"Operaciones cerradas:     {len(sells)}")
        print(f"Aciertos:                 {100 * (sells['trade_return_pct'] > 0).mean():.1f}%")
        print(f"Retorno medio por trade:  {sells['trade_return_pct'].mean():+.2f}%")
    print(f"Posiciones abiertas:      {list(final_portfolio['positions']) or 'ninguna'}")
    print(f"Config: activos={config.ALLOWED_ASSETS} | frecuencia={config.DECISION_FREQUENCY} | min_hold_dias={config.ROTATION_MIN_HOLD_DAYS} | margen={config.ROTATION_SCORE_MARGIN} | coste={config.TRANSACTION_COST_PCT:.2%}")


    print("\nBenchmarks sobre los mismos datos:")
    for symbol in config.ALLOWED_ASSETS:
        bh = buy_and_hold(market_data, symbol, start, end)
        if not bh.empty:
            stat = metrics(bh, config.INITIAL_CAPITAL)
            print(f"  {symbol:<28} retorno {stat['total_return_pct']:+10.2f}% | CAGR {stat['cagr_pct']:+7.2f}% | max DD {stat['max_drawdown_pct']:7.2f}%")


    annual = curve.resample("YE").last().pct_change().dropna() * 100
    if not annual.empty:
        print("\nRetorno por año:")
        for stamp, value in annual.items():
            print(f"  {stamp.year}: {value:+.1f}%")


    if not sells.empty:
        print("\nVentas por motivo:")
        for reason, count in sells["reason"].value_counts().items():
            print(f"  - {reason}: {count}")
    print("=" * 72)



def run_walk_forward() -> None:
    print("WALK-FORWARD: configuración diaria, ventanas independientes\n")
    rows: list[dict[str, Any]] = []
    for start, end in cfg("WALK_FORWARD_WINDOWS", []):
        print("=" * 72)
        print(f"VENTANA: {start} a {end}")
        print("=" * 72)
        log_df, _, market_data = run_backtest(start, end, verbose=True)
        if log_df.empty:
            continue
        est = metrics(equity_curve(log_df), config.INITIAL_CAPITAL)
        row: dict[str, Any] = {
            "ventana": f"{start} a {end}",
            "retorno_%": round(est["total_return_pct"], 1),
            "cagr_%": round(est["cagr_pct"], 1),
            "max_dd_%": round(est["max_drawdown_pct"], 1),
            "sharpe": round(est["sharpe"], 2),
        }
        for symbol in config.ALLOWED_ASSETS:
            bh = buy_and_hold(market_data, symbol, start, end)
            if not bh.empty:
                stat = metrics(bh, config.INITIAL_CAPITAL)
                row[f"{symbol} ret_%"] = round(stat["total_return_pct"], 1)
                row[f"{symbol} dd_%"] = round(stat["max_drawdown_pct"], 1)
        rows.append(row)
    print("\n" + "=" * 72)
    print("RESULTADO WALK-FORWARD")
    print("=" * 72)
    print(pd.DataFrame(rows).to_string(index=False))



if __name__ == "__main__":
    if "--walk" in sys.argv:
        run_walk_forward()
        raise SystemExit(0)


    print(f"Backtest diario: {config.BACKTEST_START} a {config.BACKTEST_END}")
    print(f"Capital: {config.INITIAL_CAPITAL:.2f} USDT | Narrativa LLM: {'si' if cfg('USE_LLM_NARRATIVE', False) else 'no'}\n")
    log_df, final_portfolio, market_data = run_backtest()
    if not log_df.empty:
        log_df.to_csv("backtest_log.csv", index=False)
        print("\nLog guardado en backtest_log.csv")
        print_summary(log_df, final_portfolio, market_data, config.BACKTEST_START, config.BACKTEST_END)
