"""
Definición de las tools que Claude puede invocar durante la decisión.

Incluye las funciones de tools originales y load_csv(), que necesita el
backtester determinista. v8.2 añade exclusivamente los símbolos Nasdaq-100,
Oro, Bono del Tesoro de EE. UU. 7-10 años y acciones globales ex EE. UU.
"""

from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# 1. Esquema de tools para la API de Claude (formato tool_use)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "get_market_data",
        "description": (
            "Obtiene precio actual, velas históricas recientes e indicadores "
            "técnicos básicos (SMA, RSI) para un activo."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ej: BTC/USDT"},
                "lookback_periods": {
                    "type": "integer",
                    "description": "Número de velas históricas a incluir",
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_portfolio_state",
        "description": "Devuelve el estado actual del portafolio: efectivo y posiciones abiertas.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "propose_order",
        "description": (
            "Propone una orden de compra/venta. NO ejecuta directamente; "
            "pasa por validación determinista antes de llegar al exchange."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "action": {"type": "string", "enum": ["buy", "sell", "hold"]},
                "quantity": {"type": "number"},
                "reason": {"type": "string", "description": "Justificación breve de la decisión"},
            },
            "required": ["symbol", "action", "reason"],
        },
    },
]


# ---------------------------------------------------------------------------
# 2. Implementaciones de cada tool y carga de CSV
# ---------------------------------------------------------------------------

# Cache simple para no releer el CSV en cada llamada.
_CSV_CACHE: dict[str, pd.DataFrame] = {}


# Mapea símbolo -> ruta del CSV descargado con download_data.py.
# Los seis símbolos de config.ALLOWEDASSETS deben coincidir exactamente aquí.
CSV_PATHS = {
    "BTC/USDT": "data/BTC_USDT.csv",
    "ETH/USDT": "data/ETH_USDT.csv",
    "EUR/USD": "data/EUR_USD.csv",
    "EUR/GBP": "data/EUR_GBP.csv",
    "S&P 500": "data/SP500.csv",
    "IBEX 35": "data/IBEX35.csv",
    "NASDAQ": "data/NASDAQ.csv",
    "BITI (short BTC)": "data/BITI.csv",

    # Activos nuevos del universo v8.2.
    "Nasdaq-100": "data/NASDAQ100.csv",
    "Oro": "data/ORO.csv",
    "Bono Tesoro EEUU 7-10A": "data/TREASURY_7_10Y.csv",
    "Acciones globales ex EEUU": "data/GLOBAL_EX_US.csv",
}


_REQUIRED_PRICE_COLUMNS = ["date", "open", "high", "low", "close"]


def _normalize_csv(df: pd.DataFrame, path: str) -> pd.DataFrame:
    """Normaliza el formato de los CSV descargados para todas las tools."""
    df = df.copy()
    df.columns = [str(column).strip().lower().replace(" ", "_") for column in df.columns]

    if "datetime" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"datetime": "date"})
    missing = [column for column in _REQUIRED_PRICE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{path} no contiene columnas obligatorias: {missing}")
    if "volume" not in df.columns:
        df["volume"] = 0.0

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = (
        df[["date", "open", "high", "low", "close", "volume"]]
        .dropna(subset=_REQUIRED_PRICE_COLUMNS)
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    if df.empty:
        raise ValueError(f"{path} no contiene observaciones válidas tras normalizarlo.")
    return df


def _load_csv(symbol: str) -> pd.DataFrame:
    """Carga el CSV de un símbolo y lo conserva en memoria como cache."""
    if symbol not in _CSV_CACHE:
        path = CSV_PATHS.get(symbol)
        if path is None:
            raise ValueError(f"No hay CSV configurado para {symbol} en tools.CSV_PATHS")
        if not Path(path).exists():
            raise FileNotFoundError(
                f"No se encontró {path} para {symbol}. Ejecuta download_data.py "
                "o revisa CSV_PATHS."
            )
        _CSV_CACHE[symbol] = _normalize_csv(pd.read_csv(path), path)
    return _CSV_CACHE[symbol]


def load_csv(symbol: str) -> pd.DataFrame:
    """API pública requerida por backtester_batch.py.

    Devuelve una copia para que el backtester pueda añadir indicadores sin
    modificar el objeto almacenado en la cache ni las tools de Claude.
    """
    return _load_csv(symbol).copy()


def get_market_data(symbol: str, lookback_periods: int = 30, as_of_date: str | None = None) -> dict[str, Any]:
    """Devuelve datos hasta as_of_date inclusiva, evitando look-ahead bias."""
    if lookback_periods < 1:
        raise ValueError("lookback_periods debe ser al menos 1")

    df = _load_csv(symbol)
    if as_of_date is not None:
        df = df[df["date"] <= pd.Timestamp(as_of_date)]

    if df.empty:
        return {"symbol": symbol, "error": "sin datos disponibles hasta esa fecha"}

    window = df.tail(lookback_periods).copy()
    df_with_indicators = df.copy()
    df_with_indicators["sma_10"] = df_with_indicators["close"].rolling(10).mean()
    delta = df_with_indicators["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, pd.NA)
    df_with_indicators["rsi_14"] = 100 - (100 / (1 + rs))

    window = window.merge(
        df_with_indicators[["date", "sma_10", "rsi_14"]],
        on="date",
        how="left",
    )
    last_row = window.iloc[-1]
    comparison_row = window.iloc[-min(5, len(window))]
    pct_change_5d = (last_row["close"] - comparison_row["close"]) / comparison_row["close"] * 100

    return {
        "symbol": symbol,
        "last_price": float(last_row["close"]),
        "sma_10": None if pd.isna(last_row["sma_10"]) else round(float(last_row["sma_10"]), 4),
        "rsi_14": None if pd.isna(last_row["rsi_14"]) else round(float(last_row["rsi_14"]), 1),
        "pct_change_5d": round(float(pct_change_5d), 2),
    }


def get_portfolio_state(portfolio: dict[str, Any]) -> dict[str, Any]:
    """Devuelve una copia del estado del portafolio simulado o real."""
    return {
        "cash": portfolio["cash"],
        "positions": portfolio["positions"],
    }


def propose_order(symbol: str, action: str, reason: str, quantity: float = 0.0) -> dict[str, Any]:
    """No ejecuta nada: empaqueta una propuesta para risk_manager.py."""
    return {
        "symbol": symbol,
        "action": action,
        "quantity": quantity,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# 3. Router: conecta el nombre de tool de Claude con la función real
# ---------------------------------------------------------------------------


def execute_tool(
    name: str,
    tool_input: dict[str, Any],
    portfolio: dict[str, Any],
    as_of_date: str | None = None,
) -> Any:
    if name == "get_market_data":
        # Se fuerza as_of_date desde el loop de backtest para impedir que Claude
        # solicite por error datos futuros.
        tool_input = {**tool_input, "as_of_date": as_of_date}
        return get_market_data(**tool_input)
    if name == "get_portfolio_state":
        return get_portfolio_state(portfolio)
    if name == "propose_order":
        return propose_order(**tool_input)
    raise ValueError(f"Tool desconocida: {name}")
