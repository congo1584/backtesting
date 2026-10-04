"""Configuración completa para backtester_batch_daily_28d.py.

Cambios respecto a la config semanal: DECISION_FREQUENCY = "D", permanencia mínima
de rotación en días naturales (ROTATION_MIN_HOLD_DAYS = 28) y Sharpe anualizado con 365.
Las cifras de rendimiento citadas más abajo se midieron con la versión SEMANAL y el
break-even sin retardo de vela: vuelve a medirlas con el backtester diario.


Variante v8 (validada offline sobre 2016-2026 con datos diarios BTC-USD y
^GSPC). Mantiene la filosofía return-first de v4 —exposición plena al mejor
activo elegible, sin trailing ATR, sin salida SMA50, sin circuit breaker de
drawdown— y añade sólo tres cambios estructurales:

1. Stops duros más anchos: BTC 20% y S&P 10%. Los stops de 12%/7% de v4 se
   activaban dentro del ruido normal y convertían tendencias válidas en
   pérdidas completas.
2. Protección a break-even: cuando una operación acumula +10% de beneficio,
   el stop sube al precio de entrada más costes. Un ganador nunca vuelve a
   convertirse en un perdedor grande. Es la mejora que más reduce el drawdown.
3. Reglas simétricas: el margen de rotación es el mismo para los dos activos
   (0,25). Una versión previa usaba un margen especial de 2,00 sólo para BTC;
   subía el retorno en BTC+S&P pero empeoraba en ETH+Nasdaq (+421% con DD -60%
   frente a +874% con DD -32% usando el margen simétrico). Era ajuste al
   histórico de BTC, no un principio general, y se ha eliminado.

Validación fuera de muestra: estas reglas se probaron sin cambios en pares que
no intervinieron en su diseño (ETH/Nasdaq, Nasdaq/Oro, DAX/Oro, Oro/S&P). No
baten al buy&hold del activo de riesgo en todos ellos, pero mantienen el
drawdown en la banda -23%/-32%; ver las notas al final del archivo.

Selección determinista: Python decide elegibilidad, score, rotación y tamaño.
El LLM sólo redacta la justificación (opcional, USE_LLM_NARRATIVE) y no puede
cambiar la operación. Así el resultado es reproducible y mucho más barato.
"""

# ---------------------------------------------------------------------------
# Modelo (sólo narrativa; no decide operaciones)
# ---------------------------------------------------------------------------

CLAUDE_MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 120
USE_LLM_NARRATIVE = False       # True = pide a Claude la frase de motivo
NARRATE_ONLY_ON_TRADES = True   # sólo llama al LLM en compras y ventas

# ---------------------------------------------------------------------------
# Riesgo, tamaño y costes
# ---------------------------------------------------------------------------

# Exposición plena: cada operación compromete el 100% del capital. Es una
# decisión de riesgo consciente, no heredada. Coste de reducirla (2016-2026):
#   peso BTC 1,00 -> +29.303%, DD -24,0%
#   peso BTC 0,85 -> +17.459%, DD -21,8%
#   peso BTC 0,75 -> +11.942%, DD -20,6%
#   peso BTC 0,50 ->  +3.967%, DD -17,3%
# Para operar con colchón, baja sólo el peso de BTC aquí.
MAX_POSITION_SIZE_PCT = 1.00
MIN_POSITION_WEIGHT_BY_ASSET = {"BTC/USDT": 1.00, "S&P 500": 1.00}
MAX_POSITION_WEIGHT_BY_ASSET = {"BTC/USDT": 1.00, "S&P 500": 1.00}

# Cambio 1: stops anchos, por encima del ruido semanal de cada activo.
STOP_LOSS_PCT = 0.10
STOP_LOSS_PCT_BY_ASSET = {
    "BTC/USDT": 0.20,
    "S&P 500": 0.10,
}

# Cambio 2: protección a break-even.
USE_BREAKEVEN_PROTECTION = True
BREAKEVEN_TRIGGER_PCT = 0.10     # beneficio máximo alcanzado que arma el nivel
BREAKEVEN_OFFSET_PCT = 0.002     # colchón sobre el precio de entrada (2 patas)

TAKE_PROFIT_PCT = None           # sin take-profit fijo: se deja correr la tendencia
USE_TRAILING_STOP = False        # descartado en v2/v3: corta la convexidad
USE_SMA50_TACTICAL_EXIT = False  # descartado: salidas demasiado tempranas

TRANSACTION_COST_PCT = 0.001
COOLDOWN_WEEKS_AFTER_SL = 2
MAX_CONSECUTIVE_API_ERRORS = 3

# ---------------------------------------------------------------------------
# Universo y periodo
# ---------------------------------------------------------------------------

ALLOWED_ASSETS = ["BTC/USDT", "S&P 500"]
INITIAL_CAPITAL = 10_000.0

INDICATOR_WARMUP_START = "2015-01-01"
BACKTEST_START = "2016-01-01"
BACKTEST_END = "2026-08-30"
DECISION_FREQUENCY = "D"            # señales diarias; ejecución en la fecha siguiente
SHARPE_PERIODS_PER_YEAR = 365       # 365 si hay cripto en la curva (calendario diario)

# Ventanas de walk-forward para el informe final. No se optimiza nada por
# ventana: son la misma configuración evaluada en regímenes distintos.
WALK_FORWARD_WINDOWS = [
    ("2016-01-01", "2020-12-31"),
    ("2018-01-01", "2026-08-30"),
    ("2021-01-01", "2026-08-30"),
    ("2022-01-01", "2026-08-30"),
]

# ---------------------------------------------------------------------------
# Señal: tendencia, ADX, ER y RSI (congelados desde v4)
# ---------------------------------------------------------------------------

RSI_PERIOD = 14
ADX_PERIOD = 14
ADX_MIN = 20.0

ER_PERIOD = 20
ER_MIN = 0.30

ENTRY_TREND_BUFFER_ATR = 0.5
EXIT_TREND_BUFFER_ATR = 0.5

RSI_MAX_NORMAL_ENTRY = 70.0
ADX_MIN_EXTENDED_ENTRY = 25.0
ER_MIN_EXTENDED_ENTRY = 0.40

# ---------------------------------------------------------------------------
# Momentum y rotación
# ---------------------------------------------------------------------------

MOMENTUM_5D_WEIGHT = 0.15
MOMENTUM_20D_WEIGHT = 0.35
MOMENTUM_60D_WEIGHT = 0.50

ROTATION_SCORE_MARGIN = 0.25
BTC_ROTATION_SCORE_MARGIN = 0.25   # simétrico a propósito: ver punto 3 del docstring
ROTATION_MIN_HOLD_DAYS = 28       # días naturales (equivale a las 4 semanas anteriores)

# ---------------------------------------------------------------------------
# Volatilidad / extensión de entrada
# ---------------------------------------------------------------------------

BB_PERIOD = 20
BB_STDDEV = 2.0
BB_PCT_B_MAX_ENTRY = 1.10
BB_PCT_B_MAX_EXTENDED_ENTRY = 1.20

USE_VOLUME_FILTER = False
REL_VOLUME_MIN = 1.00

# ---------------------------------------------------------------------------
# Exchange / modo operativo
# ---------------------------------------------------------------------------

EXCHANGE_NAME = "binance"
PAPER_TRADING = True

# ---------------------------------------------------------------------------
# Límites conocidos de esta configuración (medidos, no supuestos)
# ---------------------------------------------------------------------------
# - Concentración: excluyendo la mejor operación el retorno 2016-2026 baja de
#   +29.303% a +7.104%, por debajo del buy&hold de BTC (+18.147%). La ventaja
#   sobre BTC depende de capturar una o dos tendencias grandes.
# - Mercados laterales: 2018 +2,0%, nov-2021 a oct-2023 +7,9%. No pierde, pero
#   tampoco gana; el periodo más largo bajo máximos es de 82 semanas.
# - 2021-2026 pierde contra BTC (+86% vs +147%) con un tercio del drawdown.
# - Con coste 0,50% por pata el drawdown sube a -39%: rompe el límite del 35%.
# - Fuera de muestra (mismas reglas, otros pares): ETH/Nasdaq +874% DD -32%,
#   Nasdaq/Oro +77% DD -26%, DAX/Oro +164% DD -31%, Oro/S&P +195% DD -31%.
#   En esos pares no bate al activo de riesgo: la ventaja observada es
#   específica del par BTC + S&P 500.
