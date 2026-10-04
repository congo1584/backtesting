import argparse
import time
import pandas as pd


def download_yfinance(symbol, start, end, interval="1d"):
    import yfinance as yf
    df = yf.download(symbol, start=start, end=end, interval=interval,
                     progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError("Sin datos para " + symbol)
    df = df.reset_index()
    if hasattr(df.columns, "levels"):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    if "datetime" in df.columns:
        df = df.rename(columns={"datetime": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    return df[["date", "open", "high", "low", "close", "volume"]].dropna()


SYMBOL_MAP = {
    "BTC/USDT":         ("yfinance", "BTC-USD",  "data/BTC_USDT.csv"),
    "ETH/USDT":         ("yfinance", "ETH-USD",  "data/ETH_USDT.csv"),
    "EUR/USD":          ("yfinance", "EURUSD=X", "data/EUR_USD.csv"),
    "EUR/GBP":          ("yfinance", "EURGBP=X", "data/EUR_GBP.csv"),
    "S&P 500":          ("yfinance", "^GSPC",    "data/SP500.csv"),
    "IBEX 35":          ("yfinance", "^IBEX",    "data/IBEX35.csv"),
    "NASDAQ":           ("yfinance", "^IXIC",    "data/NASDAQ.csv"),
    "BITI (short BTC)": ("yfinance", "BITI",     "data/BITI.csv"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=False)
    parser.add_argument("--start",  required=True)
    parser.add_argument("--end",    required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        errores = []
        for name, (_, symbol, output) in SYMBOL_MAP.items():
            print("Descargando " + name + "...")
            try:
                df = download_yfinance(symbol, args.start, args.end)
                df.to_csv(output, index=False)
                print("  OK " + str(len(df)) + " filas -> " + output)
            except Exception as e:
                print("  ERROR: " + str(e))
                errores.append(name)
            time.sleep(3)
        if errores:
            print("Fallaron: " + str(errores))
        return

    df = download_yfinance(args.symbol, args.start, args.end)
    safe = args.symbol.replace("/", "_")
    output = args.output or (safe + "_" + args.start + "_" + args.end + ".csv")
    df.to_csv(output, index=False)
    print("OK " + output + " (" + str(len(df)) + " filas)")


if __name__ == "__main__":
    main()