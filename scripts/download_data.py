from pathlib import Path
from datetime import datetime
import pandas as pd
import yfinance as yf

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

START = "2006-01-01"
END = datetime.today().strftime("%Y-%m-%d")

ETF_TICKERS = ["SPY", "XLE", "XLF", "XLI", "XLK", "XLV"]

FRED_SERIES = {
    "VIXCLS": "vix",
    "USEPUINDXD": "epu",
    "WLEMUINDXD": "emeu",
}


def download_etfs() -> pd.DataFrame:
    # Download daily ETF price data from Yahoo Finance
    frames = []

    for ticker in ETF_TICKERS:
        df = yf.download(
            ticker,
            start=START,
            end=END,
            auto_adjust=False,
            progress=False,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df["ticker"] = ticker
        frames.append(df)

    etfs = pd.concat(frames, ignore_index=True)
    etfs.columns = [
        str(col).lower().replace(" ", "_").replace("-", "_")
        for col in etfs.columns
    ]
    etfs = etfs.rename(columns={"date": "date"})
    return etfs


def download_fred_series(series_id: str, clean_name: str) -> pd.DataFrame:
    # Download series from FRED using direct CSV endpoint
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

    df = pd.read_csv(url)
    df = df.rename(
        columns={
            "observation_date": "date",
            series_id: clean_name,
        }
    )

    df["date"] = pd.to_datetime(df["date"])
    df[clean_name] = pd.to_numeric(df[clean_name], errors="coerce")
    df = df[df["date"] >= pd.to_datetime(START)]

    return df[["date", clean_name]]


def download_fred_uncertainty() -> pd.DataFrame:
    # Download and combine all FRED uncertainty measures.
    frames = []
    for series_id, clean_name in FRED_SERIES.items():
        frames.append(download_fred_series(series_id, clean_name).set_index("date"))
    fred = pd.concat(frames, axis=1).reset_index()
    fred = fred.sort_values("date")

    return fred


def main() -> None:
    etfs = download_etfs()
    etf_path = RAW / "etf_prices.csv"
    etfs.to_csv(etf_path, index=False)
    print(f"Saved {etf_path} with shape {etfs.shape}")

    fred = download_fred_uncertainty()
    fred_path = RAW / "fred_uncertainty.csv"
    fred.to_csv(fred_path, index=False)
    print(f"Saved {fred_path} with shape {fred.shape}")


if __name__ == "__main__":
    main()
