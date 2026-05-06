from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

SECTOR_MAP = {
    "SPY": "Market", "XLE": "Energy", "XLF": "Financials",
    "XLI": "Industrials", "XLK": "Technology", "XLV": "Healthcare",
}

def realized_vol_forward(x: pd.Series, horizon: int) -> pd.Series:
    # Rolling std of future returns
    return (x.shift(-1).rolling(window=horizon, min_periods=horizon).std().shift(-(horizon - 1)))


def build_etf_features(etf: pd.DataFrame) -> pd.DataFrame:
    etf = etf.sort_values(["ticker", "date"]).copy()

    # Use adjusted close if available, otherwise, use close
    price_col = "adj_close" if "adj_close" in etf.columns else "close"

    etf["log_price"] = np.log(etf[price_col])
    etf["return_1d"] = etf.groupby("ticker")["log_price"].diff()
    etf["abs_return_1d"] = etf["return_1d"].abs()

    grouped_return = etf.groupby("ticker")["return_1d"]

    etf["past_vol_5d"] = grouped_return.transform(lambda x: x.rolling(5, min_periods=5).std())
    etf["past_vol_21d"] = grouped_return.transform(lambda x: x.rolling(21, min_periods=21).std())
    etf["past_return_5d"] = grouped_return.transform(lambda x: x.rolling(5, min_periods=5).sum())
    etf["past_return_21d"] = grouped_return.transform(lambda x: x.rolling(21, min_periods=21).sum())

    if "volume" in etf.columns:
        etf["log_volume"] = np.log1p(etf["volume"])
        etf["volume_change_1d"] = etf.groupby("ticker")["log_volume"].diff()
        etf["volume_change_5d"] = etf.groupby("ticker")["log_volume"].diff(5)

    etf["future_vol_5d"] = grouped_return.transform(lambda x: realized_vol_forward(x, 5))
    etf["future_vol_21d"] = grouped_return.transform(lambda x: realized_vol_forward(x, 21))
    etf["sector"] = etf["ticker"].map(SECTOR_MAP)
    return etf



def build_uncertainty_features(fred: pd.DataFrame) -> pd.DataFrame:
    # Build the same features for VIX, EPU and EMEU
    fred = fred.sort_values("date").copy()
    for col in ["vix", "epu", "emeu"]:
        if col in fred.columns:
            fred[col] = fred[col].ffill()

            fred[f"{col}_change_5d"] = fred[col].diff(5)
            fred[f"{col}_avg_21d"] = fred[col].rolling(21, min_periods=21).mean()

            rolling_mean = fred[col].rolling(252, min_periods=100).mean()
            rolling_sd = fred[col].rolling(252, min_periods=100).std()
            rolling_q95 = fred[col].rolling(252, min_periods=100).quantile(0.95)

            fred[f"{col}_z_252d"] = (fred[col] - rolling_mean) / rolling_sd
            fred[f"{col}_spike_95"] = (fred[col] > rolling_q95).astype(int)

    return fred


def main() -> None:
    etf = pd.read_csv(RAW / "etf_prices.csv", parse_dates=["date"])
    fred = pd.read_csv(RAW / "fred_uncertainty.csv", parse_dates=["date"])

    etf_features = build_etf_features(etf)
    fred_features = build_uncertainty_features(fred)

    df = etf_features.merge(fred_features, on="date", how="left")
    uncertainty_cols = [c for c in fred_features.columns if c != "date"]
    df[uncertainty_cols] = df[uncertainty_cols].ffill()

    model_cols = [
        "date", "ticker", "sector", "return_1d", "abs_return_1d", "past_vol_5d", "past_vol_21d", 
        "past_return_5d", "past_return_21d", "volume_change_1d", "volume_change_5d",
        "vix", "vix_change_5d", "vix_avg_21d", "vix_z_252d", "vix_spike_95", 
        "epu", "epu_change_5d", "epu_avg_21d", "epu_z_252d", "epu_spike_95",
        "emeu", "emeu_change_5d", "emeu_avg_21d", "emeu_z_252d", "emeu_spike_95",
        "future_vol_5d", "future_vol_21d",
    ]

    df = df[[c for c in model_cols if c in df.columns]]
    df = df.dropna().reset_index(drop=True)

    out = PROCESSED / "modeling_daily.csv"
    df.to_csv(out, index=False)

if __name__ == "__main__":
    main()
