# Market-Implied vs News-Based Uncertainty: Predicting Sector ETF Volatility

This project compares market-implied uncertainty, broad news-based policy uncertainty, and equity-market-related news uncertainty for forecasting future sector ETF volatility.

## Data Sources

- Sector ETF prices: Yahoo Finance via `yfinance`
- VIX: FRED series `VIXCLS`
- Daily US Economic Policy Uncertainty: FRED series `USEPUINDXD`
- Equity Market-related Economic Uncertainty: FRED series `WLEMUINDXD`

## Main Question

Do news-based uncertainty measures add predictive value beyond recent realized volatility and VIX?

## Final Modeling Design

The project predicts future realized volatility for six sector ETFs using two horizons: 5 trading days and 21 trading days. Predictors are grouped into market-history features, VIX features, EPU features, and EMEU features. For each uncertainty measure, the final feature set includes level, 5-day change, 21-day average, 252-day z-score, and 95th percentile spike.

The same nested feature-set grid is fit for Linear Regression, Random Forest, and XGBoost. Models are evaluated on a held-out 2022-2026 test period using RMSE, MAE, and R².

## Reproduce

```bash
pip install -r requirements.txt
python scripts/download_data.py
python scripts/build_features.py
python scripts/fit_models.py
python scripts/prepare_results.py
python scripts/make_figures.py
python scripts/feature_importance.py
```
