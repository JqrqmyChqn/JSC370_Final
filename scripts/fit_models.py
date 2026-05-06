from itertools import product
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)
TARGETS = ["future_vol_5d", "future_vol_21d"]
PRETEST_END = "2021-12-31"

FEATURE_FAMILIES = {
    "B": [
        "abs_return_1d",
        "past_vol_5d",
        "past_vol_21d",
        "past_return_5d",
        "past_return_21d",
        "volume_change_1d",
        "volume_change_5d",
        "ticker",
    ],
    "V": [
        "vix",
        "vix_change_5d",
        "vix_avg_21d",
        "vix_z_252d",
        "vix_spike_95",
    ],
    "P": [
        "epu",
        "epu_change_5d",
        "epu_avg_21d",
        "epu_z_252d",
        "epu_spike_95",
    ],
    "E": [
        "emeu",
        "emeu_change_5d",
        "emeu_avg_21d",
        "emeu_z_252d",
        "emeu_spike_95",
    ],
}

FEATURE_GROUP_DEFINITIONS = {
    "baseline": ["B"],
    "baseline_plus_vix": ["B", "V"],
    "baseline_plus_epu": ["B", "P"],
    "baseline_plus_emeu": ["B", "E"],
    "baseline_plus_vix_epu": ["B", "V", "P"],
    "baseline_plus_vix_emeu": ["B", "V", "E"],
    "baseline_plus_epu_emeu": ["B", "P", "E"],
    "full_uncertainty": ["B", "V", "P", "E"],
}


def get_features(feature_group: str) -> list[str]:
    features = []

    for family in FEATURE_GROUP_DEFINITIONS[feature_group]:
        features.extend(FEATURE_FAMILIES[family])

    return features


def make_random_forest() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=10,
        random_state=370,
        n_jobs=-1,
    )


def make_xgboost() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=370,
        n_jobs=-1,
    )


MODEL_FACTORIES = {
    "Linear Regression": LinearRegression,
    "Random Forest": make_random_forest,
    "XGBoost": make_xgboost,
}


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pretest = df[df["date"] <= PRETEST_END].copy()
    test = df[df["date"] > PRETEST_END].copy()

    return pretest, test


def make_preprocessor(features: list[str]) -> ColumnTransformer:
    categorical = [col for col in features if col == "ticker"]
    numeric = [col for col in features if col not in categorical]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )


def evaluate(y_true: pd.Series, y_pred) -> tuple[float, float, float]:
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return rmse, mae, r2


def validate_features(df: pd.DataFrame, features: list[str]) -> None:
    missing = [feature for feature in features if feature not in df.columns]

    if missing:
        raise ValueError(f"Missing required model features: {missing}")


def fit_and_score(
    target: str,
    model_class: str,
    feature_group: str,
    pretest: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    features = get_features(feature_group)
    validate_features(pretest, features)

    model_name = f"{model_class} | {feature_group}"

    pipe = Pipeline(
        steps=[
            ("preprocess", make_preprocessor(features)),
            ("model", MODEL_FACTORIES[model_class]()),
        ]
    )

    pipe.fit(pretest[features], pretest[target])
    prediction = pipe.predict(test[features])

    rmse, mae, r2 = evaluate(test[target], prediction)

    result = {
        "target": target,
        "model_class": model_class,
        "model": model_name,
        "feature_group": feature_group,
        "n_features_raw": len(features),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }

    pred_df = test[["date", "ticker", target]].copy()
    pred_df = pred_df.rename(columns={target: "actual"})
    pred_df["target"] = target
    pred_df["model_class"] = model_class
    pred_df["model"] = model_name
    pred_df["feature_group"] = feature_group
    pred_df["prediction"] = prediction

    pred_df = pred_df[
        [
            "date",
            "ticker",
            "target",
            "model_class",
            "model",
            "feature_group",
            "actual",
            "prediction",
        ]
    ]

    return result, pred_df


def ticker_level_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["target", "model_class", "model", "feature_group", "ticker"]

    for keys, group in predictions.groupby(group_cols):
        target, model_class, model, feature_group, ticker = keys
        rmse, mae, r2 = evaluate(group["actual"], group["prediction"])

        rows.append(
            {
                "target": target,
                "model_class": model_class,
                "model": model,
                "feature_group": feature_group,
                "ticker": ticker,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["target", "ticker", "model_class", "rmse"]
    )


def save_model_catalog() -> None:
    rows = []

    for model_class, feature_group in product(MODEL_FACTORIES, FEATURE_GROUP_DEFINITIONS):
        features = get_features(feature_group)

        rows.append(
            {
                "model_class": model_class,
                "model": f"{model_class} | {feature_group}",
                "feature_group": feature_group,
                "feature_families": "+".join(FEATURE_GROUP_DEFINITIONS[feature_group]),
                "n_features_defined": len(features),
            }
        )

    pd.DataFrame(rows).to_csv(PROCESSED / "model_catalog.csv", index=False)


def save_split_summary(pretest: pd.DataFrame, test: pd.DataFrame) -> None:
    rows = []

    for name, split in [("pretest", pretest), ("test", test)]:
        rows.append(
            {
                "split": name,
                "min_date": split["date"].min(),
                "max_date": split["date"].max(),
                "n_rows": len(split),
                "n_tickers": split["ticker"].nunique(),
            }
        )

    pd.DataFrame(rows).to_csv(PROCESSED / "split_summary.csv", index=False)


def main() -> None:
    df = pd.read_csv(PROCESSED / "modeling_daily.csv", parse_dates=["date"])
    pretest, test = split_data(df)

    save_split_summary(pretest, test)
    save_model_catalog()

    print("Split summary:")
    print(pd.read_csv(PROCESSED / "split_summary.csv"))

    results = []
    predictions = []

    for target, model_class, feature_group in product(
        TARGETS,
        MODEL_FACTORIES,
        FEATURE_GROUP_DEFINITIONS,
    ):
        print(f"Fitting {model_class} | {feature_group} for {target}...")

        result, pred_df = fit_and_score(
            target=target,
            model_class=model_class,
            feature_group=feature_group,
            pretest=pretest,
            test=test,
        )

        results.append(result)
        predictions.append(pred_df)

    results_df = pd.DataFrame(results).sort_values(
        ["target", "model_class", "rmse"]
    )
    predictions_df = pd.concat(predictions, ignore_index=True)
    ticker_results = ticker_level_metrics(predictions_df)

    results_df.to_csv(PROCESSED / "model_results.csv", index=False)
    predictions_df.to_csv(PROCESSED / "predictions_test.csv", index=False)
    ticker_results.to_csv(PROCESSED / "model_results_by_ticker.csv", index=False)


if __name__ == "__main__":
    main()