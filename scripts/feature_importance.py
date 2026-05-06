from pathlib import Path
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROCESSED = Path("data/processed")
TARGETS = ["future_vol_5d", "future_vol_21d"]
TRAIN_END = "2021-12-31"

FEATURE_FAMILIES = {
    "Market history": [
        "abs_return_1d",
        "past_vol_5d",
        "past_vol_21d",
        "past_return_5d",
        "past_return_21d",
        "volume_change_1d",
        "volume_change_5d",
    ],
    "VIX": [
        "vix",
        "vix_change_5d",
        "vix_avg_21d",
        "vix_z_252d",
        "vix_spike_95",
    ],
    "EPU": [
        "epu",
        "epu_change_5d",
        "epu_avg_21d",
        "epu_z_252d",
        "epu_spike_95",
    ],
    "EMEU": [
        "emeu",
        "emeu_change_5d",
        "emeu_avg_21d",
        "emeu_z_252d",
        "emeu_spike_95",
    ],
}

FULL_FEATURES = (
    FEATURE_FAMILIES["Market history"]
    + FEATURE_FAMILIES["VIX"]
    + FEATURE_FAMILIES["EPU"]
    + FEATURE_FAMILIES["EMEU"]
    + ["ticker"]
)


def get_pretest_data(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["date"] <= TRAIN_END].copy()

def make_preprocessor(features: list[str]) -> ColumnTransformer:
    categorical = [col for col in features if col == "ticker"]
    numeric = [col for col in features if col not in categorical]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )


def clean_feature_name(name: str) -> str:
    return name.split("__", 1)[-1]


def assign_family(feature: str) -> str:
    if feature.startswith("ticker_"):
        return "Ticker indicator"

    for family, features in FEATURE_FAMILIES.items():
        if feature in features:
            return family
        
    return "Other"

def fit_random_forest_importance(
    data: pd.DataFrame,
    target: str,
    features: list[str],
) -> pd.DataFrame:
    model = Pipeline(
        steps=[
            ("preprocess", make_preprocessor(features)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=400,
                    max_depth=8,
                    min_samples_leaf=10,
                    random_state=370,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(data[features], data[target])

    feature_names = [
        clean_feature_name(name)
        for name in model.named_steps["preprocess"].get_feature_names_out()
    ]

    importance = model.named_steps["model"].feature_importances_

    result = pd.DataFrame(
        {
            "target": target,
            "feature": feature_names,
            "importance": importance,
        }
    )
    result["family"] = result["feature"].apply(assign_family)

    return result


def main() -> None:
    df = pd.read_csv(PROCESSED / "modeling_daily.csv", parse_dates=["date"])
    pretest = get_pretest_data(df)

    features = [col for col in FULL_FEATURES if col in df.columns]

    importance_df = pd.concat(
        [
            fit_random_forest_importance(pretest, target, features)
            for target in TARGETS
        ],
        ignore_index=True,
    )

    family_df = (
        importance_df.groupby(["target", "family"], as_index=False)["importance"]
        .sum()
        .sort_values(["target", "importance"], ascending=[True, False])
    )

    importance_df.to_csv(
        PROCESSED / "random_forest_feature_importance.csv",
        index=False,
    )
    family_df.to_csv(
        PROCESSED / "random_forest_family_importance.csv",
        index=False,
    )

if __name__ == "__main__":
    main()
