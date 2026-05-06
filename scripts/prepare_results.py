from pathlib import Path
import pandas as pd

PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)


def add_overall_baseline_comparison(overall: pd.DataFrame) -> pd.DataFrame:
    baseline = overall[overall["feature_group"] == "baseline"][
        ["target", "model_class", "rmse", "mae", "r2"]
    ].rename(
        columns={
            "rmse": "baseline_rmse",
            "mae": "baseline_mae",
            "r2": "baseline_r2",
        }
    )

    comparison = overall.merge(
        baseline,
        on=["target", "model_class"],
        how="left",
    )

    comparison["rmse_change_vs_model_baseline"] = (
        comparison["rmse"] - comparison["baseline_rmse"]
    )
    comparison["rmse_pct_change_vs_model_baseline"] = (
        comparison["rmse_change_vs_model_baseline"]
        / comparison["baseline_rmse"]
        * 100
    )
    comparison["improves_model_baseline"] = (
        comparison["rmse"] < comparison["baseline_rmse"]
    )

    return comparison


def add_ticker_baseline_comparison(ticker: pd.DataFrame) -> pd.DataFrame:
    baseline = ticker[ticker["feature_group"] == "baseline"][
        ["target", "model_class", "ticker", "rmse", "mae", "r2"]
    ].rename(
        columns={
            "rmse": "baseline_rmse",
            "mae": "baseline_mae",
            "r2": "baseline_r2",
        }
    )

    comparison = ticker.merge(
        baseline,
        on=["target", "model_class", "ticker"],
        how="left",
    )

    comparison["rmse_change_vs_model_baseline"] = (
        comparison["rmse"] - comparison["baseline_rmse"]
    )
    comparison["rmse_pct_change_vs_model_baseline"] = (
        comparison["rmse_change_vs_model_baseline"]
        / comparison["baseline_rmse"]
        * 100
    )
    comparison["improves_model_baseline"] = (
        comparison["rmse"] < comparison["baseline_rmse"]
    )

    return comparison


def make_model_class_comparison(overall: pd.DataFrame) -> pd.DataFrame:
    best = (
        overall.sort_values(["target", "feature_group", "rmse"])
        .groupby(["target", "feature_group"], as_index=False)
        .first()
    )

    return best.rename(
        columns={
            "model_class": "best_model_class",
            "rmse": "best_rmse",
            "mae": "best_mae",
            "r2": "best_r2",
        }
    )[
        [
            "target",
            "feature_group",
            "best_model_class",
            "best_rmse",
            "best_mae",
            "best_r2",
        ]
    ]


def make_incremental_news_over_vix(overall: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        (
            "EPU after VIX",
            "baseline_plus_vix",
            "baseline_plus_vix_epu",
        ),
        (
            "EMEU after VIX",
            "baseline_plus_vix",
            "baseline_plus_vix_emeu",
        ),
        (
            "Full uncertainty after VIX",
            "baseline_plus_vix",
            "full_uncertainty",
        ),
    ]

    rows = []

    for target in sorted(overall["target"].unique()):
        for model_class in sorted(overall["model_class"].unique()):
            subset = overall[
                (overall["target"] == target)
                & (overall["model_class"] == model_class)
            ]

            for label, reference_group, candidate_group in comparisons:
                reference = subset[subset["feature_group"] == reference_group]
                candidate = subset[subset["feature_group"] == candidate_group]

                if reference.empty or candidate.empty:
                    continue

                reference_rmse = reference.iloc[0]["rmse"]
                candidate_rmse = candidate.iloc[0]["rmse"]
                rmse_change = candidate_rmse - reference_rmse

                rows.append(
                    {
                        "target": target,
                        "model_class": model_class,
                        "comparison": label,
                        "reference_feature_group": reference_group,
                        "candidate_feature_group": candidate_group,
                        "reference_rmse": reference_rmse,
                        "candidate_rmse": candidate_rmse,
                        "rmse_change": rmse_change,
                        "rmse_pct_change": rmse_change / reference_rmse * 100,
                        "improves_reference": candidate_rmse < reference_rmse,
                    }
                )

    return pd.DataFrame(rows).sort_values(
        ["target", "model_class", "comparison"]
    )


def main() -> None:
    overall = pd.read_csv(PROCESSED / "model_results.csv")
    ticker = pd.read_csv(PROCESSED / "model_results_by_ticker.csv")

    overall_comparison = add_overall_baseline_comparison(overall)
    ticker_comparison = add_ticker_baseline_comparison(ticker)

    ticker_comparison = ticker_comparison.sort_values(
        ["target", "ticker", "model_class", "rmse"]
    )

    model_class_comparison = make_model_class_comparison(overall)
    model_class_comparison.to_csv(
        PROCESSED / "model_class_comparison.csv",
        index=False,
    )

    feature_family_comparison = overall_comparison.sort_values(
        ["target", "model_class", "rmse"]
    )
    feature_family_comparison.to_csv(
        PROCESSED / "feature_family_comparison.csv",
        index=False,
    )

    incremental = make_incremental_news_over_vix(overall)
    incremental.to_csv(
        PROCESSED / "incremental_news_over_vix.csv",
        index=False,
    )

    best = (
        ticker_comparison.sort_values(["target", "ticker", "rmse"])
        .groupby(["target", "ticker"], as_index=False)
        .first()
    )

    best = best.rename(
        columns={"baseline_rmse": "baseline_rmse_for_same_model_class"}
    )

    best_cols = [
        "target",
        "ticker",
        "model_class",
        "feature_group",
        "model",
        "rmse",
        "mae",
        "r2",
        "baseline_rmse_for_same_model_class",
        "rmse_pct_change_vs_model_baseline",
    ]
    best = best[best_cols].sort_values(["target", "ticker"])
    best.to_csv(PROCESSED / "best_models_by_ticker.csv", index=False)

if __name__ == "__main__":
    main()
