from pathlib import Path
import json
import pandas as pd
from plotly.offline import get_plotlyjs

PROCESSED = Path("data/processed")
INTERACTIVE = Path("figures/interactive")

INTERACTIVE.mkdir(parents=True, exist_ok=True)

FEATURE_GROUP_ORDER = [
    "baseline",
    "baseline_plus_vix",
    "baseline_plus_epu",
    "baseline_plus_emeu",
    "baseline_plus_vix_epu",
    "baseline_plus_vix_emeu",
    "baseline_plus_epu_emeu",
    "full_uncertainty",
]

FEATURE_GROUP_LABELS = {
    "baseline": "Baseline",
    "baseline_plus_vix": "Baseline + VIX",
    "baseline_plus_epu": "Baseline + EPU",
    "baseline_plus_emeu": "Baseline + EMEU",
    "baseline_plus_vix_epu": "Baseline + VIX + EPU",
    "baseline_plus_vix_emeu": "Baseline + VIX + EMEU",
    "baseline_plus_epu_emeu": "Baseline + EPU + EMEU",
    "full_uncertainty": "Full uncertainty",
}

FEATURE_GROUP_COLORS = {
    "baseline": "#777777",
    "baseline_plus_vix": "#1f77b4",
    "baseline_plus_epu": "#ff7f0e",
    "baseline_plus_emeu": "#2ca02c",
    "baseline_plus_vix_epu": "#9467bd",
    "baseline_plus_vix_emeu": "#d62728",
    "baseline_plus_epu_emeu": "#8c564b",
    "full_uncertainty": "#e377c2",
}

TARGET_LABELS = {
    "future_vol_5d": "5-day",
    "future_vol_21d": "21-day",
}

TICKER_ORDER = ["SPY", "XLE", "XLF", "XLI", "XLK", "XLV"]
MODEL_CLASSES = ["Linear Regression", "Random Forest", "XGBoost"]
PLOTLY_JS = get_plotlyjs()


def write_html(path: Path, body: str) -> None:
    path.write_text(
        f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script type="text/javascript">{PLOTLY_JS}</script>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px;
      color: #222;
    }}
    .controls {{
      display: flex;
      gap: 16px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }}
    label {{
      font-size: 14px;
      font-weight: 600;
    }}
    select {{
      margin-left: 6px;
      padding: 4px 8px;
    }}
  </style>
</head>
<body>
{body}
</body>
</html>
""",
        encoding="utf-8",
    )


def records_json(df: pd.DataFrame) -> str:
    return df.to_json(orient="records", date_format="iso")


def make_model_improvement_explorer(comparison: pd.DataFrame) -> None:
    data = comparison.copy()
    data["feature_group_label"] = data["feature_group"].map(FEATURE_GROUP_LABELS)
    data["target_label"] = data["target"].map(TARGET_LABELS)
    data["improves_label"] = data["improves_model_baseline"].map(
        lambda x: "Yes" if x else "No"
    )

    body = f"""
<h2>Do Uncertainty Signals Improve Volatility Forecasts Beyond Market History?</h2>
<div class="controls">
  <label>Horizon
    <select id="model-target"></select>
  </label>
  <label>Metric
    <select id="model-metric">
      <option value="rmse_pct_change_vs_model_baseline">RMSE % change</option>
      <option value="rmse">RMSE</option>
      <option value="mae">MAE</option>
      <option value="r2">R2</option>
    </select>
  </label>
</div>
<div id="model-improvement-chart" style="width:100%;height:680px;"></div>
<script>
const modelData = {records_json(data)};
const featureOrder = {json.dumps(FEATURE_GROUP_ORDER)};
const featureLabels = {json.dumps(FEATURE_GROUP_LABELS)};
const targetLabels = {json.dumps(TARGET_LABELS)};
const modelClasses = {json.dumps(MODEL_CLASSES)};
const metricLabels = {{
  rmse_pct_change_vs_model_baseline: "RMSE % Change vs Same Model-Class Baseline",
  rmse: "RMSE",
  mae: "MAE",
  r2: "R2"
}};

function populateModelControls() {{
  const targetSelect = document.getElementById("model-target");
  [...new Set(modelData.map(d => d.target))].sort().forEach(target => {{
    const option = document.createElement("option");
    option.value = target;
    option.textContent = targetLabels[target] || target;
    targetSelect.appendChild(option);
  }});
  targetSelect.value = "future_vol_5d";
}}

function renderModelImprovement() {{
  const target = document.getElementById("model-target").value;
  const metric = document.getElementById("model-metric").value;
  const rows = modelData.filter(d => d.target === target);
  const traces = modelClasses.map(modelClass => {{
    const byGroup = Object.fromEntries(
      rows.filter(d => d.model_class === modelClass).map(d => [d.feature_group, d])
    );
    const ordered = featureOrder.map(group => byGroup[group]).filter(Boolean);
    return {{
      type: "bar",
      name: modelClass,
      x: ordered.map(d => featureLabels[d.feature_group] || d.feature_group),
      y: ordered.map(d => d[metric]),
      customdata: ordered.map(d => [
        d.feature_group_label,
        d.model_class,
        d.rmse,
        d.baseline_rmse,
        d.rmse_pct_change_vs_model_baseline,
        d.mae,
        d.r2,
        d.improves_label
      ]),
      hovertemplate:
        "<b>%{{customdata[0]}}</b><br>" +
        "Model class: %{{customdata[1]}}<br>" +
        "RMSE: %{{customdata[2]:.6f}}<br>" +
        "Baseline RMSE: %{{customdata[3]:.6f}}<br>" +
        "RMSE % change: %{{customdata[4]:.2f}}%<br>" +
        "MAE: %{{customdata[5]:.6f}}<br>" +
        "R2: %{{customdata[6]:.3f}}<br>" +
        "Improves baseline: %{{customdata[7]}}<extra></extra>"
    }};
  }});
  const shapes = metric === "rmse_pct_change_vs_model_baseline" ? [{{
    type: "line",
    xref: "paper",
    x0: 0,
    x1: 1,
    y0: 0,
    y1: 0,
    line: {{color: "#333", width: 1}}
  }}] : [];
  Plotly.react("model-improvement-chart", traces, {{
    barmode: "group",
    title: "Do Uncertainty Signals Improve Volatility Forecasts Beyond Market History? " + (targetLabels[target] || target),
    xaxis: {{title: "Feature group"}},
    yaxis: {{
      title: {{text: metricLabels[metric], standoff: 16}},
      automargin: true,
      zeroline: false
    }},
    legend: {{title: {{text: "Model class"}}}},
    hovermode: "closest",
    shapes: shapes,
    margin: {{l: 95, r: 30, t: 70, b: 150}}
  }}, {{responsive: true}});
}}

populateModelControls();
document.getElementById("model-target").addEventListener("change", renderModelImprovement);
document.getElementById("model-metric").addEventListener("change", renderModelImprovement);
renderModelImprovement();
</script>
"""
    write_html(INTERACTIVE / "model_improvement_explorer.html", body)


def make_sector_improvement_explorer(best: pd.DataFrame) -> None:
    data = best.copy()
    data["baseline_rmse"] = data["baseline_rmse_for_same_model_class"]
    data["feature_group_label"] = data["feature_group"].map(FEATURE_GROUP_LABELS)
    data["target_label"] = data["target"].map(TARGET_LABELS)

    body = f"""
<h2>Best Model Improvement by Sector and Forecast Horizon</h2>
<div class="controls">
  <label>Horizon
    <select id="sector-target"></select>
  </label>
</div>
<div id="sector-improvement-chart" style="width:100%;height:620px;"></div>
<script>
const sectorData = {records_json(data)};
const sectorTargetLabels = {json.dumps(TARGET_LABELS)};
const sectorFeatureLabels = {json.dumps(FEATURE_GROUP_LABELS)};
const sectorFeatureColors = {json.dumps(FEATURE_GROUP_COLORS)};
const tickerOrder = {json.dumps(TICKER_ORDER)};
const sectorFeatureOrder = {json.dumps(FEATURE_GROUP_ORDER)};

function populateSectorControls() {{
  const targetSelect = document.getElementById("sector-target");
  [...new Set(sectorData.map(d => d.target))].sort().forEach(target => {{
    const option = document.createElement("option");
    option.value = target;
    option.textContent = sectorTargetLabels[target] || target;
    targetSelect.appendChild(option);
  }});
  targetSelect.value = "future_vol_5d";
}}

function renderSectorImprovement() {{
  const target = document.getElementById("sector-target").value;
  const rows = tickerOrder.map(ticker => {{
    return sectorData.find(d => d.target === target && d.ticker === ticker);
  }}).filter(Boolean);
  const traces = sectorFeatureOrder.map(group => {{
    const groupRows = rows.filter(d => d.feature_group === group);
    const hasRows = groupRows.length > 0;
    return {{
      type: "bar",
      name: sectorFeatureLabels[group] || group,
      x: hasRows ? groupRows.map(d => d.ticker) : [tickerOrder[0]],
      y: hasRows ? groupRows.map(d => d.rmse_pct_change_vs_model_baseline) : [null],
      marker: {{color: sectorFeatureColors[group] || "#555555"}},
      showlegend: true,
      hoverinfo: hasRows ? "all" : "skip",
      customdata: hasRows ? groupRows.map(d => [
        d.ticker,
        sectorTargetLabels[d.target] || d.target,
        d.model_class,
        d.feature_group_label,
        d.rmse,
        d.mae,
        d.r2,
        d.baseline_rmse,
        d.rmse_pct_change_vs_model_baseline,
        d.feature_group === "baseline" ? "Baseline selected; no uncertainty improvement." : "Uncertainty feature set selected."
      ]) : [[null, null, null, null, null, null, null, null, null, null]],
      hovertemplate:
        "<b>%{{customdata[0]}}</b><br>" +
        "Horizon: %{{customdata[1]}}<br>" +
        "Best model class: %{{customdata[2]}}<br>" +
        "Best feature group: %{{customdata[3]}}<br>" +
        "RMSE: %{{customdata[4]:.6f}}<br>" +
        "MAE: %{{customdata[5]:.6f}}<br>" +
        "R2: %{{customdata[6]:.3f}}<br>" +
        "Baseline RMSE: %{{customdata[7]:.6f}}<br>" +
        "RMSE % change: %{{customdata[8]:.2f}}%<br>" +
        "%{{customdata[9]}}<extra></extra>"
    }};
  }});
  const baselineRows = rows.filter(d => d.feature_group === "baseline");
  const baselineMarkers = {{
    type: "scatter",
    mode: "markers+text",
    name: "Baseline selected; no uncertainty improvement",
    x: baselineRows.map(d => d.ticker),
    y: baselineRows.map(d => 0),
    text: baselineRows.map(d => "0%"),
    textposition: "top center",
    marker: {{
      symbol: "diamond",
      size: 12,
      color: sectorFeatureColors.baseline,
      line: {{color: "#222222", width: 1}}
    }},
    customdata: baselineRows.map(d => [
      d.ticker,
      sectorTargetLabels[d.target] || d.target,
      d.model_class,
      d.feature_group_label,
      d.rmse,
      d.mae,
      d.r2,
      d.baseline_rmse,
      d.rmse_pct_change_vs_model_baseline,
      "Baseline selected; no uncertainty improvement."
    ]),
    hovertemplate:
      "<b>%{{customdata[0]}}</b><br>" +
      "Horizon: %{{customdata[1]}}<br>" +
      "Best model class: %{{customdata[2]}}<br>" +
      "Best feature group: %{{customdata[3]}}<br>" +
      "RMSE: %{{customdata[4]:.6f}}<br>" +
      "MAE: %{{customdata[5]:.6f}}<br>" +
      "R2: %{{customdata[6]:.3f}}<br>" +
      "Baseline RMSE: %{{customdata[7]:.6f}}<br>" +
      "RMSE % change: %{{customdata[8]:.2f}}%<br>" +
      "%{{customdata[9]}}<extra></extra>"
  }};
  const plotTraces = baselineRows.length > 0 ? traces.concat([baselineMarkers]) : traces;
  Plotly.react("sector-improvement-chart", plotTraces, {{
    barmode: "group",
    title: "Best Model Improvement by Sector and Forecast Horizon: " + (sectorTargetLabels[target] || target),
    xaxis: {{title: "Sector ETF", categoryorder: "array", categoryarray: tickerOrder}},
    yaxis: {{
      title: {{text: "RMSE % Change vs Same Model-Class Baseline", standoff: 16}},
      automargin: true
    }},
    showlegend: true,
    legend: {{title: {{text: "Best feature group"}}}},
    shapes: [{{
      type: "line",
      xref: "paper",
      x0: 0,
      x1: 1,
      y0: 0,
      y1: 0,
      line: {{color: "#333", width: 1}}
    }}],
    margin: {{l: 105, r: 30, t: 70, b: 70}}
  }}, {{responsive: true}});
}}

populateSectorControls();
document.getElementById("sector-target").addEventListener("change", renderSectorImprovement);
renderSectorImprovement();
</script>
"""
    write_html(INTERACTIVE / "sector_improvement_explorer.html", body)


def make_best_model_predictions(preds: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    best_keys = best[
        [
            "target",
            "ticker",
            "model_class",
            "model",
            "feature_group",
            "rmse",
            "rmse_pct_change_vs_model_baseline",
        ]
    ].copy()

    best_preds = preds.merge(
        best_keys,
        on=["target", "ticker", "model_class", "model", "feature_group"],
        how="inner",
    )
    best_preds = best_preds.rename(columns={"prediction": "predicted"})
    best_preds = best_preds[
        [
            "date",
            "ticker",
            "target",
            "actual",
            "predicted",
            "model_class",
            "feature_group",
            "rmse",
            "rmse_pct_change_vs_model_baseline",
        ]
    ].sort_values(["target", "ticker", "date"])
    best_preds.to_csv(PROCESSED / "best_model_predictions.csv", index=False)
    return best_preds


def make_prediction_explorer(best_preds: pd.DataFrame) -> None:
    data = best_preds.copy()
    data["feature_group_label"] = data["feature_group"].map(FEATURE_GROUP_LABELS)
    data["target_label"] = data["target"].map(TARGET_LABELS)

    body = f"""
<h2>Actual vs Predicted Future Volatility During the Test Period</h2>
<div class="controls">
  <label>ETF
    <select id="prediction-ticker"></select>
  </label>
  <label>Horizon
    <select id="prediction-target"></select>
  </label>
</div>
<div id="prediction-chart" style="width:100%;height:650px;"></div>
<script>
const predictionData = {records_json(data)};
const predictionTargetLabels = {json.dumps(TARGET_LABELS)};

function populatePredictionControls() {{
  const tickerSelect = document.getElementById("prediction-ticker");
  const targetSelect = document.getElementById("prediction-target");
  {json.dumps(TICKER_ORDER)}.forEach(ticker => {{
    const option = document.createElement("option");
    option.value = ticker;
    option.textContent = ticker;
    tickerSelect.appendChild(option);
  }});
  [...new Set(predictionData.map(d => d.target))].sort().forEach(target => {{
    const option = document.createElement("option");
    option.value = target;
    option.textContent = predictionTargetLabels[target] || target;
    targetSelect.appendChild(option);
  }});
  tickerSelect.value = "XLE";
  targetSelect.value = "future_vol_5d";
}}

function renderPredictionExplorer() {{
  const ticker = document.getElementById("prediction-ticker").value;
  const target = document.getElementById("prediction-target").value;
  const rows = predictionData.filter(d => d.ticker === ticker && d.target === target)
    .sort((a, b) => new Date(a.date) - new Date(b.date));
  const actual = {{
    type: "scatter",
    mode: "lines",
    name: "Actual",
    x: rows.map(d => d.date),
    y: rows.map(d => d.actual),
    line: {{width: 2}},
    customdata: rows.map(d => [d.model_class, d.feature_group_label, d.rmse, d.rmse_pct_change_vs_model_baseline]),
    hovertemplate:
      "Actual: %{{y:.6f}}<br>" +
      "Model class: %{{customdata[0]}}<br>" +
      "Feature group: %{{customdata[1]}}<br>" +
      "RMSE: %{{customdata[2]:.6f}}<br>" +
      "RMSE % change: %{{customdata[3]:.2f}}%<extra></extra>"
  }};
  const predicted = {{
    type: "scatter",
    mode: "lines",
    name: "Predicted",
    x: rows.map(d => d.date),
    y: rows.map(d => d.predicted),
    line: {{width: 2}},
    customdata: rows.map(d => [d.model_class, d.feature_group_label, d.rmse, d.rmse_pct_change_vs_model_baseline]),
    hovertemplate:
      "Predicted: %{{y:.6f}}<br>" +
      "Model class: %{{customdata[0]}}<br>" +
      "Feature group: %{{customdata[1]}}<br>" +
      "RMSE: %{{customdata[2]:.6f}}<br>" +
      "RMSE % change: %{{customdata[3]:.2f}}%<extra></extra>"
  }};
  Plotly.react("prediction-chart", [actual, predicted], {{
    title: "Actual vs Predicted Future Volatility During the Test Period: " + ticker + " " + (predictionTargetLabels[target] || target),
    xaxis: {{
      title: "Date",
      rangeslider: {{visible: true}}
    }},
    yaxis: {{
      title: {{text: "Future Realized Volatility", standoff: 16}},
      automargin: true
    }},
    hovermode: "x unified",
    legend: {{orientation: "h", y: 1.08}},
    shapes: [
      {{
        type: "rect",
        xref: "x",
        yref: "paper",
        x0: "2022-01-01",
        x1: "2022-12-31",
        y0: 0,
        y1: 1,
        fillcolor: "rgba(220, 90, 70, 0.08)",
        line: {{width: 0}}
      }},
      {{
        type: "rect",
        xref: "x",
        yref: "paper",
        x0: "2023-03-01",
        x1: "2023-05-31",
        y0: 0,
        y1: 1,
        fillcolor: "rgba(80, 120, 220, 0.08)",
        line: {{width: 0}}
      }}
    ],
    annotations: [
      {{
        x: "2022-06-30",
        y: 1.03,
        xref: "x",
        yref: "paper",
        text: "2022 inflation/rate shock",
        showarrow: false,
        font: {{size: 11}}
      }},
      {{
        x: "2023-04-15",
        y: 0.96,
        xref: "x",
        yref: "paper",
        text: "2023 banking stress",
        showarrow: false,
        font: {{size: 11}}
      }}
    ],
    margin: {{l: 95, r: 30, t: 90, b: 70}}
  }}, {{responsive: true}});
}}

populatePredictionControls();
document.getElementById("prediction-ticker").addEventListener("change", renderPredictionExplorer);
document.getElementById("prediction-target").addEventListener("change", renderPredictionExplorer);
renderPredictionExplorer();
</script>
"""
    write_html(INTERACTIVE / "prediction_explorer.html", body)


def main() -> None:
    comparison = pd.read_csv(PROCESSED / "feature_family_comparison.csv")
    best = pd.read_csv(PROCESSED / "best_models_by_ticker.csv")
    preds = pd.read_csv(PROCESSED / "predictions_test.csv", parse_dates=["date"])

    make_model_improvement_explorer(comparison)
    make_sector_improvement_explorer(best)
    best_preds = make_best_model_predictions(preds, best)
    make_prediction_explorer(best_preds)


if __name__ == "__main__":
    main()
