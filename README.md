# Tetouan Electricity Demand: Analysis, Forecasting and Peak Warnings

MSc Business Analytics project exploring electricity demand across three distribution networks in Tetouan, Morocco. The project combines descriptive analysis, weather relationships, next-day forecasting and historical peak-warning evaluation.

## Current status

Notebooks 01–08 are implemented. Their saved outputs, explanatory cell guides, figures and result tables are included. The dashboard and several supporting modules are placeholders; a finished dashboard and final written report are still to come.

| Notebook | Purpose |
|---|---|
| [01 — Data cleaning](notebooks/01_data_cleaning.ipynb) | Load, standardise, audit and save the data |
| [02 — Demand patterns](notebooks/02_eda_demand_patterns.ipynb) | Descriptive statistics, daily/weekly/seasonal patterns and zone comparisons |
| [03 — Weather relationships](notebooks/03_weather_relationships.ipynb) | Weather associations and the influence of calendar patterns |
| [04 — Feature engineering](notebooks/04_feature_engineering.ipynb) | Construct predictors available at each daily forecast issue |
| [05 — Baseline models](notebooks/05_baseline_models.ipynb) | Compare persistence, previous-day, previous-week, seven-day mean and weekly-profile methods |
| [06 — Forecasting models](notebooks/06_forecasting_models.ipynb) | Ridge and histogram gradient boosting, with and without weather history |
| [07 — Peak warnings](notebooks/07_peak_detection_alerts.ipynb) | Frozen final evaluation and historical high-demand warning assessment |
| [08 — Extended comparison](notebooks/08_extended_model_comparison.ipynb) | Exploratory Random Forest and LightGBM comparison |

## Data and forecast design

The checked local file contains **52,416 observations**, every 10 minutes from **1 January to 30 December 2017**, with three demand series and five weather measurements. There are 364 complete days; 31 December is absent. The local count differs from the UCI catalogue's displayed instance count. See [data attribution](Dataset/README.md).

One forecast batch is issued at **23:50 on the previous day**, immediately after that reading. It contains all 144 next-day targets, with leads from 10 minutes to 24 hours. No target-day observations update that batch. The setup assumes negligible reporting and computation delay.

After the feature warm-up period, training targets run from 9 January through 31 August (33,840 rows), validation from 1 September through 31 October (8,784 rows), and test from 1 November through 30 December (8,640 rows). Model parameters stay fitted through August; eligible historical inputs advance each day. Future observed weather is not used.

Advanced models predict corrections to yesterday's same-time demand. Model selection uses validation MAE per zone, with RMSE, MSE, MAPE, WAPE and bias also reported. Combined forecasts are sums of zone forecasts. Earlier full-file descriptive analysis is disclosed in the notebooks.

## Results so far

The original advanced-model selection improved validation MAE but performed worse than the selected baseline on the later test period. The baseline uses previous-day demand for Zones 1–2 and the previous seven days' same-clock mean for Zone 3.

The extended comparison was proposed **after the original test results had been inspected**. Its settings were declared before fitting and its choices frozen using validation, but November–December results remain exploratory rather than independent confirmation.

Combined-demand November–December results:

| Strategy | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Selected baseline (05) | 2,159.19 | 3,234.92 | 3.46% |
| Original model selection (06) | 2,512.19 | 3,382.99 | 3.94% |
| Random Forest, validation-selected weather option | 2,952.61 | 4,144.00 | 4.76% |
| LightGBM, validation-selected weather option | 2,074.75 | 2,814.07 | 3.36% |
| Extended per-zone selection (08) | 2,354.44 | 3,192.95 | 3.72% |

LightGBM's combined MAE is 3.91% lower than the baseline's, **while its MAE is worse in every individual zone**. Overprediction in Zone 1 and underprediction in Zone 2 partly cancel. This is not evidence of an overall operational improvement. Random Forest improves Zone 2 MAE but is worse overall. The extended selection uses histogram gradient boosting for Zone 1, Random Forest for Zone 2 and Ridge for Zone 3, all with historical weather.

Peak thresholds are training-only 95th percentiles, not network capacity limits. Only Zone 2 has positive peaks in the test period. Its baseline precision and recall are both 91.60%; LightGBM's recall is 42.38%. Recall cannot be assessed for the other series in this period because there are no positive cases under these thresholds. The notebooks retain the false-alert counts and other limitations.

These findings describe the tested settings and periods, not universal algorithm rankings. Demand units remain labelled as source units where they have not been independently verified.

## Run locally

The saved run used Python 3.13.15. Exact package versions are recorded in [requirements.txt](requirements.txt), including LightGBM 4.7.0. Other platforms or package versions may need compatibility adjustments and may produce small numerical differences.

From the project root on Windows:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyter lab
```

In VS Code, select this environment as the notebook kernel, then run Notebooks 01–08 in order. The raw CSV is included with attribution. The notebooks regenerate the processed data and fitted models. Notebook 08 trains twelve estimators and can take several minutes on a laptop; subsequent runs reuse its saved models.

Forecast notebooks record file hashes and freeze experiment plans. When working in an existing local checkout, use consistent upstream outputs and package versions. If you intend to change an experiment, preserve its results and use a separately documented experiment directory instead of silently replacing its locked settings. A fresh clone has no saved model directory and should be run in order to generate its own linked artifacts.

## Files included

```text
Dataset/              Raw public dataset and attribution
notebooks/            Eight notebooks with saved outputs
src/                  Shared loading/configuration code; model/peak placeholders
reports/              Cell-by-cell Markdown guides and existing HTML exports
outputs/figures/      PNG and SVG plots
outputs/tables/       CSV result tables
outputs/processed/    Clean data, feature data, predictions and metadata
app/                  Dashboard placeholder
```

The local environment, temporary build/debug files, superseded rolling-forecast experiments and fitted model cache are excluded from Git. Saved metrics, plots and notebook outputs remain available for review. Open HTML exports in a browser after downloading them; GitHub displays HTML source rather than running it.

## Verification

Notebook 08 has 12 successfully executed code cells and 8 figures. Its independent audit reproduced 88 forecast-metric rows, 36 peak-warning rows, validation selections and sampled predictions from all twelve estimators. Input, model and prediction hashes matched. The repository preserves the earlier results alongside this exploratory extension.

## Attribution and reuse

The source dataset is by Abdulwahed Salam and Abdelaaziz El Hibaoui, distributed through UCI under CC BY 4.0; attribution and the licence link are in [Dataset/README.md](Dataset/README.md). This does not automatically license the group's original code or report text. No separate software licence has been selected yet.
