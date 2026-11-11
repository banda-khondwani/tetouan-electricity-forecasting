# Notebook06: Cell-by-cell guide

Run Notebooks04 and05 first using the project `.venv`.

## Cell 1 — Markdown

```markdown
# Notebook 06: Forecasting Models Against Strong Baselines
**Project F1: Electricity Consumption Analysis and Forecasting in Morocco**

## Objective and fixed evaluation contract
Issue one batch at **23:50 on the previous day** for all 144 next-day readings. Leads vary from 10 minutes to 24 hours. No target-day observations update that batch. Reporting and fitting delay are assumed negligible.

Training targets are January 9–August 31; validation targets September 1–October 31. We score **8,784 validation rows**, exactly as Notebook 05. Final test errors are not calculated. Earlier full-file descriptive exploration remains disclosed.

MAE is the declared selection criterion. RMSE, MSE, MAPE, WAPE and bias remain visible. MSE and RMSE have identical rankings on identical observations; RMSE retains original demand units.

We compare Ridge regression and histogram gradient boosting, each with and without historical weather. These four predeclared candidates provide a bounded first model comparison, not an exhaustive search. SARIMAX was a possible workflow option, not a required component; it is not fitted here.

Every model predicts a **correction to the previous-day daily profile**, rather than rebuilding the demand level from scratch. That anchor is available for all 144 targets at issue time. The Notebook 05 per-zone selected baseline remains a separate eligible benchmark and fallback.
```

## Cell 2 — Markdown

```markdown
## 1. Setup
Use the project `.venv` and run Notebook 04 then05 first. All tables and figures are saved separately. Source units remain unverified; charts use source units.
```

## Cell 3 — Code

```python
import sys, json, hashlib, importlib, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from IPython.display import display
p=Path.cwd().resolve()
ROOT=next(x for x in [p,*p.parents] if (x/'src/config.py').is_file())
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import src.config as cfg
importlib.reload(cfg)
from src.config import PROCESSED_DIR,FIGURES_DIR,TABLES_DIR,MODELS_DIR,ZONES
FIG=FIGURES_DIR/'06_forecasting_models';TAB=TABLES_DIR/'06_forecasting_models';MOD=MODELS_DIR/'06_daily_batch'
for folder in [FIG,TAB,MOD]:folder.mkdir(parents=True,exist_ok=True)
SERIES=[*ZONES,'total']
sns.set_theme(style='whitegrid')
plt.rcParams.update({'figure.dpi':110,'svg.fonttype':'none'})
def table(x,name):x.to_csv(TAB/f'{name}.csv',encoding='utf-8')
def figure(fig,name):
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(FIG/f'{name}.{ext}',dpi=200,bbox_inches='tight')
    plt.show();plt.close(fig)
print('Python:',sys.executable)
```

## Cell 4 — Markdown

```markdown
## 2. Verify Inputs and Freeze Candidate Features
The manifest's lists distinguish predictors from outcomes and timestamps. File hashes prevent accidental use of stale features or baseline selections. Scaling is learned inside the Ridge training pipeline only.

The load/calendar candidates are identical within each weather ablation pair. Optional weather uses only observations at or before the daily issue. We use no future observed weather, random split, fitted full-file imputation, or validation-driven early stopping.
```

## Cell 5 — Code

```python
mp=PROCESSED_DIR/'tetuan_features_daily_batch_manifest.json'
manifest=json.loads(mp.read_text(encoding='utf-8'))
assert manifest['design_id']=='daily_batch_2350'
feature_path=PROCESSED_DIR/manifest['feature_file']
assert hashlib.sha256(feature_path.read_bytes()).hexdigest()==manifest['feature_sha256']
assert hashlib.sha256((PROCESSED_DIR/manifest['source_file']).read_bytes()).hexdigest()==manifest['source_sha256']
contract=json.loads((MODELS_DIR/'05_baselines_daily_batch/baseline_selection_daily_batch.json').read_text(encoding='utf-8'))
assert contract['feature_manifest_sha256']==hashlib.sha256(mp.read_bytes()).hexdigest()
panel=pd.read_parquet(feature_path)
train=panel.loc[panel.split.eq('train')].copy()
val=panel.loc[panel.split.eq('validation')].copy()
assert len(train)==33840 and len(val)==8784
assert train.target_time.max()<=val.origin_time.min()
assert val.groupby(val.target_time.dt.normalize()).origin_time.nunique().eq(1).all()
groups=manifest['feature_groups']
base_features=groups['calendar']+groups['load_history']
weather_features=base_features+groups['weather_history']
assert len(base_features)==57 and len(weather_features)==72
assert not set(weather_features).intersection(manifest['target_columns'])
assert np.isfinite(panel[weather_features].to_numpy()).all()
actual=val.set_index('target_time')[[f'y_{z}' for z in ZONES]].rename(columns={f'y_{z}':z for z in ZONES})
actual['total']=actual[ZONES].sum(axis=1)
old=pd.read_parquet(PROCESSED_DIR/'baseline_validation_daily_batch.parquet')
baseline=old.loc[old.baseline.eq('selected_per_zone')].pivot(index='target_time',columns='series',values='prediction').reindex(index=actual.index,columns=SERIES)
assert baseline.notna().all().all()
baseline_actual=old.loc[old.baseline.eq('selected_per_zone')].pivot(index='target_time',columns='series',values='actual').reindex(index=actual.index,columns=SERIES)
assert np.allclose(baseline_actual,actual)
print('Training:',len(train),'Validation:',len(val),'Features without/with weather:',len(base_features),len(weather_features))
```

## Cell 6 — Markdown

```markdown
## 3. Declare Model Settings Before Comparison
Ridge uses training-only standardisation and alpha 10. Boosting uses 150 iterations, learning rate 0.05,15 leaves, minimum50 samples per leaf and L2 regularisation1. Early stopping is disabled to avoid a random internal validation split. Random seed 42 is fixed.

All zones use the same four candidate configurations. We do not repeatedly change settings until validation happens to look favourable. The correction target is actual minus previous-day demand. Negative final forecasts are clipped at zero using the predeclared nonnegative-demand constraint.
```

## Cell 7 — Code

```python
candidates={
 'ridge_no_weather':('ridge',base_features),
 'ridge_with_weather':('ridge',weather_features),
 'boost_no_weather':('boost',base_features),
 'boost_with_weather':('boost',weather_features),
}
def new_model(kind):
    if kind=='ridge':return make_pipeline(StandardScaler(),Ridge(alpha=10.0))
    return HistGradientBoostingRegressor(max_iter=150,learning_rate=.05,max_leaf_nodes=15,
        min_samples_leaf=50,l2_regularization=1.0,early_stopping=False,random_state=42)
predictions={'baseline':baseline}
fitted={};fit_rows=[]
for name,(kind,columns) in candidates.items():
    forecast=pd.DataFrame(index=actual.index)
    for zone in ZONES:
        anchor=train[f'{zone}_target_lag_1d'].to_numpy()
        residual=train[f'y_{zone}'].to_numpy()-anchor
        model=new_model(kind)
        start=time.perf_counter()
        with threadpool_limits(limits=2):
            model.fit(train[columns],residual)
            correction=model.predict(val[columns])
        forecast[zone]=np.maximum(0,val[f'{zone}_target_lag_1d'].to_numpy()+correction)
        fitted[(name,zone)]=model
        fit_rows.append({'candidate':name,'zone':zone,'features':len(columns),'fit_and_predict_seconds':time.perf_counter()-start})
        print('Fitted',name,zone,flush=True)
    forecast['total']=forecast[ZONES].sum(axis=1)
    predictions[name]=forecast
fit_report=pd.DataFrame(fit_rows)
table(fit_report,'01_fit_report')
display(fit_report.round(2))
```

## Cell 8 — Markdown

```markdown
### Interpretation: What Was Fitted
Twelve training-only estimators were fitted: four configurations for each of three zones. Validation outcomes were used for scoring, not to fit these estimators or their scalers. Runtime is environment-dependent.

The boosted models are nonlinear and can represent interactions; Ridge is a regularised linear correction. Greater complexity does not guarantee improvement over the baseline.
```

## Cell 9 — Markdown

```markdown
## 4. Compare All Metrics on Identical Validation Targets
Error is prediction minus actual, so positive bias means overprediction. MAPE averages per-reading percentages; WAPE divides summed absolute errors by summed actual values. All actual targets are positive.
```

## Cell 10 — Code

```python
def metrics(y,p):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
    assert y.ndim==p.ndim==1 and y.shape==p.shape and (y>0).all()
    assert np.isfinite(y).all() and np.isfinite(p).all()
    e=p-y;mse=np.mean(e**2)
    return {'MAE':np.abs(e).mean(),'MSE':mse,'RMSE':np.sqrt(mse),
            'MAPE_pct':np.mean(np.abs(e)/y)*100,'WAPE_pct':np.abs(e).sum()/y.sum()*100,'bias':e.mean()}
toy=metrics([100,200],[110,180]);assert np.isclose(toy['MAE'],15) and np.isclose(toy['MSE'],250)
rows=[]
for name,forecast in predictions.items():
    pd.testing.assert_index_equal(forecast.index,actual.index)
    assert np.allclose(forecast[ZONES].sum(axis=1),forecast.total)
    for zone in SERIES:rows.append({'candidate':name,'series':zone,**metrics(actual[zone],forecast[zone])})
scores=pd.DataFrame(rows)
reference=scores.loc[scores.candidate.eq('baseline')].set_index('series')
scores['MAE_improvement_pct']=[100*(1-r.MAE/reference.loc[r.series,'MAE']) for r in scores.itertuples()]
scores['RMSE_change_pct']=[100*(r.RMSE/reference.loc[r.series,'RMSE']-1) for r in scores.itertuples()]
table(scores,'02_validation_scores')
for zone in SERIES:
    print(zone);display(scores.loc[scores.series.eq(zone)].set_index('candidate').round(3))
fig,axes=plt.subplots(2,2,figsize=(13,8))
for ax,zone in zip(axes.flat,SERIES):
    d=scores.loc[scores.series.eq(zone)]
    ax.bar(d.candidate,d.MAE,color=['#777777','#0072B2','#56B4E9','#D55E00','#E69F00'])
    ax.set(title=zone,ylabel='Validation MAE (source units)');ax.tick_params(axis='x',rotation=30)
figure(fig,'01_mae_comparison')
```

## Cell 11 — Markdown

```markdown
### Interpretation: Actual Model Results
Zone 1's weather-enabled boosting model reduces MAE from **1,304.42 to1,234.16** (about **5.39%**) and RMSE from **1,829.10 to1,662.08**. Zone 3's weather-enabled Ridge reduces MAE from **711.30 to677.60** (about **4.74%**) and RMSE from **1,107.40 to951.69**.

**Zone 2 retains the baseline.** Every advanced candidate has a higher validation MAE there. This is a substantive result, not a failed notebook: the models must earn selection by beating an already strong recent-history rule.

Training uses squared-error objectives even though MAE determines validation selection. Their objectives and selection metric need not give identical rankings. We have not retuned these settings after seeing this comparison.
```

## Cell 12 — Markdown

```markdown
## 5. Weather Ablation and MAE/RMSE Trade-offs
Compare with-weather and without-weather versions of the same estimator. Positive MAE improvement means weather reduced validation MAE; a negative value means it worsened it. These are one-period validation comparisons, not causal weather effects or proof that a gain will generalise.

A lower MAE accompanied by higher RMSE can indicate larger occasional misses. We report both rather than silently changing the declared primary metric.
```

## Cell 13 — Code

```python
ablation=[]
for kind in ['ridge','boost']:
    for zone in ZONES:
        a=scores.loc[scores.candidate.eq(f'{kind}_no_weather')&scores.series.eq(zone)].iloc[0]
        b=scores.loc[scores.candidate.eq(f'{kind}_with_weather')&scores.series.eq(zone)].iloc[0]
        ablation.append({'model':kind,'zone':zone,'weather_MAE_improvement_pct':100*(1-b.MAE/a.MAE),
                         'weather_RMSE_change_pct':100*(b.RMSE/a.RMSE-1)})
ablation=pd.DataFrame(ablation);table(ablation,'03_weather_ablation');display(ablation.round(3))
fig,axes=plt.subplots(1,2,figsize=(13,5))
ablation.pivot(index='zone',columns='model',values='weather_MAE_improvement_pct').plot.bar(ax=axes[0],rot=0)
axes[0].axhline(0,color='black',linewidth=.8);axes[0].set(title='Weather ablation',ylabel='MAE improvement (%)')
for name in predictions:
    d=scores.loc[scores.candidate.eq(name)&scores.series.ne('total')]
    axes[1].scatter(d.MAE_improvement_pct,d.RMSE_change_pct,label=name)
axes[1].axhline(0,color='grey',linestyle='--');axes[1].axvline(0,color='grey',linestyle='--')
axes[1].set(title='Each point is one zone/configuration',xlabel='MAE improvement vs baseline (%)',ylabel='RMSE change vs baseline (%)')
axes[1].legend(fontsize=8)
figure(fig,'02_weather_and_metric_tradeoffs')
```

## Cell 14 — Markdown

```markdown
### Interpretation: Historical Weather Helps Some Configurations
Adding historical weather reduces Ridge MAE by about **27.93%** in Zone 3 and boosting MAE by **5.61%** in Zone 1, relative to their otherwise identical no-weather versions. However, weather worsens boosting MAE in Zone 3 by about **3.94%**, and slightly worsens Ridge in Zone 2.

Weather therefore does not provide a universal gain. The successful changes are validation associations from this particular setup, not causal effects. The weather features remain known at the previous-day issue rather than coming from the target day.
```

## Cell 15 — Markdown

```markdown
## 6. Select Per Zone, Keeping the Baseline Eligible
Choose the lowest validation MAE per zone, including the original baseline. Exact ties follow the candidate order, with baseline first. A model that does not beat its baseline is not selected merely because it is more complex.

This is validation-based selection; the selected validation score is optimistic relative to a genuinely independent final assessment. We do not claim significance for small differences or optimise every subgroup. Selected zone predictions are summed for a coherent combined forecast; per-zone choices need not minimise every combined metric.
```

## Cell 16 — Code

```python
choices={}
for zone in ZONES:
    s=scores.loc[scores.series.eq(zone)].set_index('candidate').loc[list(predictions),'MAE']
    choices[zone]=s.idxmin()
selected=pd.DataFrame({z:predictions[choices[z]][z] for z in ZONES},index=actual.index)
selected['total']=selected[ZONES].sum(axis=1)
selection_scores=pd.DataFrame([{'series':z,'candidate':choices.get(z,'sum_selected_zones'),**metrics(actual[z],selected[z])} for z in SERIES])
table(selection_scores,'04_selected_validation_scores');display(selection_scores.round(3))
print('Selected configurations:',choices)
fig,axes=plt.subplots(3,1,figsize=(13,9),sharex=True)
window=actual.index<actual.index.min()+pd.Timedelta(days=7)
for ax,zone in zip(axes,ZONES):
    ax.plot(actual.index[window],actual.loc[window,zone],color='black',label='Actual')
    ax.plot(actual.index[window],baseline.loc[window,zone],alpha=.7,label='Notebook05 baseline')
    ax.plot(actual.index[window],selected.loc[window,zone],alpha=.8,label=choices[zone])
    ax.set(title=zone,ylabel='Demand (source units)');ax.legend(fontsize=8)
axes[-1].set_xlabel('First seven validation target days in2017')
figure(fig,'03_selected_first_week')
```

## Cell 17 — Markdown

```markdown
### Interpretation: Selected Coherent Forecast
Select **boosting with historical weather for Zone 1**, **the Notebook 05 baseline for Zone 2**, and **Ridge with historical weather for Zone 3**.

Combined validation MAE is **2,177.87 source units**, approximately **6.65% lower** than the baseline's2,333.08. RMSE falls from3,489.83 to **2,915.44**, and MAPE is **3.33%**. These are validation-selection results, not final test performance.

Combined bias is approximately **−654.36 source units**, indicating average underprediction. That matters for peak warnings: lower average error does not establish good peak recall. Zone 2's limitations and the overall negative bias must remain visible when assessing alerts.
```

## Cell 18 — Markdown

```markdown
## 7. Daily and Lead-Slot Error Diagnostics
Daily MAE averages 144 target readings. Each lead slot has 61 validation observations. Since the issue clock is fixed, lead length is coupled with target time of day; these curves do not isolate a causal lead-length effect.

The display includes all scored days, including difficult September 1. We do not delete high-error days or fit to these diagnostics.
```

## Cell 19 — Code

```python
daily_baseline=(baseline-actual).abs().resample('D').mean()
daily_selected=(selected-actual).abs().resample('D').mean()
table(daily_selected,'05_selected_daily_mae')
table(daily_selected.nlargest(5,'total'),'06_difficult_days')
lead=((selected-actual).abs()/actual*100).groupby(val.lead_steps.to_numpy()).mean()
table(lead,'07_selected_lead_mape')
fig,axes=plt.subplots(1,2,figsize=(13,5))
axes[0].plot(daily_baseline.index,daily_baseline.total,label='Baseline')
axes[0].plot(daily_selected.index,daily_selected.total,label='Selected')
axes[0].set(title='Daily combined MAE',ylabel='MAE (source units)',xlabel='Validation date');axes[0].legend()
for zone in SERIES:axes[1].plot(lead.index/6,lead[zone],label=zone)
axes[1].set(title='Selected MAPE by daily-batch lead',xlabel='Hours after issue',ylabel='MAPE (%)');axes[1].legend()
import matplotlib.dates as mdates
axes[0].xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
axes[1].set_xlim(0,24)
figure(fig,'04_daily_and_lead_errors')

error=selected-actual
fig,axes=plt.subplots(2,2,figsize=(12,7))
for ax,zone in zip(axes.flat,SERIES):
    ax.hist(error[zone],bins=50,color='#0072B2',edgecolor='white')
    ax.axvline(0,color='black',linestyle='--');ax.axvline(error[zone].mean(),color='#D55E00',label=f'Bias {error[zone].mean():,.1f}')
    ax.set(title=zone,xlabel='Prediction minus actual (source units)',ylabel='Validation readings');ax.legend()
figure(fig,'05_selected_error_distributions')
```

## Cell 20 — Markdown

```markdown
## 8. Verify Persistence and Save Reproducible Results
Save training-fitted estimators and their explicit feature lists. Reload them and verify validation predictions are reproduced. These files are not final refitted deployment models and contain no test scoring.

Joblib artifacts should be loaded only from trusted sources. Later final evaluation must honour daily issues and fit only on labels available at the first issue. Estimators are frozen during this validation period; daily forecasts use updated eligible history without refitting parameters.
```

## Cell 21 — Code

```python
for (name,zone),model in fitted.items():
    path=MOD/f'{name}_{zone}.joblib'
    columns=candidates[name][1]
    bundle={'model':model,'feature_columns':columns,'anchor_column':f'{zone}_target_lag_1d',
            'fit_latest_label':str(train.target_time.max()),'design_id':manifest['design_id']}
    joblib.dump(bundle,path)
    loaded=joblib.load(path)
    with threadpool_limits(limits=2):correction=loaded['model'].predict(val[loaded['feature_columns']])
    reproduced=np.maximum(0,val[loaded['anchor_column']].to_numpy()+correction)
    assert np.allclose(reproduced,predictions[name][zone])
    if name.startswith('ridge'):
        scaler=model.named_steps['standardscaler']
        assert np.allclose(scaler.mean_,train[columns].mean().to_numpy())
        assert np.asarray(scaler.n_samples_seen_).min()==len(train)
parts=[]
for name,frame in {**predictions,'selected':selected}.items():
    for zone in SERIES:
        parts.append(pd.DataFrame({'target_time':actual.index,'origin_time':val.origin_time.to_numpy(),
            'lead_steps':val.lead_steps.to_numpy(),'candidate':name,'series':zone,
            'actual':actual[zone].to_numpy(),'prediction':frame[zone].to_numpy()}))
prediction_path=PROCESSED_DIR/'model_validation_daily_batch.parquet'
pd.concat(parts,ignore_index=True).to_parquet(prediction_path,index=False)
decision={'design_id':manifest['design_id'],'issue_clock':manifest['issue_clock'],'selection_metric':'validation MAE per zone',
    'selected_by_zone':choices,'fit_latest_label':str(train.target_time.max()),
    'feature_manifest_sha256':hashlib.sha256(mp.read_bytes()).hexdigest(),
    'baseline_selection':contract['selected_baseline_by_zone'],'test_scored':False,
    'selected_validation_scores':json.loads(selection_scores.to_json(orient='records')),
    'hyperparameter_policy':'Four predeclared configurations; no validation early stopping; no test tuning',
    'full_file_previously_explored':True}
(MOD/'model_selection_daily_batch.json').write_text(json.dumps(decision,indent=2,allow_nan=False),encoding='utf-8')
assert hashlib.sha256(feature_path.read_bytes()).hexdigest()==manifest['feature_sha256']
assert hashlib.sha256((PROCESSED_DIR/manifest['source_file']).read_bytes()).hexdigest()==manifest['source_sha256']
print('Reload checks and training-scaler checks passed. Inputs unchanged. Test errors not calculated.')
print('Models:',MOD)
```

## Cell 22 — Markdown

```markdown
## 9. Conclusions and Next Step
Review the score table and weather-ablation table before claiming an improvement. The eligible baseline is retained wherever advanced candidates fail to reduce validation MAE. MSE is now shown explicitly, but it provides the same ranking as RMSE.

Validation-based winners are provisional until an appropriately frozen final evaluation. Small improvements, worse RMSE, different monthly performance, and dependence between target readings must be discussed rather than hidden.

### Explain during the defence
- All next-day targets share one 23:50 issue; their leads differ.
- Models learn corrections to an available previous-day shape.
- Ridge standardisation uses training rows only; boosting uses no random internal early-stopping split.
- Weather ablation compares otherwise equivalent settings and does not establish causality.
- Candidate parameters were declared before this comparison; the validation window is still used for selection.
- Coherent combined forecasts can conceal zone-level errors.
- Historical error distributions are not calibrated forecast intervals.

### Next: Peak Monitoring and Final Evaluation
Before Notebook 07, agree on an operational peak definition and how to assess alerts. Any fitted peak threshold must use the designated training labels only. Freeze model choices and the final fitting/scoring procedure before calculating November–December test metrics. Do not silently reselect models on test results.
```
