import argparse
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import freeze_support
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.interpolate import CubicSpline, interp1d, UnivariateSpline, PchipInterpolator
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.robust.norms import TukeyBiweight
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)
removed_records = []

def log_removed(lifenumber, breed, stage, reason):
    if isinstance(lifenumber, (list, tuple, set, np.ndarray, pd.Series)):
        ids = pd.unique(pd.Series(list(lifenumber)).dropna())
    else:
        ids = [lifenumber]
    for lid in ids:
        removed_records.append({'lifenumber': str(lid),
            'breed': str(breed) if breed is not None else str(lid)[:2], 'stage': stage, 'reason': reason})

def save_removed_log(output_dir, farm, current_time):
    removed_df = pd.DataFrame(removed_records).drop_duplicates()
    path = os.path.join(output_dir, f'{farm}_removed_individuals_{current_time}.csv')
    removed_df.to_csv(path, index=False)
    return path


# 1. Individual body-weight reconstruction
def process_individual_weight(lifenumber, weight_record):
    temp = weight_record[weight_record['lifenumber'] == lifenumber].dropna(subset=['weight']).sort_values('age').copy()
    audit = None
    if len(temp) < 3:
        return (None, 'Fewer than 3 valid weight records.', audit)
    temp['qrl'] = temp['age'] ** 2
    X = sm.add_constant(temp[['age', 'qrl']])
    y = temp['weight']
    try:
        model = sm.RLM(y, X, M=TukeyBiweight())
        results = model.fit()
        if not hasattr(results, 'weights'):
            return (None, 'Robust regression did not return weights.', audit)
        temp['w'] = results.weights
        audit = temp[['visit_time', 'age', 'weight', 'w']].copy()
        audit['lifenumber'] = lifenumber
        audit = audit.rename(columns={'weight': 'raw_weight', 'w': 'rlm_weight'})
        audit['record_pass_w025'] = audit['rlm_weight'] > 0.25
        audit['day_max_weight'] = audit.groupby('visit_time')['rlm_weight'].transform(
            lambda z: z[z > 0.25].max() if (z > 0.25).any() else np.nan)
        audit['day_pass_w09'] = audit['day_max_weight'] >= 0.9
        audit['used_for_daily_weight'] = audit['record_pass_w025'] & audit['day_pass_w09']
        temp.loc[temp['w'] <= 0.25, 'weight'] = np.nan
        clean_temp = temp.dropna(subset=['weight', 'w'])
        if clean_temp.empty:
            return (None, 'No valid records after outlier down-weighting.', audit)
        bad_dates = clean_temp.groupby('visit_time')['w'].max()
        bad_dates = bad_dates[bad_dates < 0.9].index
        clean_temp = clean_temp[~clean_temp['visit_time'].isin(bad_dates)]
        if clean_temp.empty:
            return (None, 'All dates removed after weight quality filtering.', audit)
        daily_weight = clean_temp.groupby('visit_time').agg(weighted_weight=('weight', lambda x: np.average(x,
            weights=clean_temp.loc[x.index, 'w']))).reset_index()
        daily_weight['visit_time'] = pd.to_datetime(daily_weight['visit_time'])
        daily_weight = daily_weight.sort_values('visit_time')
        if daily_weight.shape[0] < 3:
            return (None, 'Fewer than 3 daily weight points after aggregation.', audit)
        daily_weight = daily_weight.merge(clean_temp.groupby('visit_time')['age'].mean().reset_index(),
            on='visit_time', how='left')
        daily_weight['age2'] = daily_weight['age'] ** 2
        linreg = LinearRegression()
        linreg.fit(daily_weight[['age', 'age2']], daily_weight['weighted_weight'])
        r2 = linreg.score(daily_weight[['age', 'age2']], daily_weight['weighted_weight'])
        if r2 < 0.99:
            return (None, f'fit R2 below 0.99 (R2={r2:.4f}).', audit)
        ind_times = weight_record.loc[weight_record['lifenumber'] == lifenumber, 'visit_time']
        full_dates = pd.date_range(start=min(daily_weight['visit_time'].min(), ind_times.min()),
            end=max(daily_weight['visit_time'].max(), ind_times.max()), freq='D')
        df_full = pd.DataFrame({'visit_time': full_dates})
        df_full = df_full.merge(daily_weight, on='visit_time', how='left')
        df_full['day'] = (df_full['visit_time'] - df_full['visit_time'].min()).dt.days
        df_full['interpolated_weight'] = df_full['weighted_weight'].interpolate(method='linear', limit_area='inside')
        known = df_full[df_full['interpolated_weight'].notna()]
        n = 5
        mask_head = df_full['interpolated_weight'].isna() & (df_full['day'] < known['day'].min())
        if mask_head.any():
            head_points = known.head(n)
            if len(head_points) >= 3:
                cs_head = CubicSpline(head_points['day'].values, head_points['interpolated_weight'].values, bc_type='natural')
                vals = cs_head(df_full.loc[mask_head, 'day'].values)
                boundary_head = head_points['interpolated_weight'].iloc[0]
                vals = np.clip(vals, boundary_head - 5, boundary_head + 5)
            else:
                xh = head_points['day'].values
                yh = head_points['interpolated_weight'].values
                coef = np.polyfit(xh, yh, 1)
                vals = np.poly1d(coef)(df_full.loc[mask_head, 'day'].values)
            df_full.loc[mask_head, 'interpolated_weight'] = vals
        mask_tail = df_full['interpolated_weight'].isna() & (df_full['day'] > known['day'].max())
        if mask_tail.any():
            tail_points = known.tail(n)
            if len(tail_points) >= 3:
                cs_tail = CubicSpline(tail_points['day'].values, tail_points['interpolated_weight'].values, bc_type='natural')
                vals = cs_tail(df_full.loc[mask_tail, 'day'].values)
                boundary_tail = tail_points['interpolated_weight'].iloc[-1]
                vals = np.clip(vals, boundary_tail - 5, boundary_tail + 5)
            else:
                xt = tail_points['day'].values
                yt = tail_points['interpolated_weight'].values
                coef = np.polyfit(xt, yt, 1)
                vals = np.poly1d(coef)(df_full.loc[mask_tail, 'day'].values)
            df_full.loc[mask_tail, 'interpolated_weight'] = vals
        df_full = df_full.sort_values(by='visit_time')
        df_full['weight_diff'] = df_full['interpolated_weight'].diff().abs()
        df_full['source'] = 'missing'
        df_full.loc[df_full['weighted_weight'].notna(), 'source'] = 'observed'
        mid_mask = df_full['weighted_weight'].isna() & df_full['interpolated_weight'].notna()
        df_full.loc[mid_mask, 'source'] = 'interpolated'
        if mask_head.any():
            df_full.loc[mask_head, 'source'] = 'extrapolated'
        if mask_tail.any():
            df_full.loc[mask_tail, 'source'] = 'extrapolated'
        df_full['lifenumber'] = lifenumber
        return (df_full[['lifenumber', 'visit_time', 'day', 'weighted_weight', 'interpolated_weight', 'source']], None, audit)
    except Exception as e:
        return (None, f'Exception during weight processing: {e}', audit)


# 2. Daily feeding statistics
def get_daily_stats(feed: pd.DataFrame):
    feed = feed.copy()
    feed['visit_time'] = pd.to_datetime(feed['visit_time'])
    feed['visit_date'] = feed['visit_time'].dt.date
    is_intake_ok = feed['fi_anomaly_label'] == 1
    is_duration_ok = feed['dur_anomaly_label'] == 1
    is_rate_ok = feed['rate_anomaly_label'] == 1
    cond_normal = is_intake_ok & is_duration_ok & is_rate_ok
    cond_feed = is_intake_ok & ~is_duration_ok
    feed_proc = feed.assign(
        feed_normal=np.where(cond_normal, feed['feed_intake'], 0.0),
        anom_intake=np.where(feed['fi_anomaly_label'] == -1, 1, 0),
        anom_duration=np.where(feed['dur_anomaly_label'] == -1, 1, 0),
        anom_rate=np.where(feed['rate_anomaly_label'] == -1, 1, 0),
        duration_FIV=np.where(
            (feed['fi_anomaly_label'] == -1) | (feed['rate_anomaly_label'] == -1),
            feed['duration'], 0.0),
        feed_OTV=np.where(cond_feed, feed['feed_intake'], 0.0),
        FI_anom_feed=np.where(feed['fi_anomaly_label'] == -1, feed['feed_intake'], 0.0),
        dur_anom_duration=np.where(feed['dur_anomaly_label'] == -1, feed['duration'], 0.0),
        total_feed_item=feed['feed_intake'].fillna(0.0),
        total_duration_item=feed['duration'].fillna(0.0),
    )
    daily_stats = feed_proc.groupby(['lifenumber', 'visit_date'], as_index=False).agg(
        feed_normal=('feed_normal', 'sum'),
        intake_anom_ratio=('anom_intake', 'mean'),
        rate_anom_ratio=('anom_rate', 'mean'),
        duration_anom_ratio=('anom_duration', 'mean'),
        duration_FIV=('duration_FIV', 'sum'),
        feed_OTV=('feed_OTV', 'sum'),
        FI_anom_feed_sum=('FI_anom_feed', 'sum'),
        dur_anom_dur_sum=('dur_anom_duration', 'sum'),
        total_feed_sum=('total_feed_item', 'sum'),
        total_duration_sum=('total_duration_item', 'sum'),
        daily_weight=('weight', lambda x: x.iloc[0] if not x.isna().all() else np.nan),
        location=('location', lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan),
    ).rename(columns={'visit_date': 'visit_time'}).sort_values(['lifenumber', 'visit_time'])
    daily_total_feed = daily_stats['feed_normal'] + daily_stats['feed_OTV'] + daily_stats['FI_anom_feed_sum']
    daily_stats['FI_anom_feed_ratio'] = np.where(daily_total_feed > 0, daily_stats['FI_anom_feed_sum'] / daily_total_feed, 0)
    daily_stats['FI_anom_feed_sum_ratio'] = np.where(daily_stats['total_feed_sum'] > 0,
        daily_stats['FI_anom_feed_sum'] / daily_stats['total_feed_sum'], 0.0)
    daily_stats['dur_anom_dur_sum_ratio'] = np.where(daily_stats['total_duration_sum'] > 0,
        daily_stats['dur_anom_dur_sum'] / daily_stats['total_duration_sum'], 0.0)
    adg = daily_stats.groupby('lifenumber', as_index=False).agg(start_weight=('daily_weight', 'first'),
        end_weight=('daily_weight', 'last'), start_time=('visit_time', 'first'), end_time=('visit_time',
            'last'), n_days=('visit_time', 'nunique'))
    start_dt = pd.to_datetime(adg['start_time'])
    end_dt = pd.to_datetime(adg['end_time'])
    adg['ADG'] = ((adg['end_weight'] - adg['start_weight']) / (end_dt - start_dt).dt.days).where(adg['n_days'] > 1, np.nan)
    daily_stats = daily_stats.merge(adg[['lifenumber', 'ADG']], on='lifenumber', how='left')
    return daily_stats


# 3. ADFI interpolation
def fill_and_format_ADFI(final_weight, daily_stats_all, frac=0.4):
    final_weight = final_weight.copy()
    daily_stats_all = daily_stats_all.copy()
    final_weight['visit_time'] = pd.to_datetime(final_weight['visit_time'])
    daily_stats_all['visit_time'] = pd.to_datetime(daily_stats_all['visit_time'])
    valid_ids = daily_stats_all['lifenumber'].unique()
    daily_stats_all = daily_stats_all.copy()
    daily_stats_all['date'] = pd.to_datetime(daily_stats_all['visit_time']).dt.date
    final_weight['date'] = pd.to_datetime(final_weight['visit_time']).dt.date
    merged = final_weight.merge(daily_stats_all[['lifenumber', 'date', 'ADFI']], on=['lifenumber', 'date'], how='left')
    merged = merged[merged['lifenumber'].isin(valid_ids)].copy()
    merged['source'] = np.where(merged['ADFI'].notna(), 'observed', None)
    if 'day' not in merged.columns:
        raise KeyError("'day' column missing from final_weight; cannot interpolate ADFI.")
    results = []
    for pig_id, g in merged.groupby('lifenumber'):
        temp = g.sort_values('visit_time').copy()
        if temp['ADFI'].notna().sum() > 3:
            x = temp.loc[temp['ADFI'].notna(), 'day']
            y = temp.loc[temp['ADFI'].notna(), 'ADFI']
            fitted = lowess(y, x, frac=frac, return_sorted=True)
            loess_func = interp1d(fitted[:, 0], fitted[:, 1], kind='linear', bounds_error=False, fill_value='extrapolate')
            mask = temp['ADFI'].isna()
            temp.loc[mask, 'ADFI'] = loess_func(temp.loc[mask, 'day'])
            temp.loc[mask, 'source'] = 'loess'
        if temp['ADFI'].isna().any() and temp['ADFI'].notna().sum() >= 2:
            interp_func = interp1d(temp.loc[temp['ADFI'].notna(), 'day'], temp.loc[temp['ADFI'].notna(),
                'ADFI'], kind='linear', bounds_error=False, fill_value='extrapolate')
            mask = temp['ADFI'].isna()
            temp.loc[mask, 'ADFI'] = interp_func(temp.loc[mask, 'day'])
            temp.loc[mask, 'source'] = 'linear_interp'
        results.append(temp)
    filled = pd.concat(results, axis=0).reset_index(drop=True)
    filled = filled.merge(daily_stats_all[['lifenumber', 'visit_time', 'daily_weight']].drop_duplicates(),
        on=['lifenumber', 'visit_time'], how='left')
    filled = filled.rename(columns={'lifenumber': 'id', 'visit_time': 'time', 'interpolated_weight': 'weight'})
    filled['ADFI'] = filled['ADFI'].clip(lower=0)
    return filled[['id', 'location', 'time', 'weight', 'ADFI']]




# Directional anomaly indicators and deviation severity
def add_directional_features(feed):
    feed = feed.copy()
    for prefix, (val_col, label_col, ref_col) in {
        'fi': ('feed_intake', 'fi_anomaly_label', 'ind_fi_median'),
        'dur': ('duration', 'dur_anomaly_label', 'ind_dur_median'),
        'rate': ('rate', 'rate_anomaly_label', 'ind_rate_median'),
    }.items():
        val = pd.to_numeric(feed[val_col], errors='coerce')
        ref = pd.to_numeric(feed[ref_col], errors='coerce')
        abnormal = feed[label_col].eq(-1).fillna(False)
        usable = np.isfinite(val) & np.isfinite(ref) & (val > 0) & (ref > 0)
        low = abnormal & usable & (val < ref)
        high = abnormal & usable & (val > ref)
        unknown = abnormal & ~(low | high)
        for side, mask in (('low', low), ('high', high), ('unknown', unknown)):
            feed[f'{prefix}_{side}_flag'] = mask.astype('int8')
        for side, mask, numerator, denominator in (
            ('low', low, ref, val), ('high', high, val, ref)
        ):
            values = np.zeros(len(feed), dtype='float32')
            idx = np.flatnonzero(mask.to_numpy(dtype=bool))
            if len(idx):
                r = numerator.iloc[idx].to_numpy(dtype='float64') / denominator.iloc[idx].to_numpy(dtype='float64')
                values[idx] = np.log(r).astype('float32')
            feed[f'{prefix}_{side}_severity'] = values

    return feed


def audit_direction_features(feed, audit_dir):
    """Summarise FI/DUR/RATE anomaly counts by direction (low, high, unknown)."""
    rows = []
    for breed, g in feed.groupby('breed'):
        for metric in ('fi', 'dur', 'rate'):
            for side in ('low', 'high', 'unknown'):
                rows.append({'breed': breed, 'metric': metric, 'direction': side,
                    'n_events': int(g[f'{metric}_{side}_flag'].sum())})
    pd.DataFrame(rows).to_csv(os.path.join(audit_dir, 'anomaly_direction_counts.csv'), index=False)


def add_daily_directional_features(feed, daily_stats):
    feed = feed.copy()
    feed['visit_date'] = pd.to_datetime(feed['visit_time']).dt.date
    keys = ['lifenumber', 'visit_date']
    flag_cols = [f'{m}_{s}_flag' for m in ('fi', 'dur', 'rate') for s in ('low', 'high', 'unknown')]
    severity_cols = [f'{m}_{s}_severity' for m in ('fi', 'dur', 'rate') for s in ('low', 'high')]
    grouped = feed.groupby(keys, sort=False, observed=True)
    summ = grouped[flag_cols + severity_cols].sum()
    summ['n_feed_events'] = grouped.size()
    for metric in ('fi', 'dur', 'rate'):
        for side in ('low', 'high', 'unknown'):
            summ[f'{metric}_{side}_ratio'] = summ[f'{metric}_{side}_flag'] / summ['n_feed_events']
        for side in ('low', 'high'):
            summ[f'{metric}_{side}_severity_mean'] = summ[f'{metric}_{side}_severity'] / summ['n_feed_events']
    fi_ok = feed['fi_anomaly_label'].eq(1).fillna(False)
    dur_ok = feed['dur_anomaly_label'].eq(1).fillna(False)
    rate_ok = feed['rate_anomaly_label'].eq(1).fillna(False)
    rate_only = fi_ok & dur_ok & ~rate_ok
    feed['feed_rate_only'] = feed['feed_intake'].where(rate_only, 0.0).fillna(0.0)
    feed['n_rate_only_events'] = rate_only.astype('int8')
    more = grouped[['feed_rate_only', 'n_rate_only_events']].sum()
    summ = summ.join(more, how='left')
    summ = summ.reset_index().rename(columns={'visit_date': 'visit_time'})
    enriched = daily_stats.merge(summ, on=['lifenumber', 'visit_time'], how='left', validate='one_to_one')
    if len(enriched) != len(daily_stats):
        raise AssertionError('Daily aggregation changed number of pig-days')
    return enriched


def audit_daily_partition(daily_stats, audit_dir):
    # A fourth part captures RATE-anomalous visits with valid FI and DUR.
    three = daily_stats['feed_normal'] + daily_stats['feed_OTV'] + daily_stats['FI_anom_feed_sum']
    four = three + daily_stats['feed_rate_only']
    balance = four - daily_stats['total_feed_sum']
    # Invalid negative FI is kept in the original accounting; do not silently clip it.
    tol = 1e-7 * (1.0 + daily_stats['total_feed_sum'].abs())
    mismatch = balance.abs() > tol
    daily_stats.assign(original_three_parts=three, four_parts=four,
        original_unassigned_feed=daily_stats['total_feed_sum'] - three,
        four_part_balance=balance, balance_failed=mismatch)[[
            'lifenumber', 'visit_time', 'n_feed_events', 'total_feed_sum',
            'feed_normal', 'feed_OTV', 'FI_anom_feed_sum', 'feed_rate_only',
            'original_three_parts', 'original_unassigned_feed', 'four_parts',
            'four_part_balance', 'balance_failed'
        ]].to_csv(os.path.join(audit_dir, 'daily_feed_partition_audit.csv'), index=False)
    print('Feed partition check: original rate-only unassigned days:', int((daily_stats['feed_rate_only'] != 0).sum()),
        '; unmatched four-part days:', int(mismatch.sum()),
)
    if mismatch.any():
        raise AssertionError('Event/feed accounting does not conserve source event totals; see daily_feed_partition_audit.csv')


def fit_feed_adjustment_model(breed, data_b):
    """Fit the direction-and-severity linear mixed-effects model.

    Coefficients describe associations, not verified true measurement errors.
    """
    report = {'breed': breed, 'status': 'FAILED', 'reason': '',
        'n_days': len(data_b), 'n_pigs': data_b['lifenumber'].nunique(),
        'n_model_days': 0, 'n_model_pigs': 0, 'n_predictors': 0,
        'retained_predictors': '', 'vif_removed': '', 'converged': False}
    data = data_b.loc[data_b['feed_normal'] > 0].copy()
    if len(data) < 20 or data['lifenumber'].nunique() < 2:
        report['reason'] = 'Insufficient positive-normal-feed pig-days or pigs'
        return None, report
    # Two amount/duration terms, nine directional ratios and six severities.
    candidates = ['duration_FIV', 'feed_OTV'] + [
        f'{m}_{side}_ratio' for m in ('fi', 'dur', 'rate')
        for side in ('low', 'high', 'unknown')
    ] + [
        f'{m}_{side}_severity_mean' for m in ('fi', 'dur', 'rate')
        for side in ('low', 'high')
    ]
    base = [x for x in ('location', 'daily_weight', 'ADG') if x in data and data[x].nunique(dropna=True) > 1]
    nonconstant = [x for x in candidates if x in data and data[x].nunique(dropna=True) > 1]
    if not nonconstant:
        report['reason'] = 'No varying anomaly predictors'
        return None, report
    data = data.dropna(subset=base + nonconstant + ['feed_normal', 'lifenumber'])
    if len(data) < len(base + nonconstant) + 10 or data['lifenumber'].nunique() < 2:
        report['reason'] = 'Insufficient complete cases after removing predictor missingness'
        return None, report
    # Preserve all varying candidate predictors without VIF-based deletion.
    # A rank warning is audited; failure to fit is reported explicitly.
    remaining = nonconstant[:]
    removed = []
    xx = data[remaining].to_numpy(dtype=float)
    xx = np.column_stack([np.ones(len(xx)), xx])
    rank = int(np.linalg.matrix_rank(xx[:min(len(xx), 15000)]))
    report['design_rank_sample'] = rank
    report['design_cols'] = len(remaining) + 1
    if rank < len(remaining) + 1:
        report['reason'] = 'rank-deficient design, retained nested specification'
        print(f'  WARNING {breed}: rank={rank}/{len(remaining)+1}; MixedLM may fail')
    report['planned_anomaly_predictors'] = ';'.join(candidates)
    report['omitted_constant_predictors'] = ';'.join(x for x in candidates if x not in remaining)
    predictors = base + remaining
    report['n_model_days'] = len(data)
    report['n_model_pigs'] = int(data['lifenumber'].nunique())
    report['n_predictors'] = len(predictors)
    report['retained_predictors'] = ';'.join(predictors)
    report['vif_removed'] = ';'.join(removed)
    formula = 'feed_normal ~ ' + ' + '.join(predictors)
    try:
        # Fallbacks address singular variance boundaries common in small cohorts.
        # Do not silently replace mixed models with OLS.
        fit_errors = []
        model = None
        successful_optimizer = None
        for optimizer in ('lbfgs', 'powell', 'cg'):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=ConvergenceWarning)
                    warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
                    attempt = smf.mixedlm(formula, data=data, groups=data['lifenumber']).fit(
                        reml=True, method=optimizer, maxiter=250, disp=False)
                if not np.isfinite(attempt.fe_params.to_numpy(dtype=float)).all():
                    raise FloatingPointError('Non-finite fixed-effect estimates')
                model = attempt
                successful_optimizer = optimizer
                if attempt.converged:
                    break
            except (Exception,) as exc:
                fit_errors.append(f'{optimizer}:{type(exc).__name__}:{exc}')
        if model is None or not model.converged:
            raise RuntimeError('All optimizers failed or did not converge: ' + ';'.join(fit_errors))
        report['optimizer'] = successful_optimizer
        blue = model.fe_params.to_dict()
        data['ADJ'] = 0.0
        for col in remaining:
            if col in blue:
                data['ADJ'] -= data[col] * blue[col]
        data['ADFI'] = (data['feed_normal'] + data['ADJ']).clip(lower=0)
        report['status'] = 'OK'
        report['converged'] = bool(model.converged)
        report['reason'] = 'fit complete; direction/severity adjustment applied'
        report['coefficient_json'] = json.dumps({k: float(v) for k, v in blue.items()}, ensure_ascii=False)
        print(f'  {breed}: days={len(data):,}, pigs={data.lifenumber.nunique()}, predictors={len(predictors)}, converged={model.converged}')
        return data, report
    except Exception as exc:
        report['reason'] = f'MixedLM fit failed: {type(exc).__name__}: {exc}'
        print(f'  WARNING {breed}: {report["reason"]}')
        return None, report


# 5. End-to-end QC pipeline
def main(feed_path: str, farm: str, output_dir: str, smooth_factor: float=0.5, reference_only: bool=False):
    """Process EFS records with biologically informed quality control."""
    removed_records.clear()
    os.makedirs(output_dir, exist_ok=True)
    current_time = datetime.now().strftime('%Y%m%d')
    audit_dir = os.path.join(output_dir, 'audit')
    os.makedirs(audit_dir, exist_ok=True)
    with open(os.path.join(audit_dir, 'README.txt'), 'w', encoding='utf-8') as f:
        f.write('stage_flow.csv: sequential row and animal counts; record units change to pig-days after stage 07.\n')
        f.write('weight_rlm_record_weights.csv: each non-missing single-visit weight and RLM weight, w>0.25, day max w>=0.9.\n')
        f.write('Missing source body-weight records are counted in input flow but have no fitted RLM weight.\n')
        f.write('Anomalous feeding events are generally labeled rather than removed; only missing 7-day references cause row deletion.\n')
        f.write('Rate anomalies are evaluated only after FI and DUR pass, so denominators differ across metrics.\n')
        f.write('Final ADFI combines estimated/interpolated days; final row counts are not raw feeding-event counts.\n')

    stage_rows, stage_previous, breed_stage_rows = [], {}, []

    def record_stage(stage, df, id_col='lifenumber'):
        ids = set(df[id_col].dropna().astype(str).unique()) if id_col in df else set()
        prev = stage_previous.get('last')
        stage_rows.append({'stage': stage, 'n_records': len(df), 'n_pigs': len(ids),
            'records_removed_since_previous': np.nan if prev is None else prev[0] - len(df),
            'pigs_removed_since_previous': np.nan if prev is None else len(prev[1] - ids),
            'pigs_added_since_previous': np.nan if prev is None else len(ids - prev[1])})
        if prev is not None:
            lost = prev[1] - ids
            if lost: log_removed(lost, None, stage, 'Individual absent after this stage.')
        stage_previous['last'] = (len(df), ids)
        if id_col in df:
            counts = df.assign(_breed=df[id_col].astype(str).str[:2]).groupby('_breed')[id_col].agg(['size', 'nunique'])
            for breed, values in counts.iterrows():
                breed_stage_rows.append({'stage': stage, 'breed': breed,
                    'n_records': int(values['size']), 'n_pigs': int(values['nunique'])})


    def save_audit():
        pd.DataFrame(stage_rows).to_csv(os.path.join(audit_dir, 'stage_flow.csv'), index=False)
        pd.DataFrame(breed_stage_rows).to_csv(os.path.join(audit_dir, 'stage_flow_by_breed.csv'), index=False)

        pd.DataFrame(removed_records, columns=['lifenumber', 'breed', 'stage', 'reason']).drop_duplicates().to_csv(
            os.path.join(audit_dir, 'excluded_individuals_by_stage.csv'), index=False)
        save_removed_log(output_dir, farm, current_time)

    print('步骤1: 加载和初步清洗数据...')
    feed = pd.read_csv(feed_path, dtype={'location': str, 'lifenumber': 'string'})
    record_stage('01_raw_input', feed)

    if 'start_time' in feed.columns:
        feed['start_time'] = pd.to_datetime(feed['start_time'], errors='coerce')
        feed['end_time'] = pd.to_datetime(feed['end_time'], errors='coerce')
        feed['duration'] = (feed['end_time'] - feed['start_time']).dt.total_seconds()
        feed['visit_time'] = feed['start_time'].dt.floor('D')
    if 'start_feed' in feed.columns:
        feed['feed_intake'] = feed['start_feed'] - feed['end_feed']
    feed['visit_time'] = pd.to_datetime(feed['visit_time'], errors='coerce').dt.normalize()
    feed = feed[['lifenumber', 'location', 'visit_time', 'duration', 'weight', 'feed_intake']]
    feed['lifenumber'] = feed['lifenumber'].astype('string').str.strip()
    valid_id_mask = (feed['lifenumber'].notna() & ~feed['lifenumber'].str.lower().isin(['na', 'nan', 'none', ''])
                     & ~feed['lifenumber'].str.match(r'^0+$', na=False))
    pd.DataFrame({'reason': ['missing_or_zero_identifier'], 'n_records': [int((~valid_id_mask).sum())]}).to_csv(
        os.path.join(audit_dir, 'initial_invalid_identifiers.csv'), index=False)

    feed = feed.loc[valid_id_mask].copy()
    feed['breed'] = feed['lifenumber'].str[:2]
    record_stage('02_valid_identifiers', feed)
    print(f'初始数据加载完成，共 {len(feed)} 条记录。')
    feed_record = feed.groupby('breed').agg(animal=('lifenumber', 'nunique'), feed_record_number=('lifenumber',
        'size')).reset_index()
    print(f'Feed Record by Breed:\n{feed_record}')
    print('步骤2: 无效数据处理...')
    feed['visit_time'] = pd.to_datetime(feed['visit_time']).dt.normalize()
    min_times = feed.groupby('lifenumber')['visit_time'].transform('min')
    feed['age'] = (feed['visit_time'] - min_times).dt.days + 1
    g = feed.groupby('lifenumber')['age']
    summary = pd.DataFrame({'time_span': g.max(), 'n_obs_days': g.nunique()}).reset_index()
    summary['missing_days'] = summary['time_span'] - summary['n_obs_days']
    summary['missing_rate'] = summary['missing_days'] / summary['time_span']
    summary.to_csv(os.path.join(audit_dir, 'individual_day_coverage.csv'), index=False)
    if feed.empty:
        save_removed_log(output_dir, farm, current_time)
        raise ValueError('Error: 所有个体均未通过数据完整性校验。剩余数据为空。')
    print('步骤3: 体重数据处理...')
    ids_before_weight_filter = set(feed['lifenumber'].unique())
    weight_record = feed[['lifenumber', 'visit_time', 'weight', 'age']].copy()
    ids_to_process = weight_record['lifenumber'].unique()
    final_weight_results = []
    excluded_list, weight_audits = [], []
    with ProcessPoolExecutor(max_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))) as executor:
        future_to_id = {executor.submit(process_individual_weight, lid, weight_record): lid for lid in ids_to_process}
        for future in as_completed(future_to_id):
            lid = future_to_id[future]
            try:
                res, reason, weight_audit = future.result()
                if weight_audit is not None:
                    weight_audit['individual_pass_weight_qc'] = res is not None
                    weight_audits.append(weight_audit)
                if res is not None:
                    final_weight_results.append(res)
                else:
                    excluded_list.append({'lifenumber': lid, 'reason': reason})
                    log_removed(lid, str(lid)[:2], 'Step 3: Weight processing', reason or 'Weight processing failed.')
            except Exception as e:
                excluded_list.append({'lifenumber': lid, 'reason': f'异常: {e}'})
                log_removed(lid, str(lid)[:2], 'Step 3: Weight processing', f'Unhandled exception: {e}')
    if weight_audits:
        audit_weights = pd.concat(weight_audits, ignore_index=True)
        audit_weights.to_csv(os.path.join(audit_dir, 'weight_rlm_record_weights.csv'), index=False)
        counts = audit_weights.groupby('lifenumber').agg(
            n_rlm_records=('rlm_weight', 'size'), n_pass_w025=('record_pass_w025', 'sum'),
            n_pass_day_w09=('used_for_daily_weight', 'sum'), passed_weight_qc=('individual_pass_weight_qc', 'first'))
        counts.reset_index().to_csv(os.path.join(audit_dir, 'weight_qc_individual_summary.csv'), index=False)
        del audit_weights, weight_audits
    pd.DataFrame(excluded_list, columns=['lifenumber', 'reason']).to_csv(
        os.path.join(audit_dir, 'weight_excluded_reasons.csv'), index=False)
    if final_weight_results:
        final_weight = pd.concat(final_weight_results, ignore_index=True)
    else:
        final_weight = pd.DataFrame(columns=['lifenumber', 'visit_time', 'day', 'interpolated_weight'])
    if final_weight.empty:
        save_removed_log(output_dir, farm, current_time)
        raise ValueError('Error: 没有任何个体成功完成体重处理，无法进行后续分析。')
    feed_loc = feed.groupby(['lifenumber',
        'visit_time'])['location'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan).reset_index()
    final_weight = final_weight.merge(feed_loc, on=['lifenumber', 'visit_time'], how='left')
    final_weight['location'] = final_weight.groupby('lifenumber')['location'].transform(
        lambda x: x.fillna(x.mode()[0] if len(x.mode()) > 0 else np.nan))
    final_weight.rename(columns={'interpolated_weight': 'weight'})[['lifenumber', 'location', 'visit_time',
        'weight']].to_csv(os.path.join(output_dir, f'{farm}_all_weights_byday_{current_time}.csv'), index=False)
    final_weight.to_csv(os.path.join(audit_dir, 'daily_weight_with_source.csv'), index=False)
    excluded_df = pd.DataFrame(excluded_list)

    n_excluded = excluded_df['lifenumber'].nunique() if not excluded_df.empty else 0
    if not excluded_df.empty and 'lifenumber' in excluded_df.columns:
        feed = feed[~feed['lifenumber'].isin(excluded_df['lifenumber'])]
    else:
        print('体重处理完成，无个体被剔除。')
    print(f'体重处理完成，被剔除的个体数: {n_excluded}。')
    pd.DataFrame({'metric': ['raw_weight_nonmissing', 'raw_weight_missing', 'pigs_entering_weight_qc',
        'pigs_excluded_weight_qc', 'pigs_accepted_weight_qc'],
        'value': [int(weight_record['weight'].notna().sum()), int(weight_record['weight'].isna().sum()),
            len(ids_to_process), n_excluded, len(ids_to_process) - n_excluded]}).to_csv(
        os.path.join(audit_dir, 'weight_qc_overall_counts.csv'), index=False)
    record_stage('03_weight_qc_accepted_pigs', feed)
    print('步骤4: 采食量数据处理...')
    feed = feed.copy()
    feed['breed'] = feed['lifenumber'].astype(str).str[:2]
    feed['visit_time'] = pd.to_datetime(feed['visit_time'], errors='coerce')
    feed = feed.dropna(subset=['visit_time'])
    record_stage('04_valid_visit_dates', feed)
    final_weight['visit_time'] = pd.to_datetime(final_weight['visit_time'], errors='coerce')
    feed = feed.merge(final_weight[['lifenumber', 'visit_time', 'interpolated_weight']], on=['lifenumber',
        'visit_time'], how='left')
    feed['weight'] = feed['interpolated_weight']
    feed = feed.drop(columns=['interpolated_weight'])
    if feed['weight'].isna().all():
        raise ValueError('Error: 合并后体重数据全为 NaN，请检查主表与体重表的时间戳格式是否对齐。')
    if feed.empty:
        raise ValueError('Error: 原始FCR记录与体重合并后，剩余数据为空。')
    feed = feed.dropna(subset=['weight']).copy()
    record_stage('05_joined_valid_daily_weights', feed)
    if feed.empty:
        raise ValueError('Error: 删除缺失体重后，剩余数据为空。')
    feed['visit_time'] = pd.to_datetime(feed['visit_time'])
    feed = feed.sort_values(by=['lifenumber', 'visit_time']).reset_index(drop=True)
    # Non-positive FI/DUR are labelled invalid; rolling references ignore them.
    feed['fi_anomaly_label'] = 1
    feed['dur_anomaly_label'] = 1
    feed['rate_anomaly_label'] = np.nan
    valid_fi_mask = feed['feed_intake'].notna() & (feed['feed_intake'] > 0) & np.isfinite(feed['feed_intake'])
    valid_dur_mask = feed['duration'].notna() & (feed['duration'] > 0) & np.isfinite(feed['duration'])
    valid_rate_mask = valid_fi_mask & valid_dur_mask
    feed.loc[~valid_fi_mask, 'fi_anomaly_label'] = -1
    feed.loc[~valid_dur_mask, 'dur_anomaly_label'] = -1
    feed['rate'] = np.nan
    feed.loc[valid_rate_mask, 'rate'] = feed.loc[valid_rate_mask, 'feed_intake'] / feed.loc[valid_rate_mask, 'duration']
    feed['rate'] = feed['rate'].replace([np.inf, -np.inf], np.nan)
    valid_rate_mask = valid_rate_mask & feed['rate'].notna()
    valid_flags = feed[['lifenumber', 'visit_time', 'weight', 'feed_intake', 'duration']].copy()
    valid_flags['valid_fi'] = valid_fi_mask
    valid_flags['valid_dur'] = valid_dur_mask
    valid_flags['valid_rate'] = valid_rate_mask
    valid_flags.to_csv(os.path.join(audit_dir, 'feeding_initial_validity.csv'), index=False)
    del valid_flags
    feed['fi_for_median'] = feed['feed_intake'].where(valid_fi_mask, np.nan)
    feed['dur_for_median'] = feed['duration'].where(valid_dur_mask, np.nan)
    feed['rate_for_median'] = feed['rate'].where(valid_rate_mask, np.nan)
    feed = feed.set_index('visit_time')
    feed['ind_fi_median'] = feed.groupby('lifenumber',
        group_keys=False)['fi_for_median'].transform(lambda x: x.rolling(window='7D', center=True, min_periods=1).median())
    feed['ind_dur_median'] = feed.groupby('lifenumber',
        group_keys=False)['dur_for_median'].transform(lambda x: x.rolling(window='7D', center=True, min_periods=1).median())
    feed['ind_rate_median'] = feed.groupby('lifenumber',
        group_keys=False)['rate_for_median'].transform(lambda x: x.rolling(window='7D', center=True, min_periods=1).median())
    feed = feed.reset_index()
    n_before = len(feed)
    ids_before = feed['lifenumber'].nunique()
    median_cols = ['ind_fi_median', 'ind_dur_median', 'ind_rate_median']
    valid_median_mask = feed[median_cols].notna().all(axis=1) & np.isfinite(feed[median_cols]).all(axis=1)
    n_removed = (~valid_median_mask).sum()
    print('FI滚动参考缺失记录数:', feed['ind_fi_median'].isna().sum())
    print('DUR滚动参考缺失记录数:', feed['ind_dur_median'].isna().sum())
    print('RATE滚动参考缺失记录数:', feed['ind_rate_median'].isna().sum())
    missing_medians = feed.loc[~valid_median_mask, ['lifenumber', 'visit_time', 'weight',
        'ind_fi_median', 'ind_dur_median', 'ind_rate_median']]
    missing_medians.to_csv(os.path.join(audit_dir, 'removed_missing_rolling_reference.csv'), index=False)
    feed = feed.loc[valid_median_mask].copy()
    feed = feed.reset_index(drop=True)
    if feed.empty:
        raise ValueError('Error: 剔除缺乏7天有效滚动参考的记录后，数据为空。')
    n_after = len(feed)
    ids_after = feed['lifenumber'].nunique()
    print(f'7天滚动参考筛选：原始记录 {n_before:,} 条，剔除 {n_removed:,} 条，保留 {n_after:,} 条')
    print(f'个体数：筛选前 {ids_before:,}，筛选后 {ids_after:,}，完全剔除 {ids_before - ids_after:,}')
    valid_fi_mask = feed['feed_intake'].notna() & (feed['feed_intake'] > 0) & np.isfinite(feed['feed_intake'])
    valid_dur_mask = feed['duration'].notna() & (feed['duration'] > 0) & np.isfinite(feed['duration'])
    valid_rate_mask = valid_fi_mask & valid_dur_mask & feed['rate'].notna() & np.isfinite(feed['rate'])
    record_stage('06_valid_7day_rolling_references', feed)
    feed['diff_fi_ind'] = feed['feed_intake'] - feed['ind_fi_median']
    feed['diff_dur_ind'] = feed['duration'] - feed['ind_dur_median']
    feed['diff_rate_ind'] = feed['rate'] - feed['ind_rate_median']
    feed = feed.drop(columns=['fi_for_median', 'dur_for_median', 'rate_for_median'])
    # Breed-specific 1-kg quantiles -> weighted quadratic spline -> PCHIP.
    BIN_WIDTH = 1.0
    MIN_PIGS_PER_BIN = 5
    SMOOTH_FACTOR = smooth_factor
    WEIGHT_MIN = 0.5
    WEIGHT_MAX = 2.0
    reference_cols = ['fi_Q5', 'fi_Q95', 'dur_Q5', 'dur_Q95', 'group_rate_median']
    valid_mask = (np.isfinite(feed['weight']) & np.isfinite(feed['feed_intake']) &
                  np.isfinite(feed['duration']) & np.isfinite(feed['rate']) &
                  (feed['feed_intake'] > 0) & (feed['duration'] > 0) &
                  (feed['rate'] > 0) & feed['breed'].notna())
    valid_data = feed.loc[valid_mask].copy()
    valid_data['weight_bin'] = np.floor(valid_data['weight'] / BIN_WIDTH) * BIN_WIDTH
    bin_stats = valid_data.groupby(['breed', 'weight_bin']).agg(
        n_records=('feed_intake', 'size'), n_pigs=('lifenumber', 'nunique'),
        fi_Q5=('feed_intake', lambda x: x.quantile(0.05)),
        fi_Q95=('feed_intake', lambda x: x.quantile(0.95)),
        dur_Q5=('duration', lambda x: x.quantile(0.05)),
        dur_Q95=('duration', lambda x: x.quantile(0.95)),
        group_rate_median=('rate', 'median'),
    ).reset_index()
    bin_stats = bin_stats.loc[bin_stats['n_pigs'] >= MIN_PIGS_PER_BIN].copy()
    bin_stats['weight_center'] = bin_stats['weight_bin'] + BIN_WIDTH / 2
    bin_stats.to_csv(os.path.join(output_dir, 'reference_bin_stats.csv'), index=False)
    valid_data.groupby('breed').agg(n_valid_events=('lifenumber', 'size'), n_pigs=('lifenumber', 'nunique')).to_csv(
        os.path.join(audit_dir, 'population_reference_input_counts.csv'))

    for col in reference_cols:
        feed[col] = np.nan

    def smooth_reference(x, y, n_pigs, x_pred):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n_pigs = np.asarray(n_pigs, dtype=float)
        x_pred = np.asarray(x_pred, dtype=float)
        if len(x) < 2:
            return np.full(len(x_pred), np.nan)
        w = np.sqrt(n_pigs / np.median(n_pigs))
        w = np.clip(w, WEIGHT_MIN, WEIGHT_MAX)
        if len(x) < 4:
            fitted_nodes = y.copy()
        else:
            scale = 1.4826 * np.median(np.abs(y - np.median(y)))
            if scale < 1e-08:
                scale = max(np.std(y), 1e-06)
            s_value = len(x) * (SMOOTH_FACTOR * scale) ** 2
            spline = UnivariateSpline(x, y, w=w, k=2, s=s_value)
            fitted_nodes = spline(x)
        lower = np.empty(len(x))
        upper = np.empty(len(x))
        for i in range(len(x)):
            local = y[max(0, i - 2):min(len(x), i + 3)]
            local_min, local_max = (np.min(local), np.max(local))
            margin = 0.1 * (local_max - local_min)
            lower[i] = max(0.0, local_min - margin)
            upper[i] = local_max + margin
        n_clipped = int(np.count_nonzero((fitted_nodes < lower) | (fitted_nodes > upper)))
        if n_clipped:
            print(f'    参考节点局部约束: {n_clipped}/{len(x)}')
        fitted_nodes = np.clip(fitted_nodes, lower, upper)
        x_eval = np.clip(x_pred, x.min(), x.max())
        predicted = PchipInterpolator(x, fitted_nodes, extrapolate=False)(x_eval)
        return np.maximum(predicted, 0.0)
    for breed, breed_stats in bin_stats.groupby('breed'):
        breed_stats = breed_stats.sort_values('weight_center')
        breed_mask = feed['breed'] == breed
        if not breed_mask.any():
            continue
        x = breed_stats['weight_center'].to_numpy()
        n_pigs = breed_stats['n_pigs'].to_numpy()
        x_pred = feed.loc[breed_mask, 'weight'].to_numpy()
        if len(x) < 2:
            print(f'Warning: 品种 {breed} 有效分箱不足2个，无法建立群体参考。')
            continue
        for col in reference_cols:
            y = breed_stats[col].to_numpy()
            predicted = smooth_reference(x, y, n_pigs, x_pred)
            predicted = np.maximum(predicted, 0)
            feed.loc[breed_mask, col] = predicted
        print(f'品种 {breed}: {len(breed_stats)} 个有效体重分箱，{breed_mask.sum():,} 条记录')
    crossing_counts = {}
    for lower, upper in [('fi_Q5', 'fi_Q95'), ('dur_Q5', 'dur_Q95')]:
        crossing = feed[lower].notna() & feed[upper].notna() & (feed[lower] > feed[upper])
        n_crossing = int(crossing.sum())
        crossing_counts[lower] = n_crossing
        if n_crossing:
            print(f'Warning: {lower}/{upper} 发生 {n_crossing:,} 条分位数交叉，已进行顺序修正。')
            low_values = feed.loc[crossing, [lower, upper]].min(axis=1)
            high_values = feed.loc[crossing, [lower, upper]].max(axis=1)
            feed.loc[crossing, lower] = low_values
            feed.loc[crossing, upper] = high_values
    print('群体参考曲线构建完成。')
    coverage_valid = valid_mask & feed[reference_cols].notna().all(axis=1)
    if coverage_valid.any():
        diagnostic_rows = []
        for breed, g in feed.loc[coverage_valid].groupby('breed'):
            diagnostic_rows.append({
                'breed': breed, 'smooth_factor': SMOOTH_FACTOR,
                'n_records': len(g), 'n_pigs': g['lifenumber'].nunique(),
                'fi_coverage': ((g['feed_intake'] >= g['fi_Q5']) &
                                (g['feed_intake'] <= g['fi_Q95'])).mean(),
                'dur_coverage': ((g['duration'] >= g['dur_Q5']) &
                                 (g['duration'] <= g['dur_Q95'])).mean(),
                'fi_crossing': crossing_counts.get('fi_Q5', 0),
                'dur_crossing': crossing_counts.get('dur_Q5', 0),
            })
        pd.DataFrame(diagnostic_rows).to_csv(os.path.join(output_dir, 'reference_coverage_summary.csv'), index=False)
    diagnostic_dir = os.path.join(output_dir, 'reference_curve_diagnostics')
    os.makedirs(diagnostic_dir, exist_ok=True)
    plot_configs = [('FI', 'feed_intake', 'fi_Q5', 'fi_Q95'), ('DUR', 'duration', 'dur_Q5', 'dur_Q95'), ('RATE',
        'rate', 'group_rate_median', None)]
    for breed, breed_stats in bin_stats.groupby('breed'):
        breed_feed = feed.loc[feed['breed'] == breed].copy()
        breed_stats = breed_stats.sort_values('weight_center')
        if breed_feed.empty:
            continue
        fig, axes = plt.subplots(3, 1, figsize=(10, 13), sharex=True)
        for ax, (name, observed, lower, upper) in zip(axes, plot_configs):
            pred = breed_feed[['weight', lower] + ([upper] if upper else [])].dropna().sort_values('weight')
            pred = pred.groupby('weight', as_index=False).mean()
            ax.scatter(breed_stats['weight_center'], breed_stats[lower], s=18, alpha=0.6, label=f'Observed {lower}')
            ax.plot(pred['weight'], pred[lower], linewidth=2, label=f'Smoothed {lower}')
            if upper is not None:
                ax.scatter(breed_stats['weight_center'], breed_stats[upper], s=18, alpha=0.6, label=f'Observed {upper}')
                ax.plot(pred['weight'], pred[upper], linewidth=2, label=f'Smoothed {upper}')
                ax.fill_between(pred['weight'], pred[lower], pred[upper], alpha=0.12)
            ax.set_ylabel(name)
            ax.set_title(f'{breed}: {name} reference curves')
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        axes[-1].set_xlabel('Body weight (kg)')
        fig.tight_layout()
        safe_breed = str(breed).replace('/', '_')
        fig.savefig(os.path.join(diagnostic_dir, f'{safe_breed}_reference_curves.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)
    print(f'参考曲线诊断图已保存到: {diagnostic_dir}')
    # Diagnostics-only mode stops before Isolation Forest and ADFI.
    if reference_only:
        save_audit()
        print('Reference-only mode: skipping Isolation Forest, mixed models, and ADFI.')
        return
    feed['ratio_rate_group'] = feed['rate'] / (feed['group_rate_median'] + 1e-05)
    feed['fi_lower'] = feed['feed_intake'] - feed['fi_Q5']
    feed['fi_upper'] = feed['fi_Q95'] - feed['feed_intake']
    feed['dur_lower'] = feed['duration'] - feed['dur_Q5']
    feed['dur_upper'] = feed['dur_Q95'] - feed['duration']
    fi_features = ['diff_fi_ind', 'fi_lower', 'fi_upper', 'weight']
    dur_features = ['diff_dur_ind', 'dur_lower', 'dur_upper', 'weight']
    rate_features = ['diff_rate_ind', 'ratio_rate_group', 'rate', 'weight']
    all_features = list(set(fi_features + dur_features + rate_features))
    feed[all_features] = feed[all_features].replace([np.inf, -np.inf], np.nan)

    def get_mad_threshold(scores, k):
        median_score = np.median(scores)
        left_deviations = median_score - scores[scores < median_score]
        if len(left_deviations) == 0:
            mad_left = 0
        else:
            mad_left = np.median(left_deviations)
        return median_score - k * mad_left
    k_val = 4.5
    feed['fi_anomaly_score'] = np.nan
    # Separate FI, DUR and RATE Isolation Forest models; one-sided MAD threshold.
    if_fi_mask = (feed['fi_anomaly_label'] != -1) & feed[fi_features].notna().all(axis=1)
    if if_fi_mask.any():
        X_fi = feed.loc[if_fi_mask, fi_features].values
        iforest_fi = IsolationForest(contamination=0.02, random_state=42, n_jobs=-1)
        iforest_fi.fit(X_fi)
        scores_fi = iforest_fi.score_samples(X_fi)
        threshold_fi = get_mad_threshold(scores_fi, k=k_val)
        feed.loc[if_fi_mask, 'fi_anomaly_score'] = scores_fi
        feed.loc[if_fi_mask, 'fi_anomaly_label'] = np.where(scores_fi < threshold_fi, -1, 1)
    feed['dur_anomaly_score'] = np.nan
    if_dur_mask = (feed['dur_anomaly_label'] != -1) & feed[dur_features].notna().all(axis=1)
    if if_dur_mask.any():
        X_dur = feed.loc[if_dur_mask, dur_features].values
        iforest_dur = IsolationForest(contamination=0.02, random_state=42, n_jobs=-1)
        iforest_dur.fit(X_dur)
        scores_dur = iforest_dur.score_samples(X_dur)
        threshold_dur = get_mad_threshold(scores_dur, k=k_val)
        feed.loc[if_dur_mask, 'dur_anomaly_score'] = scores_dur
        feed.loc[if_dur_mask, 'dur_anomaly_label'] = np.where(scores_dur < threshold_dur, -1, 1)
    feed['rate_anomaly_score'] = np.nan
    normal_mask = (feed['fi_anomaly_label'] == 1) & (feed['dur_anomaly_label'] == 1) & feed[rate_features].notna().all(axis=1)
    if normal_mask.sum() > 20:
        X_rate = feed.loc[normal_mask, rate_features].values
        iforest_rate = IsolationForest(contamination=0.02, random_state=42, n_jobs=-1)
        iforest_rate.fit(X_rate)
        rate_score = iforest_rate.score_samples(X_rate)
        threshold_rate = get_mad_threshold(rate_score, k=k_val)
        rate_label = np.where(rate_score < threshold_rate, -1, 1)
        feed.loc[normal_mask, 'rate_anomaly_score'] = rate_score
        feed.loc[normal_mask, 'rate_anomaly_label'] = rate_label
    fi_abn_ratio = (feed['fi_anomaly_label'] == -1).mean()
    dur_abn_ratio = (feed['dur_anomaly_label'] == -1).mean()
    rate_evaluated_mask = feed['rate_anomaly_label'].notna()
    rate_abn_ratio = (feed.loc[rate_evaluated_mask, 'rate_anomaly_label'] == -1).mean() if rate_evaluated_mask.any() else np.nan
    print(f'FI异常比例: {fi_abn_ratio:.2%}')
    print(f'DUR异常比例: {dur_abn_ratio:.2%}')
    print(f'RATE异常比例: {rate_abn_ratio:.2%}')
    print(f'RATE实际检测记录数: {rate_evaluated_mask.sum():,}')
    # Direction/severity annotations do not change Isolation Forest labels.
    feed = add_directional_features(feed)
    feed.to_csv(os.path.join(output_dir, f'{farm}_feed_with_error_{current_time}.csv'), index=False)
    record_stage('07_feeding_anomaly_labels_assigned', feed)
    audit_direction_features(feed, audit_dir)
    anomaly_counts = []
    for breed, g in feed.groupby('breed'):
        for metric in ('fi', 'dur', 'rate'):
            col = f'{metric}_anomaly_label'
            n_eval = int(g[col].notna().sum()) if metric == 'rate' else int((g[col].notna()).sum())
            n_abn = int((g[col] == -1).sum())
            anomaly_counts.append({'breed': breed, 'metric': metric, 'n_records': len(g),
                'n_evaluated_or_labeled': n_eval, 'n_anomalous': n_abn,
                'anomaly_rate_all_events': n_abn / len(g) if len(g) else np.nan,
                'anomaly_rate_evaluated': n_abn / n_eval if n_eval else np.nan})
    pd.DataFrame(anomaly_counts).to_csv(os.path.join(audit_dir, 'feeding_anomaly_summary.csv'), index=False)
    feed.groupby(['breed', 'lifenumber']).agg(n_events=('weight', 'size'),
        n_fi_anomalies=('fi_anomaly_label', lambda x: int((x == -1).sum())),
        n_dur_anomalies=('dur_anomaly_label', lambda x: int((x == -1).sum())),
        n_rate_anomalies=('rate_anomaly_label', lambda x: int((x == -1).sum()))).to_csv(
        os.path.join(audit_dir, 'feeding_anomalies_by_individual.csv'))

    daily_stats = get_daily_stats(feed)
    daily_stats = add_daily_directional_features(feed, daily_stats)
    audit_daily_partition(daily_stats, audit_dir)
    daily_stats.to_csv(os.path.join(audit_dir, 'daily_stats_before_feed_filter.csv'), index=False)
    ids_before_feed_filter = set(daily_stats['lifenumber'].unique())
    initial_mask = daily_stats['feed_normal'] > 0
    filtered_stage1 = daily_stats[initial_mask]
    daily_stats = daily_stats.loc[initial_mask].copy()
    record_stage('08_daily_positive_normal_feed', daily_stats)
    ids_after_feed_filter = set(daily_stats['lifenumber'].unique())
    feed_excluded_ids = list(ids_before_feed_filter - ids_after_feed_filter)
    if feed_excluded_ids:
        log_removed(feed_excluded_ids, None, '08_daily_positive_normal_feed', 'No day with positive normal feed.')
    daily_stats['breed'] = daily_stats['lifenumber'].str[:2]
    daily_stats.to_csv(os.path.join(output_dir, f'{farm}_daily_stats_{current_time}.csv'), index=False)
    # Fit the anomaly-adjustment model independently for each breed.
    model_audit = []
    results_list = []
    print('\n========== Fitting feeding-event adjustment models ==========')
    for breed in daily_stats['breed'].unique():
        data_b = daily_stats[daily_stats['breed'] == breed].copy()
        fitted, report = fit_feed_adjustment_model(breed, data_b)
        model_audit.append(report)
        if fitted is not None:
            results_list.append(fitted)
    pd.DataFrame(model_audit).to_csv(os.path.join(output_dir, 'model_fit_audit.csv'), index=False)
    if not results_list:
        save_audit()
        raise RuntimeError('No breed models converged. See model_fit_audit.csv.')
    daily_stats_all = pd.concat(results_list, ignore_index=True)
    daily_stats_all.to_csv(os.path.join(output_dir, 'daily_stats_after_mixed_model.csv'), index=False)
    record_stage('09_mixed_model_accepted_days', daily_stats_all)
    final_ADFI = fill_and_format_ADFI(final_weight, daily_stats_all)
    if final_ADFI.empty:
        save_audit()
        raise RuntimeError('The QC pipeline produced an empty ADFI table.')
    final_ADFI['breed'] = final_ADFI['id'].astype(str).str[:2]
    final_ADFI.to_csv(os.path.join(output_dir, f'{farm}_weight_ADFI_{current_time}.csv'), index=False)
    final_ADFI.to_csv(os.path.join(output_dir, 'weight_ADFI.csv'), index=False)
    record_stage('10_final_ADFI', final_ADFI, 'id')
    summary = []
    for breed, g in final_ADFI.groupby('breed'):
        summary.append({'breed': breed, 'status': 'OK',
            'n_pigs': int(g['id'].nunique()), 'n_days': len(g),
            'mean_ADFI': float(g['ADFI'].mean()), 'sum_ADFI': float(g['ADFI'].sum())})
    pd.DataFrame(summary).to_csv(os.path.join(output_dir, 'qc_summary.csv'), index=False)
    save_audit()
    print(f'QC processing complete: {final_ADFI.id.nunique()} pigs, {len(final_ADFI)} pig-days')
    print(f'Results saved to: {output_dir}')


if __name__ == '__main__':
    freeze_support()
    parser = argparse.ArgumentParser(description='Quality control and daily feed-intake correction for pig EFS visits.')
    parser.add_argument('--feed', type=str, required=True, help='Input EFS visits CSV.')
    parser.add_argument('--farm', type=str, required=True, help='Dataset label for output files.')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory.')
    parser.add_argument('--smooth-factor', type=float, default=0.5,
        help='Reference curve smoothing factor (default: 0.5).')
    parser.add_argument('--reference-only', action='store_true',
        help='Generate body weight / population reference diagnostics, skipping ADFI modelling.')
    args = parser.parse_args()
    if not np.isfinite(args.smooth_factor) or args.smooth_factor < 0:
        parser.error('--smooth-factor must be finite and >= 0')
    main(args.feed, args.farm, args.output_dir, args.smooth_factor, args.reference_only)
