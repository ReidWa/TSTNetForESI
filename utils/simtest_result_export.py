import re
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ['SD', 'DLE', 'AUC', 'SE']
TRIAL_COLUMNS = [
    'run_id',
    'test_name',
    'method',
    'run_index',
    'trial_index_in_run',
    'trial_index_global',
    'n_channels',
    'n_times',
    'SD',
    'DLE',
    'AUC',
    'SE',
]


def order_trial_columns(df):
    ordered = [column for column in TRIAL_COLUMNS if column in df.columns]
    return df[ordered + [column for column in df.columns if column not in ordered]]


def _valid_summary(summary_df):
    if 'error' not in summary_df.columns:
        return summary_df.copy()
    return summary_df[summary_df['error'].isna()].copy()


def _build_method_summary(summary_df):
    valid = _valid_summary(summary_df)
    columns = [
        column for column in ['n_runs', 'n_trials', *METRICS]
        if column in valid.columns
    ]
    if valid.empty:
        return pd.DataFrame(
            columns=columns,
            index=pd.Index([], name='test_name'),
        )
    return valid.groupby('test_name')[columns].mean(numeric_only=True)


def _build_comparison(method_summary, method):
    comparison = method_summary.reset_index()
    return comparison.rename(
        columns={
            column: f'{method}_{column}'
            for column in method_summary.columns
        }
    )


def _build_overall(method_summary, method):
    row = {}
    for column in method_summary.columns:
        values = pd.to_numeric(method_summary[column], errors='coerce')
        row[f'{column}_mean'] = values.mean()
    for column in method_summary.columns:
        values = pd.to_numeric(method_summary[column], errors='coerce')
        row[f'{column}_std'] = values.std()
    return pd.DataFrame([row], index=[method])


def _build_testset_stats(test_df, method):
    row = {
        'test_name': test_df['test_name'].iloc[0],
        'method': method,
        'n_trials': int(len(test_df)),
    }
    for metric in METRICS:
        values = pd.to_numeric(test_df[metric], errors='coerce')
        row.update({
            f'{metric}_mean': values.mean(),
            f'{metric}_std': values.std(),
            f'{metric}_median': values.median(),
            f'{metric}_min': values.min(),
            f'{metric}_max': values.max(),
            f'{metric}_zero_n': int((values == 0).sum()),
            f'{metric}_nan_n': int(values.isna().sum()),
        })
    return row


def _parse_test_condition(test_name):
    patterns = (
        (r'^extents_(-?\d+)mm$', 'extent_mm', int),
        (r'^firingsource_(\d+)$', 'n_sources', int),
        (r'^snr_(minus)?(\d+)dB$', 'snr_db', None),
    )
    for pattern, parameter, converter in patterns:
        match = re.match(pattern, test_name)
        if not match:
            continue
        if parameter == 'snr_db':
            value = int(match.group(2))
            if match.group(1):
                value = -value
        else:
            value = converter(match.group(1))
        return parameter, value
    return None, None


def _build_parameter_summaries(trial_df, all_stats):
    condition_columns = [
        'parameter', 'value', 'test_name', 'method', 'n_trials',
        'SD_mean', 'SD_std', 'SD_median',
        'DLE_mean', 'DLE_std', 'DLE_median',
        'AUC_mean', 'AUC_std', 'AUC_median',
        'SE_mean', 'SE_std', 'SE_median', 'DLE_zero_n',
    ]
    condition_rows = []
    for row in all_stats.to_dict('records'):
        parameter, value = _parse_test_condition(row['test_name'])
        if parameter is None:
            continue
        condition_rows.append({
            'parameter': parameter,
            'value': value,
            **{column: row.get(column) for column in condition_columns[2:]},
        })
    condition = pd.DataFrame(condition_rows, columns=condition_columns)

    enriched = trial_df.copy()
    if not enriched.empty:
        parsed = enriched['test_name'].map(_parse_test_condition)
        enriched['parameter'] = parsed.map(lambda item: item[0])
        enriched = enriched[enriched['parameter'].notna()]

    group_rows = []
    if not enriched.empty:
        for (parameter, method), group in enriched.groupby(
            ['parameter', 'method'], sort=False
        ):
            row = {
                'parameter': parameter,
                'method': method,
                'n_trials': len(group),
            }
            for metric in METRICS:
                values = pd.to_numeric(group[metric], errors='coerce')
                row[f'{metric}_mean'] = values.mean()
                row[f'{metric}_std'] = values.std()
            group_rows.append(row)
    group_columns = [
        'parameter', 'method', 'n_trials',
        'SD_mean', 'SD_std', 'DLE_mean', 'DLE_std',
        'AUC_mean', 'AUC_std', 'SE_mean', 'SE_std',
    ]
    grouped = pd.DataFrame(group_rows, columns=group_columns)

    best_rows = []
    for row in condition_rows:
        for metric in METRICS:
            best_rows.append({
                'parameter': row['parameter'],
                'value': row['value'],
                'test_name': row['test_name'],
                'metric': metric,
                'best_method': row['method'],
                'best_value': row[f'{metric}_mean'],
            })
    best = pd.DataFrame(best_rows, columns=[
        'parameter', 'value', 'test_name', 'metric',
        'best_method', 'best_value',
    ])
    return condition, grouped, best


def _save_per_testset_stats(trial_df, output_dir, method):
    output_dir.mkdir(parents=True, exist_ok=True)
    if trial_df.empty or 'test_name' not in trial_df.columns:
        empty = pd.DataFrame()
        empty.to_csv(
            output_dir / 'all_testsets_method_stats.csv',
            index=False,
            encoding='utf-8-sig',
        )
        empty.to_excel(
            output_dir / 'all_testsets_method_stats.xlsx',
            index=False,
        )
        empty.to_csv(
            output_dir / 'all_testsets_best_methods.csv',
            index=False,
            encoding='utf-8-sig',
        )
        return empty
    rows = []
    for test_name in sorted(trial_df['test_name'].dropna().unique()):
        test_df = trial_df[trial_df['test_name'] == test_name]
        row = _build_testset_stats(test_df, method)
        rows.append(row)
        pd.DataFrame([row]).to_csv(
            output_dir / f'{test_name}_method_stats.csv',
            index=False,
            encoding='utf-8-sig',
        )

    all_stats = pd.DataFrame(rows)
    all_stats.to_csv(
        output_dir / 'all_testsets_method_stats.csv',
        index=False,
        encoding='utf-8-sig',
    )
    all_stats.to_excel(
        output_dir / 'all_testsets_method_stats.xlsx',
        index=False,
    )
    best_rows = []
    for row in rows:
        for metric in METRICS:
            best_rows.append({
                'test_name': row['test_name'],
                'metric': metric,
                'best_method': method,
                'best_value': row[f'{metric}_mean'],
            })
    pd.DataFrame(best_rows).to_csv(
        output_dir / 'all_testsets_best_methods.csv',
        index=False,
        encoding='utf-8-sig',
    )
    return all_stats


def export_simtest_results(
    summary_rows,
    trial_rows,
    save_dir,
    timestamp,
    method='oriISTA',
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows)
    if 'method' not in summary_df.columns:
        summary_df['method'] = method

    trial_df = pd.DataFrame(trial_rows)
    if trial_df.empty:
        trial_df = pd.DataFrame(columns=TRIAL_COLUMNS)
    trial_df = order_trial_columns(trial_df)
    if 'method' not in trial_df.columns:
        trial_df['method'] = method
        trial_df = order_trial_columns(trial_df)

    method_summary = _build_method_summary(summary_df)
    comparison = _build_comparison(method_summary, method)
    overall = _build_overall(method_summary, method)

    paths = {
        'method_summary': save_dir / f'test_results_{method}_{timestamp}.csv',
        'method_trial': save_dir / f'test_results_{method}_trial_level_{timestamp}.csv',
        'all_trials': save_dir / f'test_results_all_methods_trial_level_{timestamp}.csv',
        'comparison': save_dir / f'test_results_comparison_{timestamp}.csv',
        'overall': save_dir / f'test_results_overall_{timestamp}.csv',
        'parameter_condition_summary': (
            save_dir / f'parameter_condition_method_summary_{timestamp}.csv'
        ),
        'parameter_group_summary': (
            save_dir / f'parameter_group_method_summary_{timestamp}.csv'
        ),
        'parameter_condition_best': (
            save_dir / f'parameter_condition_best_methods_{timestamp}.csv'
        ),
        'per_testset_dir': save_dir / f'per_testset_method_stats_{timestamp}',
    }

    method_summary.to_csv(paths['method_summary'])
    trial_df.to_csv(paths['method_trial'], index=False)
    trial_df.to_csv(paths['all_trials'], index=False)
    comparison.to_csv(paths['comparison'], index=False)
    overall.to_csv(paths['overall'])
    all_stats = _save_per_testset_stats(
        trial_df, paths['per_testset_dir'], method
    )
    condition, grouped, best = _build_parameter_summaries(
        trial_df, all_stats
    )
    condition.to_csv(paths['parameter_condition_summary'], index=False)
    grouped.to_csv(paths['parameter_group_summary'], index=False)
    best.to_csv(paths['parameter_condition_best'], index=False)
    return paths
