import pandas as pd
import numpy as np
from scipy import stats

def compute_poisson(df, interval=3600):
    df = df.copy()
    df['time_sec'] = df['time'] / 1_000_000
    
    fail_events = df[df['label_raw'] == 1].copy()
    fail_events['time_bin'] = (fail_events['time_sec'] // interval).astype(int)
    
    fail_counts = (
        fail_events.groupby(['machine_id', 'time_bin'])
        .size()
        .reset_index(name='fail_count')
    )
    
    lambda_ = fail_counts['fail_count'].mean()
    variance = fail_counts['fail_count'].var()
    
    lambda_per_machine = (
        fail_counts.groupby('machine_id')['fail_count'].mean()
        .sort_values(ascending=False)
    )
    
    return {
        'fail_counts': fail_counts,
        'lambda_': lambda_,
        'variance': variance,
        'lambda_per_machine': lambda_per_machine
    }


def compute_conditional_poisson(df, resource_col='assigned_memory',
                                 threshold=None, interval=3600):
    df = df.copy()
    df['time_sec'] = df['time'] / 1_000_000

    if threshold is None:
        threshold = df[resource_col].median()

    df['resource_group'] = np.where(
        df[resource_col] > threshold,
        f'{resource_col} HIGH (>{threshold:.3f})',
        f'{resource_col} LOW (<={threshold:.3f})'
    )

    results = {}
    for group in df['resource_group'].unique():
        subset = df[df['resource_group'] == group]
        fail_events = subset[subset['label_raw'] == 1].copy()
        fail_events['time_bin'] = (fail_events['time_sec'] // interval).astype(int)

        fail_counts = (
            fail_events.groupby(['machine_id', 'time_bin'])
            .size()
            .reset_index(name='fail_count')
        )

        if len(fail_counts) > 0:
            results[group] = {
                'fail_counts': fail_counts,
                'lambda_': fail_counts['fail_count'].mean(),
                'variance': fail_counts['fail_count'].var(),
                'n': len(fail_counts)
            }

    return results, threshold


def build_window_features(df, interval=3600):
    """
    Aggregate raw row-level telemetry into per-machine per-time-bin features.
    Each row in the output = one machine in one time window.
    This gives the model a view of system behaviour over time
    rather than isolated snapshots.
    """
    df = df.copy()
    df['time_sec'] = df['time'] / 1_000_000
    df['time_bin'] = (df['time_sec'] // interval).astype(int)


    if 'label' not in df.columns:
        df = df.sort_values(['machine_id', 'time'])
        df['max_time'] = df.groupby('machine_id')['time'].transform('max')
        df['time_to_end'] = df['max_time'] - df['time']
        HORIZON = 30 * 60 * 1_000_000  # 30 min in microseconds (time col is in µs)
        df['label'] = ((df['label_raw'] == 1) & (df['time_to_end'] <= HORIZON)).astype(int)

    # aggregate resource signals per machine per window
    window_features = df.groupby(['machine_id', 'time_bin']).agg(
        # memory signals
        avg_memory=('assigned_memory', 'mean'),
        max_memory=('assigned_memory', 'max'),
        std_memory=('assigned_memory', 'std'),

        # page cache signals
        avg_page_cache=('page_cache_memory', 'mean'),
        max_page_cache=('page_cache_memory', 'max'),

        # cpu efficiency signals
        avg_cpi=('cycles_per_instruction', 'mean'),
        max_cpi=('cycles_per_instruction', 'max'),
        std_cpi=('cycles_per_instruction', 'std'),

        # memory pressure signals
        avg_mapi=('memory_accesses_per_instruction', 'mean'),
        max_mapi=('memory_accesses_per_instruction', 'max'),

        # failure signals within this window
        fail_count=('label_raw', 'sum'),
        event_count=('event', 'count'),
        failure_rate=('label_raw', 'mean'),

        # label: was there a pre-warning failure in this window?
        label=('label', 'max'),
    ).reset_index()

    # fill NaN std values (single-event windows have no std)
    window_features['std_memory'] = window_features['std_memory'].fillna(0)
    window_features['std_cpi'] = window_features['std_cpi'].fillna(0)

    return window_features