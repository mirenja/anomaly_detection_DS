import pandas as pd

def load_and_prepare(path='data/borg_traces_data.csv'):
    df = pd.read_csv(path)
    df['time_sec'] = df['time'] / 1_000_000
    df['label_raw'] = ((df['event'] == 'FAIL') | (df['failed'] == 1)).astype(int)
    df = df.sort_values(['machine_id', 'time'])
    df['machine_failed'] = df.groupby('machine_id')['label_raw'].transform('max')
    df['max_time'] = df.groupby('machine_id')['time'].transform('max')
    df['time_to_end'] = df['max_time'] - df['time']
    
    HORIZON = 30 * 60  # 30 min in seconds
    df['label'] = ((df['label_raw'] == 1) & (df['time_to_end'] <= HORIZON)).astype(int)
    
    return df