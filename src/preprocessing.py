import pandas as pd
import gdown
from pathlib import Path


DATA_PATH = Path("data/borg_traces_data.csv")
FILE_ID   = "1jhOhl4CcGbnkocOFHw9EYownBRSUQtV6"

DATA_PATH = Path("data/borg_traces_data.csv")


def load_and_prepare():
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(exist_ok=True)
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}",str(DATA_PATH),quiet=False)
    df = pd.read_csv(DATA_PATH)
    df['time_sec'] = df['time'] / 1_000_000
    df['label_raw'] = ((df['event'] == 'FAIL') | (df['failed'] == 1)).astype(int)
    df = df.sort_values(['machine_id', 'time'])
    df['machine_failed'] = df.groupby('machine_id')['label_raw'].transform('max')
    df['max_time'] = df.groupby('machine_id')['time'].transform('max')
    df['time_to_end'] = df['max_time'] - df['time']
    
    HORIZON = 30 * 60  # 30 min in seconds
    df['label'] = ((df['label_raw'] == 1) & (df['time_to_end'] <= HORIZON)).astype(int)
    
    return df