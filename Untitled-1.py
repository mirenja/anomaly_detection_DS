# %%
import pandas as pd
import numpy as np

df = pd.read_csv("../data/borg_traces_data.csv")
df.head()


# %%
print("columns")
print(df.columns)
print("shape")
print(df.shape)

# %%
df['event'].value_counts()

# %%
df['failed'].value_counts()

# %% [markdown]
# consider FAILURE = event == FAIL OR failed == 1
# define the new column to identify all our fail instances

# %%
df['label_raw'] = ((df['event'] == 'FAIL') | (df['failed'] == 1)).astype(int)

# %%
df = df.sort_values(['machine_id', 'time'])
print(df.columns)
df.head(10)

# %% [markdown]
# check if the machine ever experience failure at some point, then append that to that machine ID
# use case machine  168846390496

# %%
df['machine_failed'] = df.groupby('machine_id')['label_raw'].transform('max')
df.head(10)


# %% [markdown]
# assesing time to failure, how far is the system from end of its observed life

# %%
df['max_time'] = df.groupby('machine_id')['time'].transform('max')
df['time_to_end'] = df['max_time'] - df['time']
df.head(10)


# %%
df['machine_id'].dtype

# %%
df[df['machine_id'] == -1]

# %% [markdown]
# #building a prewarning window 30 min,
# if the difference between the time for the machine and the maximum time the machine run (time_to_end) is less than the rediction window, and the machine is selected as fail , the label it as 1

# %%
HORIZON = 30 * 60


# %%
df['label'] = ((df['label_raw'] == 1) & (df['time_to_end'] <= HORIZON)).astype(int)
df.head(5)
df[df['label'] == 1].head(2)

# %% [markdown]
# FEATURES:modelling the system behaviour, we just pick the runtime resource-performance metrics

# %%
features = df[[
    'assigned_memory',
    'page_cache_memory',
    'cycles_per_instruction',
    'memory_accesses_per_instruction'
]]

# %% [markdown]
# what patterns correlate failure rates, does it make sense to track a trainig loss?

# %%
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

X = features
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=False
)

model = XGBClassifier()
model.fit(X_train, y_train)


