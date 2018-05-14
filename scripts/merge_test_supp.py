import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc
import matplotlib.pyplot as plt
import os
import pickle


print("\nLoading data...")

dtypes = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint8',
            'os'            : 'uint16',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8',
            'click_id'      : 'uint32',
            }

test_supp_df = pd.read_csv("../input/test_supplement.csv", parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
test_df = pd.read_csv("../input/test.csv", parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])

sub = pd.read_pickle('../input/new_features_v8/sub_test_suppl_v1.pkl') 

test_supp_df['is_attributed'] = sub['is_attributed'].values
print(test_supp_df.head(5))

del sub
gc.collect()

print('\nprojecting prediction onto test')

join_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
all_cols = join_cols + ['is_attributed']

test_df = test_df.merge(test_supp_df[all_cols], how='left', on=join_cols)

test_df = test_df.drop_duplicates(subset=['click_id'])

print("\nWriting the submission data into a csv file...")

test_df[['click_id', 'is_attributed']].to_csv('sub_ft_v15.csv', index=False)

print("All done...")