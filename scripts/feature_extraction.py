# Script for feature extraction - v5

# Import Statements
import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc
import matplotlib.pyplot as plt
import os




# Extract time delta features
def do_next_Click( df,agg_suffix='nextClick', agg_type='float32'):
    print(">> \nExtracting time calculation features...\n")
    GROUP_BY_NEXT_CLICKS = [
    {'groupby': ['ip', 'app', 'device', 'os', 'channel']},
    {'groupby': ['ip', 'os', 'device', 'app']},    
    {'groupby': ['app', 'device', 'channel']}
    ]

    # Calculate the time to next click for each group
    for spec in GROUP_BY_NEXT_CLICKS:   
       # Name of new feature
        new_feature = '{}_{}'.format('_'.join(spec['groupby']),agg_suffix)    
        # Unique list of features to select
        all_features = spec['groupby'] + ['click_time']
        # Run calculation
        df[new_feature] = (df[all_features].groupby(spec[
            'groupby']).click_time.shift(-1) - df.click_time).dt.seconds.astype(agg_type)        
        gc.collect()
    return (df)




# Extract aggregate features

# Extract count feature using different columns
def count_feat( df, group_cols, agg_type='uint16', show_max=False, show_agg=True ):
    agg_name='{}count'.format('_'.join(group_cols))  
    if show_agg:
        print( "\nAggregating by ", group_cols ,  '... and saved in', agg_name )
    gp = df[group_cols][group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )
    

# Extract unique count feature using different cols
def count_unique( df, group_cols, counted, agg_type='uint8', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_countuniq'.format(('_'.join(group_cols)),(counted))  
    if show_agg:
        print( "\nCounting unqiue ", counted, " by ", group_cols ,  '... and saved in', agg_name )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].nunique().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )


# Extract cumulative count feature  from different cols    
def cumulative_count( df, group_cols, counted,agg_type='uint16', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_cumcount'.format(('_'.join(group_cols)),(counted)) 
    if show_agg:
        print( "\nCumulative count by ", group_cols , '... and saved in', agg_name  )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].cumcount()
    df[agg_name]=gp.values
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )



    
# Main Script File
def DO(frm,to,nchunk):
    dtypes = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint8',
            'os'            : 'uint16',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8',
            'click_id'      : 'uint32',
            }

    print('loading train data...',frm,to)
    train_df = pd.read_csv("../input/train.csv", parse_dates=['click_time'], skiprows=range(1,frm), nrows=to-frm, dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])

    print('loading test data...')
    test_df = pd.read_csv("../input/test_supplement.csv", parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])

    df_converted = pd.DataFrame()
    
    # Load chunks of 5 million
    chunksize = (10 ** 6)*5
    chunk_ct = 0
    # Filter values that have 'is_attributed'==1, and merge these values into one dataframe
    for chunk in pd.read_csv('../input/train.csv', chunksize=chunksize, parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed']):
        filtered = (chunk[(np.where(chunk['is_attributed']==1, True, False))])
        df_converted = pd.concat([df_converted, filtered], ignore_index=True, )
        chunk_ct = chunk_ct+5
        if chunk_ct==135:
            break


    print("\nEntries with attr=1 size: ", len(df_converted))        
    train_df = df_converted.append(train_df)    

    len_train = len(train_df)
    train_df=train_df.append(test_df)
    
    del test_df,df_converted        
    gc.collect()

    train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('int8')
    train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('int8') 
    train_df = do_next_Click( train_df,agg_suffix='nextClick', agg_type='float32'  ); gc.collect()
    
    # Count unique features
    train_df = count_unique( train_df, ['ip'], 'channel' ); gc.collect() 
    train_df = count_unique( train_df, ['ip'], 'app'); gc.collect() 
    train_df = count_unique( train_df, ['ip'], 'device'); gc.collect() 
    
    # Cumulative count features
    train_df = cumulative_count( train_df, ['ip', 'device', 'os'], 'app'); gc.collect() 
    
    # Count features
    train_df = count_feat( train_df, ['ip', 'day', 'hour'] ); gc.collect() 
    train_df = count_feat( train_df, ['ip', 'app']); gc.collect() 
    train_df = count_feat( train_df, ['ip', 'app', 'os']); gc.collect() 


    del train_df['day']
    gc.collect()


    test_df = train_df[len_train:]
    val_df = train_df[(len_train-val_size):len_train]
    train_df = train_df[:(len_train-val_size)]
    

    print("\ntrain size: ", len(train_df))
    print("\nvalid size: ", len(val_df))
    print("\ntest size : ", len(test_df))
    print(train_df.columns.values)
    print(len(train_df.columns.values))

    # Feature Dimension is 15+4 --> Note train label("is_attributed") is also inluded in the dataframe
    print("\nsaving test data")
    test_df.to_pickle('../input/new_features_v11/test.pkl') # Dimn: 18790469 x 31
    del test_df
    gc.collect()    

    print("\nsaving val data")
    val_df.to_pickle('../input/new_features_v11/val.pkl') # Dimn: 2500000 x 31
    del val_df
    gc.collect()

    print("\nsaving train data")
    train_df.to_pickle('../input/new_features_v11/train.pkl') # Dimn: 65000000 x 31
    del train_df
    gc.collect()

    print("\ndone")

    


# Final Run
nrows=184903891-1
nchunk=150000000 
val_size=15000000

frm=nrows-nchunk
to=frm+nchunk

DO(frm,to,nchunk)



DO(frm,to,nchunk, FILENO)