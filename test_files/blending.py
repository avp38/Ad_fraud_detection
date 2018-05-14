import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os

print("Reading the data...\n")

df1 = pd.read_csv('models/sub-it200102.csv')
df2 = pd.read_csv('models/sub_stacked.csv')
df3 = pd.read_csv('models/gpu_test1.csv')
df4 = pd.read_csv('sub_ft_v10.csv')
models = {
    'df1':{
        'name': 'blended_1',
        'score': 98.11,
        'df': df1
    },
    'df2':{
        'name': 'linear_stack',
        'score': 97.78,
        'df': df2
    },
    'df3':{
        'name': 'gpu',
        'score': 96.64,
        'df': df3
    },
    'df4':{
        'name': 'my_lgbm',
        'score': 98.09,
        'df': df4
    }
}

count_models = len(models)  

isa_lg = 0
isa_hm = 0
isa_am = 0
isa_gm=0
print("Blending...\n")
for df in models.keys() : 
    isa_lg += np.log(models[df]['df'].is_attributed)
    isa_hm += 1/(models[df]['df'].is_attributed)
    isa_am += isa_am
    isa_gm *= isa_gm
isa_lg = np.exp(isa_lg/count_models)
isa_hm = count_models/isa_hm
isa_am = isa_am/count_models
isa_gm = (isa_gm)**(1/count_models)

print("Isa log\n")
print(isa_lg[:count_models])
print()
print("Isa harmo\n")
print(isa_hm[:count_models])

sub_log = pd.DataFrame()
sub_log['click_id'] = df1['click_id']
sub_log['is_attributed'] = isa_lg
sub_log.head()

sub_hm = pd.DataFrame()
sub_hm['click_id'] = df1['click_id']
sub_hm['is_attributed'] = isa_hm
sub_hm.head()

sub_fin=pd.DataFrame()
sub_fin['click_id']=df1['click_id']
sub_fin['is_attributed']= (7*isa_lg+3*isa_hm)/10

print("Writing...")
sub_fin.to_csv('sub_ft_v18.csv', index=False, float_format='%.9f')