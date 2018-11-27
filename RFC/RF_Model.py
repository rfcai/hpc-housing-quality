#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 10:07:12 2018

@author: kevinhsu

This is an example testing out the sklearn RFC,
The model predicts the rank of floor with given features:
 ['int_year',
 'housing_roof_num',
 'housing_wall_num',
 'iso3',
 'roof_rank',
 'wall_rank']
 
 Install seaborn to plot heatmap
"""

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from Load_data import Load_Data, Cleaning_Block
from sklearn.metrics import accuracy_score
import seaborn as sn
import matplotlib.pyplot as plt

#Load housing_data
df = Load_Data('housing_data.csv')

#dataframe with na in any row
withna = df[df.isnull().any(axis=1)]

#drop missing data
df = df.dropna()


#data cleaning

vals = ['roof','wall','floor']
for val in vals:
    Cleaning_Block(df, val)

#ranking
for val in vals:
    df[val+'_rank'] = (df['housing_'+val+'_num']/10).apply(np.floor)
    df.loc[df[val+'_rank'] >= 4, val+'_rank'] = 4

df = df[df.floor_rank != -1]
df = df[df.floor_rank != -10]

#list features
features = list(df)
features.remove('housing_floor_num')
features.remove('floor_rank')

#split data to test and training set
df['train'] = np.random.uniform(0, 1, len(df)) <= .75
train = df[df['train']==True]
test = df[df['train']==False]


#numeric label
train['iso3'] = pd.factorize(train['iso3'])[0]
test['iso3'] = pd.factorize(test['iso3'])[0]


train_feat = train[features]
test_feat = test[features]


y = train['floor_rank'].values.astype(int)

#build model
RFC = RandomForestClassifier(n_jobs=10, random_state=0)

#data fitting
RFC.fit(train_feat, y)

#test model
preds = RFC.predict(test_feat)

ans = test['floor_rank'].values.astype(int)

#confusion matrix 
conf_matrix = pd.crosstab(ans, preds, rownames=['Actual'], colnames=['Predicted'])


a_score = accuracy_score(ans, preds)

print('Accuracy score:{}'.format(a_score))


#plot heatmap
plt.figure(figsize = (10,7))
sn.heatmap(conf_matrix, annot=True)

plt.title('Accuracy score:'+str(a_score))
