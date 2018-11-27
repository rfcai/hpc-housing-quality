#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 10:07:12 2018

@author: kevinhsu
"""

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from Load_data import Load_Data, Cleaning_Block
from sklearn.metrics import accuracy_score


#Load housing_data
df = Load_Data('housing_data.csv')

withna = df[df.isnull().any(axis=1)]


df = df.dropna()

zero_df = df[df['housing_wall_num']!=0]
features = list(df)
features.remove('housing_floor_num')



#split data
df['train'] = np.random.uniform(0, 1, len(df)) <= .75
train = df[df['train']==True]
test = df[df['train']==False]


#numeric label
train['iso3'] = pd.factorize(train['iso3'])[0]
test['iso3'] = pd.factorize(test['iso3'])[0]



#data cleaning

vals = ['roof','wall','floor']
for val in vals:
    Cleaning_Block(train, val)
    Cleaning_Block(test, val)


train_feat = train[features]
test_feat = test[features]



y = train['housing_floor_num'].values.astype(int)

#build model
RFC = RandomForestClassifier(n_jobs=10, random_state=0)

RFC.fit(train_feat, y)
preds = RFC.predict(test_feat)

#ans = pd.factorize(test['iso3'])[0]
ans = test['housing_floor_num'].values.astype(int)

conf_matrix = pd.crosstab(ans, preds, rownames=['Actual'], colnames=['Predicted'])

a_score = accuracy_score(ans, preds)

