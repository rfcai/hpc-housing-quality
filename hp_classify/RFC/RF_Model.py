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


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import matplotlib.pyplot as plt
import seaborn as sn
from Functions import Load_Data, Ranking, Shuffle_Redistribute, Train_Test_Split
from Build_Model import Build_RFC
#Load housing_data
df = Load_Data('housing_data.csv')

#dataframe with na in any row
withna = df[df.isnull().any(axis=1)]

#drop missing data
df = df.dropna()


LABEL = 'floor'

vals = ['roof','wall','floor']

#Clean and Rank
df = Ranking(df, vals)

#list features
features = list(df)
features.remove('housing_'+LABEL+'_num')
features.remove(LABEL+'_rank')


new_df = Shuffle_Redistribute(df, LABEL, Redistribute=True)
rank_dist = new_df.groupby(LABEL+'_rank').size().tolist()


#spilt train and test set 
train, test = Train_Test_Split(new_df)


X_train = train[features]
X_test = test[features]


Y_train = train[LABEL+'_rank'].values.astype(int)
Y_test = test[LABEL+'_rank'].values.astype(int)


#build model
RFC = RandomForestClassifier(n_estimators=17 , random_state=0, max_features=2, oob_score=True)

#data fitting
RFC.fit(X_train, Y_train)

#test model
Pred_test = RFC.predict(X_test)   
Pred_train = RFC.predict(X_train)

test_score = accuracy_score(Y_test, Pred_test)
train_score = accuracy_score(Y_train, Pred_train)
print('Train set Accuracy score:{:.2f}%'.format(train_score*100))
print('Test set Accuracy score:{:.2f}%'.format(test_score*100))

target_names = ['rank 1', 'rank 2', 'rank 3']
print(classification_report(Y_test, Pred_test, target_names=target_names))


#confusion matrix 
conf_matrix = pd.crosstab(Y_test, Pred_test, rownames=['Actual'], colnames=['Predicted'])

#plot heatmap
plt.figure(figsize = (10,7))
sn.heatmap(conf_matrix, annot=True)

_, score = Build_RFC(df,'roof')