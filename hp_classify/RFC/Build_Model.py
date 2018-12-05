#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 18:24:59 2018

@author: kevinhsu
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from Functions import Load_Data, Ranking, Shuffle_Redistribute, Train_Test_Split


def Build_RFC(df, LABEL):
    
    vals = ['roof','wall','floor']

    #Clean and Rank
    df = Ranking(df, vals)
    #list features
    features = list(df)
    features.remove('housing_'+LABEL+'_num')
    features.remove(LABEL+'_rank')
    #features.remove('train')
    new_df = Shuffle_Redistribute(df, LABEL, Redistribute=True)


    #spilt train and test set 
    train, test = Train_Test_Split(new_df)
    
    X_train = train[features]
    X_test = test[features]

    Y_train = train[LABEL+'_rank'].values.astype(int)
    Y_test = test[LABEL+'_rank'].values.astype(int)

    RFC = RandomForestClassifier(n_estimators=17 , random_state=0, max_features=2)

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
    return RFC, test_score