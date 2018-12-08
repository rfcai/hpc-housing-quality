#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 21:26:13 2018

@author: kevinhsu
"""

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib 


import matplotlib.pyplot as plt
import seaborn as sn


def rfc_model(x, y, label):
    """
    This function builds a random forest model and saves the model as .sav in
    the current directory.
    
    Args:
        x: Feature vector
        y: Label
        Label: This is the label specifed under prediction
    Returns:
        RFC: Return the built model
        
    """
    #build model
    RFC = RandomForestClassifier(n_estimators=17 , random_state=0, max_features=2)
    #data fitting
    RFC.fit(x, y)
    # save the model to directory
    filename = 'finalized_'+label+'_model.sav'
    joblib.dump(RFC, filename)
    return RFC
    
def confusion_matrix(y, pred, plot=False):
    """
    This function generates the confusion matrix of the prediction, Enable plot 
    to generate the figure
    
    Args: 
        y: This is the actual rank of the data
        pred: This is the predicted rank of the data
    """
    #confusion matrix 
    conf_matrix = pd.crosstab(y, pred, rownames=['Actual Rank'], colnames=['Predicted Rank'])
    
    if plot:
        #plot heatmap
        plt.figure(figsize = (10,7))
        sn.heatmap(conf_matrix, annot=True)
    
def load_model(filename):
    """
    This function loads the saved model in the current directory
    
    Args:
        filename: specifed .sav model file
    Return:
        loaded_model: Return the RFC model to the module
    """
    loaded_model = joblib.load(filename)
    
    return loaded_model
    
