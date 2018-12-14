# write tests
"""This is a module used to test a module: "rfc_build.py"  main functions
The rfc_model function is a wrapping of building the random forest classifier with give feature and label set,
it also saves the trained model as .sav file in the repo

The confusion matrix function creates a heatmap of the prediction with the diagonal representing correctly
predicted predictions

The load_model function loads in the .sav model into the module

"""

# import packages
import pytest
import pandas as pd
import numpy as np
import sklearn
from sklearn.externals import joblib


# import custom modules fpr testing
import sys
sys.path.append('../hp_classify')
import prep.prep_data as prep
import rfc_build as rf

#Gobals
FILEPATH = '../data/housing_data.csv'
LABEL = 'roof'
ATTR = ['int_year','housing_roof_num', 'housing_wall_num',
        'housing_floor_num', 'iso3']
VAR = ['roof','floor','wall']

#test setup for rfc model data preprocessing functions
df = prep.load_data(FILEPATH)
df = prep.ranking(df, VAR)
FEATURES = prep.extract_features(df,LABEL)
df, RANK_NUM = prep.shuffle_redistribute(df, LABEL)
x_train, x_test, y_train, y_test = prep.train_test_split(df, FEATURES, LABEL)
RFC = rf.rfc_model(x_train, y_train, LABEL)
pred_test = RFC.predict(x_test) 
c_matrix = rf.confusion_matrix(pred_test, y_test)
loaded_model = rf.load_model('finalized_roof_model.sav')


def test_rfc_build():
    """This function test if the rfc is correctly generated 
    """
    #assert that the rfc is a sklearn rfc model
    assert type(RFC) == sklearn.ensemble.forest.RandomForestClassifier
    
def test_confusion_matrix():
    """This function test if the return confusion matrix is correctly generated
    """
    #assert that the confusion matrix is 3 x 3 each x and y-axis representing 1-3 rank
    assert len(c_matrix) == 3
    assert len(c_matrix.iloc[:,1]) == 3
    
def test_load_model():
    """This function test if the rfc model is correctly loaded by asserting the model type
    """
    #assert the the loaded model is a sklearn rfc model
    assert type(loaded_model) == sklearn.ensemble.forest.RandomForestClassifier
    

    
    
    
    
    



