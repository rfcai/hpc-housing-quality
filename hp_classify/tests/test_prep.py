#write tests
"""This is a module used to test a module: "prep.py" and its relevant functions read_then_clean and clean_text

read_then_clean is a function that takes a csv with messy string values and 
creates then cleans a pandas df
using clean_text

This module tests that function by ensuring that it returns expected exceptions and
does not contain unexpected values.

This module also uses the opportunity of having the df loaded to tests the 
functions later in the data cleaning pipeline, including 
remove_garbage_codes, which removes unacceptable values and replaces them with NaN
and extract_ranking, which generates the ordinal ranking variable from an input numerical code
"""
# import packages
import pytest
import pandas as pd
import re

#import custom modules fpr testing
import sys 
sys.path.append('.')
import prep.prep_data as prep

#set globals for tests
#set globals for tests
FILEPATH = '../data/housing_data.csv'
CLEAN_COLS = ['housing_roof', 'housing_wall', 'housing_floor']

DIGITS = str([str(x) for x in range(100 + 1)])
PUNCT = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
SPACE = '     '

# if you compile the regex string first, it's even faster
re_dig = re.compile('\d')
re_punct = re.compile('\W+')
re_white = re.compile(' +')

STR_VARS = ['housing_roof', 'housing_wall', 'housing_floor']
NUM_VARS = [s + '_num' for s in STR_VARS]
RANK_VARS = [s + '_rank' for s in STR_VARS]
LABEL = 'roof'
ATTR = ['int_year', 'housing_roof_num', 'housing_wall_num',
        'housing_floor_num', 'iso3']
VAR = ['roof','floor','wall']

STR_GARBAGE = ['nan', 'other', 'not a dejure resident', 'not dejure resident']
RANK_GARBAGE = ['4', '5', '6', '7', '8', '9', 'n']

#read in the df using our function in order to pass to later tests
#read in df using your function and then using pandas regular csv read, then compare the resulting dfs
df = prep.read_then_clean(FILEPATH, CLEAN_COLS)
raw_csv = pd.read_csv(FILEPATH)

#also passed it through the rest of the cleaning pipeline on order to compare df to df_clean
df_clean = prep.remove_garbage_codes(df, STR_VARS, STR_GARBAGE)
df_clean = prep.extract_ranking(df_clean, NUM_VARS)
df_clean = prep.remove_garbage_codes(df_clean, RANK_VARS, RANK_GARBAGE)

#test setup for rfc model data preprocessing functions
df_rfc = prep.load_data(FILEPATH)
df_rank_check = prep.ranking(df_rfc, VAR)
FEATURES = prep.extract_features(df_rank_check,LABEL)
df_shuffle_check, RANK_NUM = prep.shuffle_redistribute(df_rank_check, LABEL)
x_train, x_test, y_train, y_test = prep.train_test_split(df_shuffle_check, FEATURES, LABEL)


def test_globals():
    """This function tests that the test globals are properly defined.
    """
    #assert that digits are removed
    assert re_dig.search(DIGITS) != None, "global doesn't contain digits!" 
    #assert that punctutation is removed
    assert re_punct.search(PUNCT) != None, "global doesn't contain punctuation!"
    #assert that excessive whitespace is removed
    assert re_white.search(SPACE) != None, "global doesn't contain whitespace!"
    

def test_clean_text():
    """This function tests that the clean text function is doing its job.
    """
    #assert that digits are removed
    assert re_dig.search(prep.clean_text(DIGITS)) == None, "clean_text did not remove the digits from test global." 
    #assert that punctutation is removed
    assert re_punct.search(prep.clean_text(PUNCT)) == None, "clean_text did not remove the punctuation from test global."
    #assert that excessive whitespace is removed
    assert re_white.search(prep.clean_text(SPACE)) == None, "clean_text did not remove the whitespace from test global."

def test_read_then_clean():
    """This function tests our master function and the subsquent data cleaning pipeline.
    """    
    #assert that our function did not add or remove rows
    assert len(raw_csv) == len(df), "read_then_clean function is modifying the original csv's length"
    assert len(df.columns) == len(raw_csv.columns), "read_then_clean function is modifying the original csv's width"
    
    #assert that our initial read function cleaned up the strings in the columns we provided
    #TODO: this test will fail if the columns were entirely clean to begin with (is this possible?)
    for x in CLEAN_COLS:
        assert (set(df[x].unique()) == set(raw_csv[x].unique())) == False, "string columns are unmodified"

def test_cleaning_pipeline():
    """This function tests our cleaning pipeline to make sure that 
    garbage values are removed and ranks are create
    """ 
    #assert that rankings were generated in the next step of the pipeline
    for x in RANK_VARS:
        #verify that it wasnt originally present in df
        assert (x in df) == False, "rank column present in raw data"
        #assert that this column was added 
        assert x in df_clean, "rank column was not added by extract_ranking fx"
        
    #assert that garbage was removed 
    for x in STR_VARS:
        for y in STR_GARBAGE:
            print(x, y)
            #assert that it is removed
            assert (y in df_clean[x].unique()) == False, "garbage values not removed from clean dataframe"    

def test_load_data():
    """This function tests that our output dataframe contains the correct 
    columns and have no missing data nor zeros
    """
    #assert that dataframe contains the specified columns
    assert list(df_rfc) == ATTR
    #assert that there is no zeros in dataframe
    assert df_rfc.eq(0).any().any() == False
    #assert that there is no missing values in dataframe
    assert df_rfc.notnull().values.any() == True
    
def test_df_manipulation():
    """This function tests if the special characters are filtered out, and if the rank columns are 
    correctly created
    """
    #assert that the three columns for rank have been created 
    assert set(['roof_rank','wall_rank','floor_rank']).issubset(df_rank_check.columns) == True
    
    #
    assert df_rank_check.eq('.').any().any() == False
    assert df_rank_check.eq('?').any().any() == False
    
    #
    for var in VAR:
        assert df_rank_check['housing_'+var+'_num'].dtype == 'float64'
        assert (df_rank_check['housing_'+var+'_num']>35).any() == False
        assert (df_rank_check['housing_'+var+'_num']<10).any() == False
        assert df_rank_check[var+'_rank'].eq(0).any() == False
        assert (df_rank_check[var+'_rank']>3).any() == False
        
def test_features():
    """This function tests if the features are the six input features as designed
    """
    #assert that length is correct, and label related features are excluded
    assert len(prep.extract_features(df_rank_check,'roof')) == 6
    assert LABEL+'_rank' not in prep.extract_features(df_rank_check,'roof')
    assert 'housing'+LABEL+'_num' not in prep.extract_features(df_rank_check,'roof')
    
def test_shuffle():
    """This function tests if the data is shuffle randomly and equally
    distributed rank dataframe 
    """
    import random
    threshold = 1000
    #check is the data is approx equally distributed based on each rank with a give therhold
    avg = int(sum(RANK_NUM)/3)
    for rank in RANK_NUM:
        assert abs(avg - rank) < threshold
    #randomly check if the data frame is shuffled
    for rand in range(10):
        rand = random.randint(1,len(df_shuffle_check))
        assert df_shuffle_check.iloc[rand,:].any() != df_rank_check.iloc[rand,:].any()
        
def test_data_split():
    """This function checks if the feature and label length matchs, and also the
    data type of the encoded feature iso3
    """
    #assert the length of the feature and label set in both testing and training data
    assert len(x_test) == len(y_test)
    assert len(x_train) == len(y_train)
    
    #assert that the iso3 country codes have been encoded to integers
    assert x_test['iso3'].dtype == 'int64'
        