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
FILEPATH = '../data/example_data.csv'
CLEAN_COLS = ['housing_roof', 'housing_wall', 'housing_floor']

DIGITS = str([str(x) for x in range(100 + 1)])
PUNCT = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
SPACE = '     '

STR_VARS = ['housing_roof', 'housing_wall', 'housing_floor']
NUM_VARS = [s + '_num' for s in STR_VARS]
RANK_VARS = [s + '_rank' for s in STR_VARS]

STR_GARBAGE = ['nan', 'other', 'not a dejure resident', 'not dejure resident']
RANK_GARBAGE = ['4', '5', '6', '7', '8', '9', 'n']

# if you compile the regex string first, it's even faster
re_dig = re.compile('\d')
re_punct = re.compile('\W+')
re_white = re.compile(' +')

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

# This is our base dataset and it needs to be cleaned properly. The second argument specifies
# the cols with string values that we want to be cleaned.


#TODO, this test is for multiple functions. I think it is probably bad practice to test multiple functions in one,
#but it takes so long to read in the df and clean it that it seems to me more efficient to use a single shot.
def test_read_then_clean():
    """This function tests that a custom exception called RowCountException
    will be returned when more than 1k rows are expected.
    """
    #read in df using your function and then using pandas regular csv read, then compare the resulting dfs
    df = prep.read_then_clean(FILEPATH, CLEAN_COLS)
    raw_csv = pd.read_csv(FILEPATH)
    
    #also passed it through the rest of the cleaning pipeline on order to compare df to df_clean
    df_clean = prep.remove_garbage_codes(df, STR_VARS, STR_GARBAGE)
    df_clean = prep.extract_ranking(df_clean, NUM_VARS)
    df_clean = prep.remove_garbage_codes(df_clean, RANK_VARS, RANK_GARBAGE)
    
    #assert that our function did not add or remove rows
    assert len(raw_csv) == len(df), "read_then_clean function is modifying the original csv's length"
    assert len(df.columns) == len(raw_csv.columns), "read_then_clean function is modifying the original csv's width"
    
    #assert that our initial read function cleaned up the strings in the columns we provided
    #TODO: this test will fail if the columns were entirely clean to begin with (is this possible?)
    for x in CLEAN_COLS:
        assert (set(df[x].unique()) == set(raw_csv[x].unique())) == False, "string columns are unmodified"
        
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
