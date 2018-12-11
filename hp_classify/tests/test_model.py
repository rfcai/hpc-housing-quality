#write tests
"""This is a module used to test a module: "model.py" and its relevant functions x and y

xx is a function that 


"""
# import packages
import pytest
import pandas as pd
import re

#import custom modules fpr testing
import sys 
sys.path.append('.')
import prep.prep_data as prep
import model.fuzzy as fz

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

#read in example data using your function and then pass it through the cleaning pipeline
df = prep.read_then_clean(FILEPATH, CLEAN_COLS)
df_clean = prep.remove_garbage_codes(df, STR_VARS, STR_GARBAGE)
df_clean = prep.extract_ranking(df_clean, NUM_VARS)
df_clean = prep.remove_garbage_codes(df_clean, RANK_VARS, RANK_GARBAGE)

#build corpus of known and unknown strings
str_list, idk_strings = fz.build_corpus(df, base_var, rank_var, rank_values)

def test_build_corpus():
    """This function tests a function that is used to build corpora of known and unknown words from a df 
    that contains columns with string value descriptions. The testing is done to confirm that the resulting corpuses are built
    entirely from words that are present in the pandas df column that was passed in, and furthermore in the rows that result when
    subsetting by the rank class that they are supposed to be a part of.
    """  
    for x in STR_VARS:

        rank_var = x + "_rank"
        str_list, idk_strings = fz.build_corpus(df_clean, x, rank_var, RANK_LIST)

        #verify that each of the unknown strings exist in the appropriate column in the input pandas df
        for y in idk_strings:
            assert (y in df_clean[x].unique()) == True

        #verify that each of the known strings exist in the appropriate column in the input pandas df
        #note that here we subset the pandas df to the correct rank before testing the column values
        for rank, rank_num in zip(RANK_LIST, range(len(RANK_LIST))):
            for z in np.random.choice(str_list[rank_num], 5): #only pull 5 random strings and test for speed purposes

                assert (z in df_clean[df_clean[rank_var] == rank][x].unique()) == True
    
def test_fuzzy_scan():
    #find distribution of scores for each string
    distrib = fz.fuzzy_scan(idk_strings, str_list)