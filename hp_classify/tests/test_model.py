#write tests
"""This is a module used to test a module: "model.py" and its main functions, including build_corpus, fuzzy_scan,
and fuzzy_predict.

build_corpus is used to define corpora of words associated with a given ranking and also a corpus of unknown words.
Here, this functionality is tested to ensure that the corpora returned contain only words that are actually in the
specified rows and columns of the pandas df they were pulled from.

fuzzy_scan is a function that takes an unknown word and scans it for similarity against a list of known words that
subdivided by class. A distribution of values is returned that can be used to predict which class is most probable
for the unknown word. Here, several aspects of this functionality are tested, including xxx

fuzzy_predict is a function that takes a distribution of values for each class and predicts which class an unknown
word is most likely to be based on a given similarity threshold. Here, this function is tested by yyy


"""
# import packages
import pytest
import pandas as pd
import re
import numpy as np

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

RANK_LIST = ['1', '2', '3']

#read in example data using your function and then pass it through the cleaning pipeline
df = prep.read_then_clean(FILEPATH, CLEAN_COLS)
df_clean = prep.remove_garbage_codes(df, STR_VARS, STR_GARBAGE)
df_clean = prep.extract_ranking(df_clean, NUM_VARS)
df_clean = prep.remove_garbage_codes(df_clean, RANK_VARS, RANK_GARBAGE)

def test_build_corpus():
    """This function tests a function that is used to build corpora of known and unknown words from a df 
    that contains columns with string value descriptions. The testing is done to confirm that the resulting corpuses are built
    entirely from words that are present in the pandas df column that was passed in, and furthermore in the rows that result when
    subsetting by the rank class that they are supposed to be a part of.
    """  
    import numpy as np
    
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
    
def test_fuzzy():
"""This function tests a series function that are used to predict the unknown ranking of string values using a 
training dataset in which the rankings are known for other string values. Corpora for each ranking are compiled and
then the unknown values are compared against these in order to predict the most likely ranking.
"""
    #build a fake dataset from which to generate a distribution with expected results
    df_sim = pd.DataFrame({ 'piggy' : pd.Series(['straw', 'straws', 'stick', 'sticks', 'brick', 'bricks', 'brickz']),
                            'piggy_rank' : [1, 1, 2, 2, 3, 3, np.nan],
                            'piggy_rank_og' : [1, 1, 2, 2, 3, 3, 3]})
    sim_rank_list = [1,2,3] #save a list with the expected rank levels in your simulated df

    #build a corpus based on the simulated dataset
    str_list, idk_strings = fz.build_corpus(df_sim, 'piggy', 'piggy_rank', sim_rank_list)

    assert len(idk_strings) == 1

    #find distribution of scores for each string
    distrib = fz.fuzzy_scan(idk_strings, str_list)

    #the length of the output df should be equal to the length of the longest corpora
    assert len(distrib) == len(max(str_list, key=len)), "the output distribution df is not the correct length"

    #the output df should have the a# of columns that equals # of input rank categories + 1
    assert len(distrib.columns) == len(piggy_rank_list)+1, "the output distribution df is not the correct width"

    #the output df should have a column called word that contains only the values in idk_strings
    assert distrib.word.unique() in idk_strings
