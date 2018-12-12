#write tests
"""This is a module used to test the filter_one_word_materials function
"""
# import packages

#import custom modules fpr testing
import os
import pytest
import sys
sys.path.append('.')
import prep.prep_data as prep
import model.semantic as sem

#set globals for tests
FILEPATH = '../data/test.pkl'

#setup lists of vars to work with
STR_VARS = ['housing_roof', 'housing_wall', 'housing_floor']
NUM_VARS = [s + '_num' for s in STR_VARS]
RANK_VARS = [s + '_rank' for s in STR_VARS]

#which variable do you want to predictn (currently: floor/wall/roof)
DEP_VAR = "housing_roof"
PRED_VAR = DEP_VAR + "_rank" #will always be using the strings to predict ranking

#setup a filter to select which surveys you want to work with
SVY_FILTER = ['MACRO_DHS']

#garbage lists
STR_GARBAGE = ['nan', 'other', 'not a dejure resident', 'not dejure resident']
RANK_GARBAGE = ['4', '5', '6', '7', '8', '9', 'n']

#dictionaries
PRED_DICT = {'natural':'1', 'rudimentary':'2', 'finished':'3'} #map categories back to ranks

df = prep.read_then_clean(FILEPATH, STR_VARS)
df_clean = prep.remove_garbage_codes(df, STR_VARS, STR_GARBAGE)
df_clean = prep.extract_ranking(df_clean, NUM_VARS)
df_clean = prep.remove_garbage_codes(df_clean, RANK_VARS, RANK_GARBAGE)
df_clean = df_clean.dropna(subset=[DEP_VAR])

def test_expected_number_of_rows():
    """Has the function successfully filtered out all the materials described with more than one word?"""

    test_df = df_clean[0:20]
    assert (sem.filter_one_word_materials(test_df, DEP_VAR).shape[0] ==
            sum(test_df[DEP_VAR].str.get_dummies(sep=' ').T.sum() == 1))

def test_raise_error_if_no_material_with_one_word():
    """Does the function raise an error if there is no material described with one word in the corpus?"""

    with pytest.raises(Exception) as err:
        test_df = df_clean[0:2]
        sem.filter_one_word_materials(test_df, DEP_VAR)

    assert 'NoOneWordException' in str(err) #verify that your custom error was returned