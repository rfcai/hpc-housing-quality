# write tests
"""This is a module used to test a module: "model.py" and its main functions, including build_corpus, fuzzy_scan,
and fuzzy_predict.

build_corpus is used to define corpora of words associated with a given ranking and also a corpus of unknown words.
Here, this functionality is tested to ensure that the corpora returned contain only words that are actually in the
specified rows and columns of the pandas df they were pulled from.

fuzzy_scan is a function that takes an unknown word and scans it for similarity against a list of known words that
subdivided by class. A distribution of values is returned that can be used to predict which class is most probable
for the unknown word. Here, several aspects of this functionality are tested, including whether it returns a df that is
the expect size and shape.

fuzzy_predict is a function that takes a distribution of values for each class and predicts which class an unknown
word is most likely to be based on a given similarity threshold. Here, this function is tested by checking the size and
shape of the df that it returns and also testing whether or not it can accurately predict on a dummy dataset.

The the latter two functions is tested by constructing a simulated dataframe in which we expect the prediction to be 100%
accurate. We will follow this dataframe through each function in the fuzzy modelling pipeline and then test to
assert that all behaviour is as expected.


"""
# import packages
import pytest
import pandas as pd
import numpy as np

# import custom modules fpr testing
import sys

sys.path.append('.')
import prep.prep_data as prep
import model.fuzzy as fz

# set globals for tests
FILEPATH = '../data/test.pkl'

DIGITS = str([str(x) for x in range(100 + 1)])
PUNCT = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
SPACE = '     '

STR_VARS = ['housing_roof', 'housing_wall', 'housing_floor']
NUM_VARS = [s + '_num' for s in STR_VARS]
RANK_VARS = [s + '_rank' for s in STR_VARS]

STR_GARBAGE = ['nan', 'other', 'not a dejure resident', 'not dejure resident']
RANK_GARBAGE = ['4', '5', '6', '7', '8', '9', 'n']

RANK_LIST = ['1', '2', '3']

# read in example data using your function and then pass it through the cleaning pipeline
df = prep.read_then_clean(FILEPATH, STR_VARS)
df_clean = prep.remove_garbage_codes(df, STR_VARS, STR_GARBAGE)
df_clean = prep.extract_ranking(df_clean, NUM_VARS)
df_clean = prep.remove_garbage_codes(df_clean, RANK_VARS, RANK_GARBAGE)

# also simulate a dataframe with predictions we can expect and run it through the model pipeline to test results
df_sim = pd.DataFrame({'piggy': pd.Series(['straw', 'straws', 'stick', 'sticks', 'brick', 'bricks', 'brickz']),
                           'piggy_rank': [1, 1, 2, 2, 3, 3, np.nan],
                           'piggy_rank_og': [1, 1, 2, 2, 3, 3, 3],
                           'train': [1, 1, 1, 1, 1, 1, 0]})
sim_rank_list = [1, 2, 3]  # save a list with the expected rank levels in your simulated df
rank_dictionary = {'natural': 1, 'rudimentary': 2, 'finished': 3}
rank_values = list(rank_dictionary.values())
rank_keys = list(rank_dictionary.keys())

def test_build_corpus():
    """This function tests a function that is used to build corpora of known and unknown words from a df
    that contains columns with string value descriptions. The testing is done to confirm that the resulting corpora are built
    entirely from words that are present in the pandas df column that was passed in, and furthermore in the rows that result when
    subsetting by the rank class that they are supposed to be a part of.
    """
    import numpy as np

    for x in STR_VARS:

        rank_var = x + "_rank"
        str_list, idk_strings = fz.build_corpus(df_clean, x, rank_var, RANK_LIST)

        # verify that each of the unknown strings exist in the appropriate column in the input pandas df
        for y in idk_strings:
            assert y in df_clean[x].unique()

        # verify that each of the known strings exist in the appropriate column in the input pandas df
        # note that here we subset the pandas df to the correct rank before testing the column values
        for rank, rank_num in zip(RANK_LIST, range(len(RANK_LIST))):
            for z in np.random.choice(str_list[rank_num], 5):  # only pull 5 random strings and test for speed purposes

                assert z in df_clean[df_clean[rank_var] == rank][x].unique()


def test_sim_fuzzy_corpus():
    """This function tests the build_corpus function from our fuzzy module by making sure that it returns an expected
    number of unknown words from our simulated df and also that it returns the expected corpora for each known rank
    grouping.
    """

    # build a corpus based on the simulated dataset
    sim_str_list, sim_idk_strings = fz.build_corpus(df_sim, 'piggy', 'piggy_rank', sim_rank_list)

    # there should only be one unknown word in the simulated df
    assert len(sim_idk_strings) == 1, "the simulated df only has one unknown word!"

    # each rank in the simulated df has two words
    for x in range(len(sim_str_list)):
        assert len(sim_str_list[x]) == 2, "each class of simulated df can only have two known words in the corpus"

def test_sim_fuzzy_distrib():
    """This function tests the fuzzy_distrib function from our fuzzy module by making sure that it returns output that
    is the expected length and shape. ALso tested is the functionality that the words being analayzed in the output
    dataframe are present in the input list of unknown strings.
    """
    # build a corpus based on the simulated dataset
    sim_str_list, sim_idk_strings = fz.build_corpus(df_sim, 'piggy', 'piggy_rank', sim_rank_list)

    # find distribution of scores for each string
    sim_distrib = fz.fuzzy_scan(sim_idk_strings, sim_str_list)

    # the length of the output df should be equal to the length of the longest corpora
    assert len(sim_distrib) == len(max(sim_str_list, key=len)), "the output distribution df is not the correct length"

    # the output df should have the a# of columns that equals # of input rank categories + 1
    assert len(sim_distrib.columns) == len(sim_rank_list) + 1, "the output distribution df is not the correct width"

    # the output df should have a column called word that contains only the values in idk_strings
    assert sim_distrib.word.unique() in sim_idk_strings, "the output distribution should only analyze words in idk_strings"

def test_sim_fuzzy_pred():
    """This function tests the fuzzy_pred function from our fuzzy module by making sure that it returns output that
    is the expected length and shape. ALso tested is the functionality that the column of predictions has been generated
    and that the output was 100% accurate as we should expect from the construction of our simulated df.
    """
    # build a corpus based on the simulated dataset
    sim_str_list, sim_idk_strings = fz.build_corpus(df_sim, 'piggy', 'piggy_rank', sim_rank_list)
    # find distribution of scores for each string
    sim_distrib = fz.fuzzy_scan(sim_idk_strings, sim_str_list)

    # predict class based on probability of exceeding similarity cutoff of 75
    sim_preds = fz.fuzzy_predict(sim_distrib, rank_keys, 'word', 75, rank_dictionary)

    # the length of the prediction df should be equal to the length of the unknown words corpus
    assert len(sim_preds) == len(sim_idk_strings), "the output prediction df is not the correct length"

    # the prediction df should have # of columns that equals # of input rank categories + 1
    assert len(sim_preds.columns) == len(sim_rank_list) + 1, "the output prediction df is not the correct width"

    # the prediction df should contain a column called "pred"
    assert ("pred" in sim_preds.columns), "prediction column not being generated"

    # merge results back on the test data to validate
    sim_out = df_sim[df_sim['train'] == 0]
    sim_out = pd.merge(sim_out,
                       sim_preds,
                       left_on='piggy',
                       right_on='word',
                       how='left')

    # assert that the prediction was accurate, as expected
    assert np.allclose(sim_out['piggy_rank_og'], sim_out['pred']), "prediction was not correct!"
