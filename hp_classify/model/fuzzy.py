def build_corpus(df, str_var, rank_var, rank_list):
    """This is a helper function for this module. It is used to build the corpuses that will be used to analyze
    word similarity in order to make a prediction for unknown words. It returns a separate corpus for each provided
    rank and also a vector of unknown words which currently are not associated with a rank and need to be predicted.

    TODO: ?

    :param df: This is a pandas df that has words which are associated with a rank and unknown words.
    :param str_var: This is a string indicating the variable you want to analyze the string values of.
    :param rank_var: This is a string indicating the variable you want to classify the string_var into ranks by.
    :param rank_list: This is a list of the rank categories you want to classify by. rank_var should contain these vals.

    :return: out: This is a list which length=length(rank_list). Each object in the list is a vector of words (corpus)
        that are associated with this rank.
    :return: other: This is a vector of unknown words that we will compare against every corpus in out to classify.
    """
    # import necessary modules
    import pandas as pd
    import numpy as np
    from fuzzywuzzy import fuzz
    from fuzzywuzzy import process

    out = []

    for x in rank_list:
        print("building corpus for rank #", x)
        out.append(df[df[rank_var] == x][str_var].values)

    print("extracting unknown strings")
    other = df[~df[rank_var].isin(rank_list)][str_var].unique()
    other = other[~pd.isnull(other)]  # cant classify NaN
    print("need to classify", len(other), "unknown strings")

    return (out, other)

def fuzzy_scan(unknown_list, corpus_list, jupyter=False):
    """This is a helper function for this module. It is used to scan each word on the list of unknown words by
    comparing it to every word in each corpus in corpus_list. The words are compared using the fuzzywuzzy package and a
    similarity score is computed using the Levenshtein distance. For each unknown word, a distribution of similiarity
    scores is provided for every known rank. This distribution is later used to predict the most likely rank for the
    given word.

    TODO: ?

    :param unknown_list: This is a vector of words with an unknown quality ranking that we want to analyze and predict.
    :param corpus_list: This is a list of vectors that contain all the words associated with each quality ranking level.
    :param jupyter: This is a boolean that tells us if we are running in a jupyter nb. If so, we use a different tqdm
        progress bar.

    :return: distrib: This is a pandas df with all the similarity scores for each unknown word.
    """
    # import necessary modules
    import pandas as pd
    import numpy as np
    from fuzzywuzzy import fuzz
    from fuzzywuzzy import process

    #note that if we are using a jupyter notebook we want to use a different tqdm call as it displays badly otherwise
    if jupyter == True:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm

    distrib = []

    # loop over each unknown string
    for x in tqdm(range(len(unknown_list)), desc="classifying unknown strings", leave=False):
        unknown_str = unknown_list[x]
        print('analyzing...', unknown_str)

        out = []
        # loop over each corpus to compute similarity scores for all words in a given housing quality score
        for y in range(len(corpus_list)):
            print('~>corpus#', y)
            corpus = corpus_list[y]

            scores = []
            # loop over each word and compute the similarity score
            for z in range(len(corpus)):
                scores.append(fuzz.WRatio(unknown_str, corpus[z]))

            out.append(scores)  # append scores to create a distribution for the entire corpus

        # append distributions of scores
        #TODO: make this call more flexible, currently the colnames are hard coded so this fx will not work for
        #other classification exercises
        distrib.append(pd.DataFrame({'word': unknown_str,
                                     'natural': pd.Series(out[0]),
                                     'rudimentary': pd.Series(out[1]),
                                     'finished': pd.Series(out[2])
                                     # note series method used to overcome differing lengths
                                     }))

    return (pd.concat(distrib))


def fuzzy_predict(df, var_list, grouping, cutoff, dictionary):
    # calculate the probability that a classification score exceeds cutoff
    out = df.groupby(grouping)[var_list].apply(lambda c: (c > cutoff).sum() / len(c))

    # return column w/ max value and map to rank with dictionary
    out['pred'] = out[var_list].idxmax(axis=1).map(dictionary)

    return (out)


def fuzzy_transform(df, var_list, grouping, fx, stub):
    for var in var_list:
        print('calculating prob for...', var)

        kwargs = {var + stub: lambda x: x[var] / x.groupby(grouping)[var].transform(fx)}
        df = df.assign(**kwargs)

    return (df)


def fuzzy_density(df, facet, var_list, color_list, variant="", cutoff=None):
    # import necessary modules
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    g = sns.FacetGrid(df, col=facet, col_wrap=5, height=3)

    for var in var_list:
        ('plotting...', var)
        g = g.map(sns.kdeplot, var + variant, shade=True, color=color_list[var])

        # add cutoff line if provided
        if cutoff != None:
            g = g.map(plt.axvline, x=cutoff, color='grey', linestyle='dashed')

    g = g.add_legend()

    return (g)
