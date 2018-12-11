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

    out = [] #initialize list to store loop vals

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

    distrib = [] #initialize list to store loop vals

    # loop over each unknown string
    for x in tqdm(range(len(unknown_list)), desc="classifying unknown strings", leave=False):
        unknown_str = unknown_list[x] #subset the list of strings to the current working string
        print('analyzing...', unknown_str)

        out = [] #initialize list to store loop vals
        # loop over each corpus to compute similarity scores for all words in a given housing quality score
        for y in range(len(corpus_list)):
            print('~>corpus#', y)
            corpus = corpus_list[y] #subset the list of corpora to the given corpus to compare against unknown word

            scores = [] #initialize list to store loop vals

<<<<<<< HEAD
            # loop over each word and compute the similarity score using fuzzywuzzy
            for z in range(len(corpus)):
=======
            scores = []
            #loop over each word and compute the similarity score
            for z in range(len(corpus)): #tqdm=progress bar
>>>>>>> last progress on semantic similarity
                scores.append(fuzz.WRatio(unknown_str, corpus[z]))

            out.append(scores) # append scores to create a distribution for the entire corpus

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
<<<<<<< HEAD
    """This is a helper function for this module. It is used to predict the most likely ranking level for a given string
    based on the distribution of its similarity scores against each corpus from each ranking level. The cutoff level is
    used to determine which rank to predict, as the prediction will be based on which ranking level has the highest
    probability (number of instances > cutoff / total number) of being above the cutoff.
=======
    
    #calculate the probability that a classification score exceeds cutoff
    ret = df.groupby(grouping)[var_list].apply(lambda c: (c>cutoff).sum()/len(c))
    
    #return column w/ max value and map to rank with dictionary
    ret['pred'] = ret[var_list].idxmax(axis=1).map(dictionary) 
    
    return(ret)
>>>>>>> last progress on semantic similarity

    TODO: ?

    :param df: This is the input pandas df of words we wanted to predict for using the words that are known per rank.
    :param var_list: This is a list of str column names, each containing the distribution of values for a given rank.
    :param grouping: This is a str value that specifies the column name that each distribtuion is grouped by.
    :param cutoff: This is the similarity score cutoff we think implies sufficient semantic meaning in word similarity.
    :param dictionary: This is a dictionary we can use to transform the column names back into an ordinal rank values.

    :return: out: This is a pandas df which is a copy of the input df, but has a new column added of predicted rank.
    """
    # calculate the probability that a classification score exceeds cutoff
    out = df.groupby(grouping)[var_list].apply(lambda c: (c > cutoff).sum() / len(c))

    # return column w/ max value and map to rank with dictionary
    out['pred'] = out[var_list].idxmax(axis=1).map(dictionary)

    return (out)

def fuzzy_density(df, facet, var_list, color_list, variant="", cutoff=None):
    """This is a helper function for this module. It is used to generate density plots showing distributions of scores
    for each word, with the colors indicating each different quality ranking. A cutoff argument can be passed to draw a
    vertical line on the plot to indicate which probability mass is going to be used to derive the predictions.

    TODO: ?

    :param df: This is the input pandas df of score distributions by quality ranking for each unknown word
    :param facet: This is a str specifying the column name that we want to facet on for our plots.
    :param var_list: This is a list of str column names, each containing the distribution of values for a given rank.
    :param color_list: This is a dictionary of color values, defined by the column name for each quality rank.
    :param cutoff: This is the similarity score cutoff we think implies sufficient semantic meaning in word similarity.
    :param variant: This is an optional parameter that can be used if we transform the score column before plotting.

    :return: g: This is an object containing the generated plots.
    """
    # import necessary modules
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    #specify the facet grid structure and the variable to use for the facetting
    g = sns.FacetGrid(df, col=facet, col_wrap=5, height=3)

    for var in var_list:
        ('plotting...', var)
<<<<<<< HEAD
        #create a density plot using seaborn with the specified color mapping
        g = g.map(sns.kdeplot, var + variant, shade=True, color=color_list[var])

        # add cutoff line if provided
=======
        g = g.map(sns.kdeplot, var+variant, shade=True, color=color_list[var])
        #add cutoff line if provided
>>>>>>> last progress on semantic similarity
        if cutoff != None:
            g = g.map(plt.axvline, x=cutoff, color='grey', linestyle='dashed')

    #add the legend for colors
    g = g.add_legend()
<<<<<<< HEAD

    return (g)
=======
    
    return(g)
>>>>>>> last progress on semantic similarity
