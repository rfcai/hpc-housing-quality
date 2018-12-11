def semantic_similarity_scan(unknown_list, corpus_list):
    """This function takes a list of materials for which the rank is unknown (i.e. a word outside our "dictionnary")
    as input and calculates a score of semantic similarity with each word of the list of known material (our "dictionnary").

    :param unknown_list: This is a list of strings whose rank is unknown
    :param corpus_list: This is a list of the strings for which the rank is known. The strings are classified
        within one of three categories of materials.
    :return: distrib (DataFrame): The distribution of the similarity scores between each unknown material in the unknown list and known material
        in the corpus_list.
    """

    import pandas as pd
    import numpy as np
    import nltk
    from nltk.corpus import wordnet as wn
    from fuzzywuzzy import fuzz
    from fuzzywuzzy import process

    distrib = []

        #loop over each unknown string
    for x in range(len(unknown_list)):
        unknown_str = unknown_list[x]
        print('analyzing...', unknown_str)
        unknw_syn = wn.synsets(unknown_str)
        out = []
        #loop over each corpus to compute similarity scores for all words in a given housing quality score
        for y in range(len(corpus_list)):
            print('~>corpus#', y)
            corpus = corpus_list[y]


            scores = []
                #loop over each word and compute the similarity score
            for z in range(len(corpus)): #tqdm=progress bar
                list_syn = wn.synsets(corpus[z])
                for s1 in unknw_syn:
                    score = [s1.path_similarity(s2) for s2 in list_syn if s1.path_similarity(s2) is not None]
                scores.append(sum(score))

            out.append(scores) #append scores to create a distribution for the entire corpus

            #append distributions of scores
        distrib.append(pd.DataFrame({'word': unknown_str,
                                    'natural':pd.Series(out[0]),
                                    'rudimentary':pd.Series(out[1]),
                                    'finished':pd.Series(out[2]) #note series method used to overcome differing lengths
                                    }))
    return(pd.concat(distrib))
