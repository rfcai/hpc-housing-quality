def filter_one_word_materials(df, base_var):
    
    """This function takes an input dataframe and returns a subset of it, where all materials described with more than one word
    have been filtered out. As we match materials on their semantic similarity, we want to ensure that we only have materials described
    with one word. Using our semantic similarity function on short text is ambiguous and needs to be further investigated.

    :param df: This is a cleaned dataframe containing all the information from the surveys.
    :param base_var: The variable of interest for which some materials are unknown.

     :return subset: it returns a dataframe containing the subset of rows for which base_var was described with one word.

    """
    
    df_one_word = df[base_var].str.get_dummies(sep=' ').T
    df['count_word'] = df_one_word.sum()
    
    if(1 not in df_one_word.sum()):
        raise NoOneWordException("No material with only one word!")

    subset = df [df.count_word == 1]
        
    return(subset)

def check_if_english(df, base_var):

    """This function goes through a dataframe and verifies if, within the column of interest, all materials correspond to
    actual english words. As we match materials on their semantic similarity, we want to ensure that we drop typos and foreign words.
    This function can be used to subset a dataframe, and only keep observations in english.

    :param df: This is a cleaned dataframe containing all the information from the surveys.
    :param base_var: The variable of interest for which some materials are unknown.

     :return english_material: it returns a list of boolean indicating for each row, whether the material in base_var is described
     using a word in english.

    """
    from nltk.corpus import words
    from itertools import compress

    list_material = df[base_var].unique().tolist()

    boolean = []
    for x in range(len(list_material)):
        boolean.append(list_material[x] in words.words())

    english_material = list(compress(list_material, boolean))

    return(english_material)

def semantic_similarity_scan(unknown_list, corpus_list):

    """This function takes a list of materials for which the rank is unknown (i.e. a word outside our "dictionnary")
    as input and calculates a score of semantic similarity with each word of the list of known material (our "dictionnary").

    :param df: unknown_list: This is a list of strings whose rank is unknown
    :param corpus_list: This is a list of the strings for which the rank is known. The strings are classified
     within one of three categories of materials.

     :return distrib: The distribution of the similarity scores between each unknown material in the unknown list and known material
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
                score = 0
                for s1 in unknw_syn:
                    score = score + sum([s1.path_similarity(s2) for s2 in list_syn if s1.path_similarity(s2) is not None])
                scores.append(score)

            out.append(scores) #append scores to create a distribution for the entire corpus

            #append distributions of scores
        distrib.append(pd.DataFrame({'word': unknown_str,
                                    'natural':pd.Series(out[0]),
                                    'rudimentary':pd.Series(out[1]),
                                    'finished':pd.Series(out[2]) #note series method used to overcome differing lengths
                                    }))
    return(pd.concat(distrib))

def fuzzy_semantic_cv(cv_list, base_var, rank_dictionary, subset=None, threshold=.5):

    #import packages
    from fuzzywuzzy import fuzz
    from fuzzywuzzy import process
    import pandas as pd
    import numpy as np

    #import custom modules
    import semantic.semantic as sem
    import model.fuzzy as fz
    
    #setup objects
    rank_var = base_var + '_rank'
    og_var = rank_var + '_og'
    
    #TODO validate syntax
    rank_values = list(rank_dictionary.values())
    rank_keys = list(rank_dictionary.keys())
    
    #create lists to store loop outputs
    cv_distrib = []
    
    #loop over each cross validation:
    for i in range(len(cv_list)):
        
        print('working on cv loop #', i)
        df = cv_list[i].copy() #subset the cv list to the current df

        #build corpus of known and unknown strings
        str_list, idk_strings = fz.build_corpus(df, base_var, rank_var, rank_values)
        str_list_unique = []
        for x in range(3):
            str_list_unique.append(np.unique(str_list[x]))
        #subset the unknown strings to allow for faster testing
        if subset != None:
            idk_strings = idk_strings[subset]
        
        #find distribution of scores for each string
        distrib = sem.semantic_similarity_scan(idk_strings, str_list_unique)
        
        #append results to prep for next loop
        cv_distrib.append(distrib)

    return(cv_distrib, cv_preds, cv_results, cv_df)

