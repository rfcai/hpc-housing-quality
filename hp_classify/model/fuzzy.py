
#define function to replace meaningless values with NaNs
# def extract_ranking(df, vars_to_clean):
#     """This helper function is used to 

#     Args:
#     df (pandas df): This is a pandas df that has 
#     dep_var (str): This is the name of a column

#     Returns:
#         df_out: 
        
#     TODO: ?

#     """
        
#     df_out = df.copy()

#     #output a clean dataset
#     return 

def build_corpus(df, str_var, rank_var, rank_list):
    
    #import necessary modules
    import pandas as pd
    import numpy as np
    from fuzzywuzzy import fuzz
    from fuzzywuzzy import process
    
    out = []
    
    for x in rank_list:
        print("building corpus for rank #", x)
        out.append(df[df[rank_var]==x][str_var].values)    

    print("extracting unknown strings")
    other = df[~df[rank_var].isin(rank_list)][str_var].unique()
    other = other[~pd.isnull(other)] #cant classify NaN

    return(out, other)

def fuzzy_scan(unknown_list, corpus_list):
    
    #import necessary modules
    import pandas as pd
    import numpy as np
    from fuzzywuzzy import fuzz
    from fuzzywuzzy import process
    from tqdm import tqdm_notebook

    distrib = []

    #loop over each unknown string
    for x in tqdm_notebook(range(len(unknown_list)), desc="outer loop"): 
        unknown_str = unknown_list[x]
        print('analyzing...', unknown_str)

        out = []
        #loop over each corpus to compute similarity scores for all words in a given housing quality score
        for y in range(len(corpus_list)):
            print('~>corpus#', y)
            corpus = corpus_list[y]


            scores = []
            #loop over each word and compute the similarity score
            for z in tqdm_notebook(range(len(corpus)), desc="inner loop", leave=False): #tqdm=progress bar
                scores.append(fuzz.WRatio(unknown_str, corpus[z]))

            out.append(scores) #append scores to create a distribution for the entire corpus

        #append distributions of scores
        distrib.append(pd.DataFrame({'word': unknown_str, 
                                     'natural':pd.Series(out[0]), 
                                     'rudimentary':pd.Series(out[1]), 
                                     'finished':pd.Series(out[2]) #note series method used to overcome differing lengths
                                    }))


    return(pd.concat(distrib))

def fuzzy_predict(df, var_list, grouping, cutoff, dictionary):
    
    #calculate the probability that a classification score exceeds cutoff
    out = df.groupby(grouping)[var_list].apply(lambda c: (c>cutoff).sum()/len(c))
    
    #return column w/ max value and map to rank with dictionary
    out['pred'] = out[var_list].idxmax(axis=1).map(dictionary) 
    
    return(out)

def fuzzy_transform(df, var_list, grouping, fx, stub):

    for var in var_list:

        print('calculating prob for...', var)

        kwargs = {var+stub : lambda x: x[var]/x.groupby(grouping)[var].transform(fx)}
        df = df.assign(**kwargs)

    return(df)

def fuzzy_density(df, facet, var_list, color_list, variant="", cutoff=None):
    
    #import necessary modules
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    g = sns.FacetGrid(df, col=facet, col_wrap=5, height=3)

    for var in var_list:
        ('plotting...', var)
        g = g.map(sns.kdeplot, var+variant, shade=True, color=color_list[var])
        
        #add cutoff line if provided
        if cutoff != None:
            g = g.map(plt.axvline, x=cutoff, color='grey', linestyle='dashed')
        
    g = g.add_legend()
    
    return(g)