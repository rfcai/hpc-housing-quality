
#define necessary helper functions
def cv_censor_col(df, colname, pct=.2, weight_var=None, reps=5):
    
    """This function is used to create pandas dfs where a specified % of the values in a column have been censored
    and replaced with NaN, so that they can be predicted in a cross-validation methodology. It returns a list of such
    dfs that is the length of the reps argument.

    Args:
        df (pandas df): This is a pandas df that has columns with garbage values to be removed.
        colname (str): This is a string indicating the name of a column that you want to censor and later predict.
        pct (float): This is a value between 0-1 that indicates the fraction of values you want to censor. Default = 20%
        weight_var (str): This is a string indicating the column name is used to weighted the sample. Default = No weight.
        reps (int): This is an integer indicating the number of different training datasets to create. Default = 5x

    Returns:
        df_clean: This function returns a pandas df where the garbage codes have been replaced with NaN.
        
    TODO: ?

    """
    
    #import packages
    import pandas as pd
    import numpy as np
    
    out = []
    
    for x in range(reps):
            
        print("sampling df, iteration #", x)
    
        #first archive your old column in order to test later
        new_df = df.copy()
        new_df[colname + '_og'] = new_df[colname]
        df['train'] = 1 #set column to specify whether training or test data

        #draw a weighted sample if weight var is specified
        if weight_var != None:
            df_censor = new_df.sample(frac=pct, weights=weight_var)
        else:
            df_censor = new_df.sample(frac=pct)
            
        #now replace the sampled column with missing values in order to try and predict
        #note that replacement is only done on the sampled indices
        df['train'] = 0
        df_censor[colname] = "replace_me"
        new_df.update(df_censor, overwrite=True)
        new_df[colname].replace("replace_me", np.nan, inplace=True)
        #TODO unsure if this is pythonic method but it seems like df.update won't replace values with NaN, 
        #as such, need to do this workaround
        
        #store the result (df with columns censored)
        out.append(new_df)
    
    #return the list of sampled dfs
    return(out)