
#define necessary helper functions
def cv_censor_col(df, colname, pct, weight_var, reps):
    
    #import packages
    import pandas as pd
    import numpy as np
    
    out = []
    
    for x in range(reps):
            
        print("sampling df, iteration #", x)
    
        #first archive your old column in order to test later
        new_df = df.copy()
        new_df[colname + '_og'] = new_df[colname]

        #draw a weighted sample
        df_censor = new_df.sample(frac=pct, weights=weight_var)

        #now replace the sampled column with missing values in order to try and predict
        #note that replacement is only done on the sampled indices
        df_censor[colname] = "replace_me"
        new_df.update(df_censor, overwrite=True)
        new_df[colname].replace("replace_me", np.nan, inplace=True)
        #TODO unsure if this is pythonic method but it seems like df.update won't replace values with NaN, 
        #as such, need to do this workaround
        
        #store the result (df with columns censored)
        out.append(new_df)
    
    #return the list of sampled dfs
    return(out)