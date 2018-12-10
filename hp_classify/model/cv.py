
def fuzzy_cv(cv_list, base_var, rank_dictionary, threshold=75, jupyter=False):

    #import packages
    from fuzzywuzzy import fuzz
    from fuzzywuzzy import process
    import pandas as pd
    import numpy as np

    if jupyter == True:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm

    #import custom modules
    import model.fuzzy as fz
    
    #setup objects
    rank_var = base_var + '_rank'
    og_var = rank_var + '_og'
    
    #TODO validate syntax
    rank_values = list(rank_dictionary.values())
    rank_keys = list(rank_dictionary.keys())
    
    #create lists to store loop outputs
    cv_distrib = []
    cv_preds = []
    cv_results = []
    cv_df = []
    
    #loop over each cross validation:
    for i in tqdm(range(len(cv_list)), desc="cv loop"):
        
        print('working on cv loop #', i)
        df = cv_list[i].copy() #subset the cv list to the current df

        #build corpus of known and unknown strings
        str_list, idk_strings = fz.build_corpus(df, base_var, rank_var, rank_values)
        
        #find distribution of scores for each string
        distrib = fz.fuzzy_scan(idk_strings[1:4], str_list)
        
        #TODO, output plots of distribution for analysis

        #predict class based on probability of exceeding similarity cutoff
        preds = fuzzy_predict(distrib, rank_keys, 'word', threshold,
                                var_dictionary)

        #merge results back on the test data to validate
        out = df[df['train']==0]
        out = pd.merge(out,
                       preds,
                       left_on=base_var,
                       right_on='word',
                       how='left')

        #calculate success rate and tabulate
        out['success'] = np.where(out[og_var] == out['pred'], 1, 0)
        success_rate = pd.crosstab(out[~pd.isnull(out['pred'])]['success'], columns='count')
        
        #append results to prep for next loop
        cv_distrib.append(distrib)
        cv_preds.append(preds)
        cv_results.append(success_rate)
        cv_df.append(out)
        
    return(cv_distrib, cv_preds, cv_results, cv_df)