def fuzzy_cv(cv_list, base_var, rank_dictionary, subset=None, threshold=75, jupyter=False):
    """This is the master function for this module. It is used to loop over our list of randomly sampled
    cross-validation dfs and run the fuzzy prediction pipeline on them in order to return results and accuracy metrics
    for each. It reads in our custom fuzzy module in order to use its functions in sequence on each cv run.

    TODO: ?

    :param cv_list: This is a list of pandas df, each containing a different cross-validation run.
    :param base_var: This is a string indicating the variable you want to analyze the string values of and predict rank
    :param rank_dictionary: This is a dictionary that can be used to map the str names of the ranks back to ordinal vals
    :param subset: This is an optional parameter that can be used to subset our list of unknown words for testing
    :param threshold: This is the similarity score threshold, above which we think implies sufficient semantic meaning
    in word similarity to accurately predict the words quality ranking.
    :param jupyter: This is a boolean that tells us if we are running in a jupyter nb. If so, we use a different tqdm
        progress bar.

    :return: cv_distrib: This is a list of len=len(cv_list), containing pandas dfs that have the distributions of scores
     for each unknown word
    :return: cv_preds: This is a list of len=len(cv_list), containing pandas dfs that have the prediction for each word
    based on the distributions of scores
    :return: cv_results: This is a list of len=len(cv_list), containing pandas crosstabs that indicate the accuracy
    score result for each cv run
    :return: cv_df: This is a list of len=len(cv_list), containing pandas dfs in which each of the unknown words has a
    prediction column added based on the fuzzy model process
    """

    #import packages
    from fuzzywuzzy import fuzz
    from fuzzywuzzy import process
    import pandas as pd
    import numpy as np
    
    if jupyter == True:
        from tqdm import tqdm_notebook as tqdm
    else: 
        from tqdm import tqdm as tqdm
    
    #import custom modules
    import sys
    sys.path.append('../hp_classify')
    import model.fuzzy as fz
    
    #setup objects
    rank_var = base_var + '_rank'
    og_var = rank_var + '_og'
    
    #TODO validate syntax
    rank_values = list(rank_dictionary.values())
    rank_keys = list(rank_dictionary.keys())

    # initialize lists to store loop vals
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
        
        #subset the unknown strings to allow for faster testing
        if subset != None:
            idk_strings = idk_strings[subset]
        
        #find distribution of scores for each string
        distrib = fz.fuzzy_scan(idk_strings, str_list, jupyter=jupyter)
        
        #TODO, output plots of distribution for analysis
        
        #predict class based on probability of exceeding similarity cutoff
        preds = fz.fuzzy_predict(distrib, rank_keys, 'word', threshold,
                                 rank_dictionary)

        #merge results back on the test data to validate
        train = df[df['train']==0]
        out = pd.merge(train,
                       preds,
                       left_on=base_var,
                       right_on='word',
                       how='left')

        # Verify that rows have neither been added or lost by merging on predictions
        if len(train) != len(out):
            class RowCountException(Exception):
                """Custom exception class.

                This exception is raised when the rowcount is not as expected.

                """
                pass

            raise RowCountException("Rowcount was modified by merge, output df is no longer representative")

        #calculate success rate and tabulate
        out['success'] = np.where(out[og_var] == out['pred'], 1, 0)
        success_rate = pd.crosstab(out[~pd.isnull(out['pred'])]['success'], columns='count')
        
        #append results to prep for next loop
        cv_distrib.append(distrib)
        cv_preds.append(preds)
        cv_results.append(success_rate)
        cv_df.append(out)
        
    return(cv_distrib, cv_preds, cv_results, cv_df)


def save_results_df(df_list, out_dir, out_name):
    """This is a helper function for this module. It is used to save the generated results to csv.

    TODO: Move this function to a more general module?

    :param df: This is the input pandas df that we want to save
    :param out_dir: This is the directory that we want to save our csv file into
    :param out_name: This is the name that we want to call our csv file

    :return: out_path: This returns the path of the saved file to the user for review.
    """

    #concatenate the path using the dir and name of choice
    out_path = f'{out_dir}//{out_name}.csv'    
    print('saving df to', out_path)
    
    df = pd.concat(df_list) #concat the results list (over all cv runs) into a single pandas df
    df.to_csv(out_path, header=False, sep=';')
    
    return(out_path)
