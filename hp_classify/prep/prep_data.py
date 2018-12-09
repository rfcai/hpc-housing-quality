import timeit
import numpy as np
import pandas as pd

#define necessary helper functions
def clean_text(text):
    """This function is used to clean a selection of text.
    It uses several regular expressions and built in text commands in order to remove commonly seen
    errors,
    nonsense values,
    punctuation,
    digits, and
    extra whitespace.
    TODO: Add functionality to impute a selected value for NaN or missing values?

    :param text (str): This is a text value that needs to be cleaned.

    :return: text: This function returns a cleaned version of the input text.
    """
    #import necessary modules
    import re
    
    #force all vals in series to string
    text = str(text)
    
    #first remove uppercase
    text = text.lower()
    
    #remove common errors
    text = re.sub(r"\[.]", "", text) 
    text = re.sub(r"\<ff>", "", text)   
    text = re.sub(r"\<fb>", "", text)
    text = re.sub(r"\<a\d>", "", text)   
    text = re.sub(r"\<c\d>", "", text)   
    text = re.sub(r"\<d\d>", "", text)
    text = re.sub(r"\<e\d>", "", text)   
    text = re.sub(r"\<f\d>", "", text)   
    text = re.sub(r"\d+\.", "", text)

    # remove the characters [\], ['] and ["]
    text = re.sub(r"\\", "", text)    
    text = re.sub(r"\'", "", text)    
    text = re.sub(r"\"", "", text)   

    # replace punctuation characters with spaces
    filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)
    
    # remove any remaining digit codes
    text = re.sub(r"\d+", "", text)
    
    # remove any leading/trailing/duplicate whitespace
    text = re.sub(' +', ' ', text.strip())
    
    return text
    
#define master function
def read_then_clean(file_path, vars_to_clean, filter_series=None):
    """This is the master function for this module. It uses the previously defined helper functions,
    in order to output a clean dataset for user. It reads in a selected .csv file from a given filepath,
    and applies the previously defined cleaning functions to a list of variables provided by user.
    
    It can also optionally filter the df based on the survey series or TODO language.

    Args:
        file_path (str): This is a string indicating which file that you want to read in.
        vars_to_clean (list): This is a list of strings that indicate which columns you want to clean.
        filter_series (list): This is a list of strings that indicate which survey series to keep.

    Returns:
        df_clean: This is a pandas df that has columns of text values that have been cleaned using the helper function.
        
    TODO: Is it better to return an obj called df_clean to be more explicit to user?

    """
    #import necessary modules
    import pandas as pd
    import numpy as np
    
    #read in your data
    print("~begin reading")
    df_raw = pd.read_csv(file_path, low_memory=False)
    min_nrow = len(df_raw) #save the row count to test after cleaning and verify that rows are not being dropped
    print("data read!")
    
    #cleanup
    print("~begin cleaning")
    df_clean = df_raw.copy()
    for var in vars_to_clean:
        df_clean[var] = df_clean[var].apply(clean_text)
    print("data clean!")
    
    # Verify that the minimum rowcount continues to be met
    if len(df_clean) < min_nrow:
        class RowCountException(Exception):
            """Custom exception class.
            
            This exception is raised when the minimum row is unmet.

            """
            pass
        
        raise RowCountException("Minimum number of rows were not returned after cleaning. Data is being lost!")
        
    # Filter data if filter arguments are provided by user
    if filter_series != None:
        print("~applying filter")
        df_clean = df_clean[df_clean['survey_series'].isin(filter_series)]
        
    #output a clean dataset
    return df_clean

#define function to replace meaningless values with NaNs
def remove_garbage_codes(df, vars_to_clean, garbage_list):
    """This helper function is used to remove garbage values from a pandas df, replacing them with NaN.

    Args:
    df (pandas df): This is a pandas df that has columns with garbage values to be removed.
    vars_to_clean (list): This is a list of strings that indicate which columns you want to clean.
    garbage_list (list): This is a list of strings that indicate which garbage values to replace with NaN

    Returns:
        df_clean: This function returns a pandas df where the garbage codes have been replaced with NaN.
        
    TODO: ?

    """
    
    #import necessary modules
    import pandas as pd
    import numpy as np
    
    df_clean = df.copy()
    
    # build dictionary to map all garbage values to NaN
    garb_dict = {}
    for string in garbage_list:
        garb_dict[string] = np.nan
    
    print(garb_dict)
    
    for var in vars_to_clean:
        print("removing garbage from ", var)
        df_clean[var].replace(garb_dict, inplace=True)
        
    #output a clean dataset
    return df_clean

#define function to replace meaningless values with NaNs
def extract_ranking(df, vars_to_clean):
    """This helper function is used to extract the ordinal rankings from numerical coding.

    Args:
    df (pandas df): This is a pandas df that has columns with garbage values to be removed.
    vars_to_rank (list): This is a list of strings that indicate which columns you want to extract ranks from.

    Returns:
        df_out: This function returns a pandas df with new vars added with the ordinal rank cols defined.
        
    TODO: ?

    """
    
    #import necessary modules
    import pandas as pd
    import numpy as np
    import re
    
    df_out = df.copy()
    
    for var in vars_to_clean:
        print("defining ranking for ", var)
        newcol = re.sub("_num", "_rank", var) 
        df_out[newcol] = df_out[var].astype(str).str[0]

    #output a clean dataset
    return df_out

def load_data(file_name):
    """This helper function is used to load a dataset from a selected csv file. 
    It monitors the rumtime of the loading process and also truncates the dataset
    to six specified columns as the training atrributes. values with 0 within the numbers
    of roof, wall, floor are filtered out as well as rows containing nan.
    
    Args:
        file_name (.csv): This is the file name of the csv file to load in for processing
    
    Returns:
        df: This function returns a dataframe with specifed columns and no missing data
    """    
    #define start time
    start = timeit.default_timer()
    
    df = pd.read_csv(file_name,low_memory=False) 
    #truncate process if the dataframe to specified columns and remove missing data
    attr = ['int_year',
            'housing_roof_num','housing_wall_num','housing_floor_num','iso3']

    df = df[attr]
    df = df[df['housing_wall_num']!=0]
    df = df[df['housing_roof_num']!=0] 
    df = df[df['housing_floor_num']!=0] 
    df.dropna()
    
    #define end time
    stop = timeit.default_timer()
    print("Runtime:{:.2f} sec".format(stop - start))
    
    return df


def cleaning_block(df, word):
    """
    This helper function filters out special characters within number columns, 
    and convert numbers from string to int
    
    Args:
        df (pandas df):This is a dataframe that is read in and munipulated to 
        clean out special characters '.' and '?' within the number columns and 
        covert the data type of the numbers from string to integer
    """ 
    df.loc[:,'housing_'+word+'_num'] = df.loc[:,'housing_'+word+'_num'].replace('?',None)
    df.loc[:,'housing_'+word+'_num'] = df.loc[:,'housing_'+word+'_num'].replace('.',None)
    df.loc[:,'housing_'+word+'_num'] = pd.to_numeric(df.loc[:,'housing_'+word+'_num'])
    
def ranking(df, vals):
    """
    This helper function ranks number columns by the ten digit,
    then keeps values within [10,35], cause the values 
    out of this set is considered as missing values.
    
    Args:
        df (pandas df): This is a dataframe read in be ranked
        vals: This is a list of category names (eg. roof, wall, floor) 
        that is specified to be ranked
    Return:
        df (pandas df): This function returns a pandas df with the ordinal rank
        cols added to the read in datafame
        
    """
    for val in vals:
        cleaning_block(df, val)
        df = df[(df['housing_'+val+'_num']<=35) & (df['housing_'+val+'_num']>=10)]
        df.loc[:,val+'_rank'] = (df['housing_'+val+'_num']/10).apply(np.floor)
        df.loc[df[val+'_rank'] == 0, val+'_rank'] = 0
        df.loc[df[val+'_rank'] >= 4, val+'_rank'] = 4
    return df

def extract_features(df, LABEL):
    """
    This is a helper function that gives a list of features used for model training
    
    Args:
        df (pandas df): This is a dataframe read in for feature extraction
        LABEL: This is the label specifed under prediction
        
    Return:
        features : This is a list of features with label(answers) columns dropped
    """
    #list features
    features = list(df)
    features.remove('housing_'+LABEL+'_num')
    features.remove(LABEL+'_rank')
    return features

def shuffle_redistribute(df, LABEL, Redistribute=True):
    """This helper function greneralizes the distribution of the data set
    with respect to the three ranks (1,2,3), then it shuffles the data set to 
    unbias.
    
    Args:
        df (pandas df): This is a dtaframe read in to shuffle and redistribute
        LABEL: This is the label specifed under prediction
        Redistribute: set True for data redistribution (default True)
    Return:
        df (pandas df): Returns the modified dataframe
        rank_dist: This is a list of the number of data within each rank(1,2,3)
    """
    if Redistribute:
        #split the dataframe to three subsets grouped by rank 1,2,3,
        rank1 = df[df[LABEL+'_rank']==1]
        rank2 = df[df[LABEL+'_rank']==2]
        rank3 = df[df[LABEL+'_rank']==3]
        
        list_for_rank_df = [rank1, rank2, rank3]
        new_list = []
        
        #determine the smallest dataframe grouped by rank 1,2,3
        base = min(len(rank1),len(rank2),len(rank3))
        
        #redistribution procees by dividing the portion based on the smallest dataset
        for rank in list_for_rank_df:
            if len(rank) != base:
                rank.loc[:,'rand_num'] = np.random.uniform(0,1, len(rank)) 
                rank = rank[rank['rand_num']<= base/len(rank)]
                rank = rank.drop(columns = 'rand_num')
            new_list.append(rank)
        #concat the three sub dataframes
        df = pd.concat(new_list)
    # count the total amount of data of each rank 1,2,3
    rank_dist = df.groupby(LABEL+'_rank').size().tolist()
    #randomly shuffle the rows of the overall dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    return df, rank_dist

def train_test_split(df, features, LABEL):
    """This function splits the dataframe to random testing and training set 
    with 75% and 25%each. 
    
    Args:
        df (pandas df): This is the dataframe to be split
        features: This is the list of splecified features
        LABEL: This is the label specifed under prediction
        
    Return:
        x_train: This is the returned train dataset
        x_test: This is the returned test dataset
        y_train: This is the returned training label
        y_test: This is the returned testing label
    """
    #assign a random float from 0~1 to each row with uniform distribution
    df['train'] = np.random.uniform(0, 1, len(df)) <= .75
    #label encoding, map each categorical data to a coresponding number
    df['iso3'] = pd.factorize(df['iso3'])[0]
    #detemine trani and test set, True = training set and False = test set
    train = df[df['train']==True]
    test = df[df['train']==False]
    #dataset and feature splitting for training nd testing
    x_train = train[features]
    x_test = test[features]
    y_train = train[LABEL+'_rank'].values.astype(int)
    y_test = test[LABEL+'_rank'].values.astype(int)
    return x_train, x_test, y_train, y_test
    
    