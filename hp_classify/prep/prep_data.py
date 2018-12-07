#define necessary helper functions
def clean_text(text):
    """This function is used to clean a selection of text. 
    It uses several regular expressions and built in text commands in order to remove commonly seen 
    errors,
    nonsense values, 
    punctuation, 
    digits, and 
    extra whitespace.

    Args:
        text (str): This is a text value that needs to be cleaned.

    Returns:
        text: This function returns a cleaned version of the input text.
        
    TODO: Add functionality to impute a selected value for NaN or missing values?

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