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
        
    TODO: Add functionality to input a selected value for NaN or missing values?

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