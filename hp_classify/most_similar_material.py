# %load most_similar_material.py
def most_similar_material(input_word, vocabulary):
    """This function takes an "unknown"  word (i.e. a word outside our "dictionnary")
    as input and look for the most similar word within a list of words (our "dictionnary").

    :param input_word: This is a string whose meaning is ambiguous and therefore needs to be matched with a word
        within our dictionnary.
    :param vocabulary: This is a list of the words included in our vocabulary, i.e we know their rank
    :return: most_similar_word: This function returns the most similar word within our list of words.
    """
    #import packages
    import nltk
    from nltk.corpus import wordnet as wn

    # Verify that the input_word is a string
    if type(input_word) is not str:
        class TypeException(Exception):
            """Custom exception class.
            This exception is raised when the input word is not a string.
            """
            pass

        raise TypeException("The input word is not a string!")

    # first, store each word of the vocabulary and its different meanings in a dictionary
    vocabulary_wn = []
    d = {}
    for x in vocabulary:
        vocabulary_wn.append(wn.synsets(x))
        d[x] = wn.synsets(x)

    # then, we define a function returning a key when we call its value
    def getKeysByValue(dictOfElements, valueToFind):
        listOfKeys = list()
        listOfItems = dictOfElements.items()
        for item  in listOfItems:
            if item[1] == valueToFind:
                listOfKeys.append(item[0])
        return  listOfKeys

    # finally, we loop through the different meanings of our new word
    # we compare them to the different meanings of our vocabulary
    s_list = []
    s1 = wn.synsets(input_word)
    maximum = 0
    synonym = None
    for s2 in vocabulary_wn:
        for word1 in s1:
            best = [word1.path_similarity(word2) for word2 in s2 if word1.path_similarity(word2) is not None]
            b = pd.Series(best).max()
            s_list.append(b)
            if max(s_list) > maximum:
                maximum = max(s_list)
                synonym = s2

    # We want to make sure we actually found a synonym
    if synonym is None:
        class NoneException(Exception):
            """Custom exception class.
            This exception is raised when there is no synonym found.
            """
            pass

        raise NoneException("No synonym found for this material")

    # We return the word with the closest meaning according to wod2vec
    return(getKeysByValue(d,synonym))
