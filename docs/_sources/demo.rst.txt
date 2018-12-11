Demo
=====

`Fuzzy String Classification <http://nbviewer.jupyter.org/github/jfrostad/hpc-housing-quality/blob/master/examples/example_fuzzy.ipynb>`_
-------------------------------------------------------------------------------------------------------------------------------------------
This classifier is built on `fuzzywuzzy` developed by SeatGeek, which accomplishes fuzzy string matching using Levenshtein distance between words. In this example, three corpora are defined using all known words in each rank class. Then, a comparison is made between a given unknown word and all the words in each corpus. This returns the distribution of similarity scores for each class, which is then used to predict the actual class.


`Random Forest Classifier <http://nbviewer.jupyter.org/github/jfrostad/hpc-housing-quality/blob/master/examples/example_rfc.ipynb>`_
---------------------------------------------------------------------------------------------------------------------------------------
This example demonstrates the modeling workflow of the Random Forest Classifier prediction. This is achieved through the following processes. First, we preprocess the data by simple data cleaning, ranking the housing materials, and shuffling the data then redistributing the rank to remove model prediction bias. Then, we train the model using a training dataset, apply the model on the testing dataset, and monitor the accuracy. Within the demo, the user can type in an input and receive an output of the rank prediction.