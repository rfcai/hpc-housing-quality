Software Components
===================

Database manager
----------------
Simplified interface to access the databases containing the raw input data and the output predictions of ordinal score for housing quality
This component will be responsible for data pre-processing and for building representative test and training datasets in order to inform a cross-validation modelling approach

Model manager
-------------
Accesses pre-processed input data as training/test subdata and uses ML/NLP toolkits in order to analyze relationships between string keywords and ordinal values relating to quality


Visualization manager (coming soon)
-----------------------------------
Accesses intermediate and final outputs in order to create maps of housing quality and visualize the networks resulting from interdependencies across the universe of potential string and ordinal values


Prediction manager (coming soon)
--------------------------------
Input the characteristics of housing (material of the roof/walls/floor), and results of model built/optimized using training dataset, then produces as output an ordinal score for housing quality