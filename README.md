# ADnEV-PyTorch

This repository present a pytorch implementation od the ADnEV algorithm present in [ADnEV: Cross-Domain Schema Matching using Deep Similarity Matrix Adjustment and Evaluation](http://www.vldb.org/pvldb/vol13/p1401-shraga.pdf)

## pre-requirements
- Python 3.7 or higher.
- packages from requirements.txt in this repo.
- Highly recommended- machine with gpu (google colab did the trick for me).

## config.py
- contains an array of the csv filed. Each csv file contains the following columns:
  + instance- the schema pair ID.
  + alg- the algorithm that provide the matching between the schemas.
  + candName- the name of the attribute of the first schema (e.g table_name.attribute_name).
  + targetName- the same as above.
  + conf- the confidence level (value between 0 to 1) of the algorithm in the attribute matching.
  + realConf- 1 if the attributes are really matched together and 0 otherwise. 
- The code is suitable for the case that each csv file contains only one pair. it can be easily customized to a different input configuration. 

## Data_prep.py
- Create the Dataset from the csv files. Contain a __getitem__ function that returns the algorithm output matrix, the real matrix, and a dictionary of scores for that matching (precision, recall, F1 and cosine similarity).

## ADnEV.py
- Contains the full algorithm implementation- attempt to improve the input matrix and predict it's score.

## Adaptor.py
- Contain a full implementation of the network that only attempt to improve the input matrix.

## Evaluator.py
- Contain a full implementation of the network that only attempt to predict the input matrix's score.

