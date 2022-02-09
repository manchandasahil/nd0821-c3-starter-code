# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Author: Sahil Manchanda <br>
Gradient Boosting Classifier using the default hyperparameters in scikit-learn.<br>

## Intended Use
It has been used to predict the salary of a person based off a some attributes about it's financials.<br>

## Training / Evaluation Data
source: https://archive.ics.uci.edu/ml/datasets/census+income <br>
train/test split: 75/25 <br>

## Metrics
Metric used: Accuracy <br>
Taining Accuracy: 0.83 <br>

## Ethical Considerations
This model has data points related to origin, gender and race of an individual. Investigation is needed before using it. It could be biased on these counts.

## Caveats and Recommendations
The model could be overfitting in case of certain 'country of origin' as some values have very less
examples of data it was trained on.
