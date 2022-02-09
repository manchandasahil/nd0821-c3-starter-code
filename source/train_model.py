# Script to train machine learning model.
import pandas as pd
import numpy as np
import os
import logging
from numpy import mean, std
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from joblib import dump, load
from .model import compute_model_metrics

# Add the necessary imports for the starter code.

if not os.path.isdir('model/'):
    os.mkdir('model/')
# Add code to load in the data.
# data = pd.read_csv("data/cleaned/census_cleaned.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
# X_train, y_train, encoder, lb = process_data(
#     train, categorical_features=cat_features, label="salary", training=True
# )

# Proces the test data with the process_data function.

def process_data(
    X, categorical_features=[], label=None, training=True, encoder=None,
    lb=None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features
    and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in
    functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in
        `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will
        be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the
        encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the
        binarizer
        passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb

def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.
    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    cv = KFold(n_splits=10, shuffle=True, random_state=1)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    scores = cross_val_score(model, X_train, y_train, scoring='accuracy',
                             cv=cv, n_jobs=-1)
    logging.info('Accuracy: %.2f (%.2f)' % (mean(scores), std(scores)))
    return model

def infer(model, X):
    """ Run model inferences and return the predictions.
    """
    y_preds = model.predict(X)
    return y_preds


def train_test_model():
    """
    Execute model training
    """
    df = pd.read_csv("data/cleaned/census_cleaned.csv")
    train, _ = train_test_split(df, test_size=0.20)

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features,
        label="salary", training=True
    )
    trained_model = train_model(X_train, y_train)

    dump(trained_model, "model/model.joblib")
    dump(encoder, "model/encoder.joblib")
    dump(lb, "model/lb.joblib")

def evaluate():
    if data is None:
        df = pd.read_csv("data/cleaned/census_cleaned.csv")
    else:
        df = data
    _, test = train_test_split(df, test_size=0.20)

    model = load("model/model.joblib")
    encoder = load("model/encoder.joblib")
    lb = load("model/lb.joblib")
    output_slices = []
    for _categories in cat_features:
        for _classes in test[_categories].unique():
            df_temp = test[test[_categories] == _classes]
            X_test, y_test, _, _ = process_data(
                df_temp,
                categorical_features=cat_features,
                label="salary", encoder=encoder, lb=lb, training=False)
            y_preds = model.predict(X_test)
            precision, recall, fbeta = compute_model_metrics(y_test,y_preds)
            results = f"Cat: {_categories}, Precision: {precision}, recall: {recall}, fbeta: {fbeta}\n"
            logging.info(results)
            output_slices.append(results)
    with open('model/slice_output.txt', 'w') as out:
        out.writelines(output_slices)