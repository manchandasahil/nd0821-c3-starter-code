import pandas as pd
import os

data_path = "data/census.csv"
data_clean_path = "data/cleaned/"

if not os.path.isdir(data_clean_path):
    os.mkdir(data_clean_path)

def pre_process_dataset(df):
    """
    Remove highly correlated columns and 
    columns that are zero, as found in our eda
    Arguments:
    df : the dataframe to be cleaned
    """
    df.drop("education-num", axis="columns", inplace=True)
    df.drop("fnlgt", axis="columns", inplace=True)
    df.drop("capital-gain", axis="columns", inplace=True)
    df.drop("capital-loss", axis="columns", inplace=True)

    df.replace({'?': None}, inplace=True)
    df.dropna(inplace=True)
    
    return df


def clean_data():
    """
    Execute data cleaning
    """
    df = pd.read_csv(data_path, skipinitialspace=True)
    df = pre_process_dataset(df)
    df.to_csv(data_clean_path + "census.csv", index=False)