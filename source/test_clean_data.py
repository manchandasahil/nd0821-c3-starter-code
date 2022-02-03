"""
Basic cleaning module test
"""
import pandas as pd
import pytest
import source.clean_data


@pytest.fixture
def data():
    """
    Get dataset
    """
    df = pd.read_csv("data/census.csv", skipinitialspace=True)
    df = source.clean_data.pre_process_dataset(df)
    return df


def test_null(data):
    """
    Data is assumed to have no null values
    """
    assert data.shape == data.dropna().shape


def test_removed_columns(data):
    """
    Data is assumed to have no question marks value
    """
    assert "capital-gain" not in data.columns
    assert "capital-loss" not in data.columns
    assert "fnlgt" not in data.columns
    assert "education-num" not in data.columns
    