from .train_model import *

def test_process_data():
    """
    test the pre process of the data.
    """
    df = pd.read_csv("data/cleaned/census_cleaned.csv")
    X, Y, _, _ = process_data(
                df,
                categorical_features=cat_features,
                label="salary", encoder=None, lb=None, training=True)

    assert X.shape[0] == df.shape[0]

def test_evaluate_slices():
    """
    test the evaluate function
    """
    evaluate()
    assert os.path.isfile("./model/slice_output.txt")

def test_evaluate_full():
    results = evaluate_full()
    assert len(results) > 0