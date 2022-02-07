from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome!"}

def test_invalid_post():
    r = client.post("/", json={
        "age": 39,
        "workclass": "ERROR",
        "education": "Some-college",
        "maritalStatus": "ERROR",
        "occupation": "WRONG",
        "relationship": "Husband",
        "race": "Black",
        "sex": "Male",
        "hoursPerWeek": 60,
        "nativeCountry": "United-States"
    })
    assert r.status_code == 422

def test_normal():
    r = client.post("/", json={
            'age': 32,
            'workclass': 'Private',
            'fnlgt': 149184,
            'education': 'HS-grad',
            'marital_status': 'Never-married',
            'occupation': 'Prof-specialty',
            'relationship': 'Not-in-family',
            'race': 'White',
            'sex': 'Male',
            'hoursPerWeek': 60,
            'nativeCountry': 'United-States'
    })
    print(r)
    assert r.status_code == 200
    assert r.json() == {"prediction": "<=50K"}

def test_wrong_url():
    r = client.get("/infer_on_whim")
    assert r.status_code != 200