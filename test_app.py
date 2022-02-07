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

# 28,Private,Bachelors,Married-civ-spouse,Prof-specialty,Wife,Black,Female,40,Cuba,<=50K
def test_normal_below():
    r = client.post("/", json={
        "age": 28,
        "workclass": "Private",
        "education": "Bachelors",
        "maritalStatus": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Wife",
        "race": "Black",
        "sex": "Female",
        "hoursPerWeek": 40,
        "nativeCountry": "Cuba"
    })
    print(r)
    assert r.status_code == 200
    assert r.json() == {"prediction": "<=50K"}

def test_normal_above():
    r = client.post("/", json={
        "age": 32,
        "workclass": "Private",
        "education": "Some-college",
        "maritalStatus": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "hoursPerWeek": 60,
        "nativeCountry": "United-States"
    })
    print(r)
    assert r.status_code == 200
    assert r.json() == {"prediction": ">50K"}

def test_wrong_url():
    r = client.get("/infer_on_whim")
    assert r.status_code != 200