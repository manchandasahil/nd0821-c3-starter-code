import requests

data = {
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
    }

r = requests.post('https://sahil-census-app.herokuapp.com/', json=data)

# assert r.status_code == 200

print(f"Response code: {r.status_code}")
print(f"Response body: {r.json()}")