"""
A script to test the deployed app on heroku.
"""

import requests

data = {
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
    }

r = requests.post('https://sahil-census-app.herokuapp.com/', json=data)

assert r.status_code == 200

print(f"Response code: {r.status_code}")
print(f"Response body: {r.json()}")