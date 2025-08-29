import requests

sample = {
    "Pclass": 3,
    "Age": 22.0,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Sex_vec": 1.0,
    "Embarked_vec": 0.0
}

response = requests.post("http://localhost:8000/predict", json=sample)
print("Prediction:", response.json())

