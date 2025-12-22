import requests


url = "http://localhost:9696/predict"
url1 = "http://localhost:8080/predict"

client = {"job": "management", "duration": 400, "poutcome": "success"}
response = requests.post(url1, json=client).json()

print(response)
