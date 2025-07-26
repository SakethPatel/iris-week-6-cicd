import requests

url = "http://35.202.110.128/predict"

sample_input = {
    "features": [5.1, 3.5, 1.4, 0.2]
}

try:
    response = requests.post(url, json=sample_input)
    response.raise_for_status()
    print("Status Code:", response.status_code)
    print("Prediction:", response.json())
except requests.exceptions.RequestException as e:
    print("Request failed:", e)
