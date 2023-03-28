import requests

sample_request = {
  "sepal length (cm)": 5.1,
  "sepal width (cm)": 3.5,
  "petal length (cm)": 1.4,
  "petal width (cm)": 0.2
} 

result = requests.post('http://127.0.0.1:5000/api/iris', json=sample_request)

print(result.status_code, result.text)