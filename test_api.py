import requests


BASE_URL = "https://fastapi-ml-pipeline-app-28cb55f41122.herokuapp.com/"


def test_get():
    response = requests.get(f"{BASE_URL}/")
    print(f"GET Response: {response.status_code}, {response.json()}")


def test_post():
    data = {
        "age": 30,
        "workclass": "Private",
        "fnlgt": 123456,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
        "salary": ">50k"
    }
    response = requests.post(f"{BASE_URL}/inference", json=data)
    print(f"POST Response: {response.status_code}, {response.json()}")


if __name__ == "__main__":
    test_get()
    test_post()
