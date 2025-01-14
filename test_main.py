from fastapi.testclient import TestClient
from main import app
import unittest

client = TestClient(app)


class TestAPI(unittest.TestCase):
    def test_root(self):
        r = client.get("/")
        assert r.status_code == 200

    def test_predict_below_50k(self):
        valid_data = {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
            }

        response = client.post("/inference", json=valid_data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"prediction": [" <=50K"]})

    def test_predict_above_50k(self):
        valid_data = {
            "age": 35,
            "workclass": "Private",
            "fnlgt": 234721,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital-gain": 14000,
            "capital-loss": 0,
            "hours-per-week": 50,
            "native-country": "United-States"
              }

        response = client.post("/inference", json=valid_data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"prediction": [" >50K"]})


if __name__ == "__main__":
    unittest.main()
