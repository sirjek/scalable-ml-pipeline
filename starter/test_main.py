from fastapi.testclient import TestClient
from main import app
import unittest

client = TestClient(app)

class TestAPI(unittest.TestCase):
    def test_root(self):
        r = client.get("/")
        assert r.status_code == 200

    def test_inference_valid(self):
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
                "salary": ">50k"
            }

        response = client.post("/inference", json=valid_data)
        self.assertEqual(response.status_code, 200)

    def test_inference_invalid(self):
        invalid_data = {
                "age": "invalid_age",
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
                "salary": ">50k"
            }

        response = client.post("/inference", json=invalid_data)
        self.assertEqual(response.status_code, 422)

if __name__ == "__main__":
    unittest.main()