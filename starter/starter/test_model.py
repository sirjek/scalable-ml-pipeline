import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression
from starter.ml.model import compute_model_metrics, inference
from starter.train_model import save_model


class TestModelFunctions(unittest.TestCase):
    def test_train_model(self):

        X_train, y_train, X_test, y_test, model = save_model()
        self.assertIsInstance(model, LogisticRegression)

    def test_compute_model_metrics(self):

        X_train, y_train, X_test, y_test, model = save_model()
        preds = inference(model, X_test)
        precision, recall, fbeta = compute_model_metrics(y_test, preds)

        self.assertIsInstance(precision, float)
        self.assertIsInstance(recall, float)
        self.assertIsInstance(fbeta, float)

    def test_inference(self):

        X_train, y_train, X_test, y_test, model = save_model()
        preds = inference(model, X_test)

        self.assertIsInstance(preds, np.ndarray)


if __name__ == '__main__':
    unittest.main()
