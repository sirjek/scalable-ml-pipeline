# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
* Model Name: Logistic Regression Classifier for Income Prediction
* Version: 1.0
* Type: Binary Classification
* Input Features:
    * workclass, education, marital-status, occupation, relationship, race, sex, native-country
* Labels:
    * Salary

## Intended Use
* Deploying a ML Model to Cloud Application Platform with FastAPI

## Training Data
* Source: Public dataset based on the U.S. Census.
* Sample Size: 48842

## Evaluation Data
* Source: Same as training data but split into 20% for testing purposes.
* Preprocessing Steps:
    * Categorical features encoded using one-hot encoding.
## Metrics
_Please include the metrics used and your model's performance on those metrics._
* Evaluation Metrics Used:
    * Precision: Measures the proportion of positive identifications that were actually correct.
    * Recall: Measures the proportion of actual positives identified correctly.
    * fbeta: Measures the harmonic mean of precision and recall.
* Model Performance:
    * Precision: 0.728
    * Recall: 0.276
    * fbeta: 0.400


## Ethical Considerations
* Bias in Data:
    * The dataset reflects historical trends and may contain biases related to race, gender, or occupation.
    * Caution should be exercised to ensure predictions do not reinforce existing societal biases.
## Caveats and Recommendations
* The model assumes consistent data quality similar to the training dataset. Poor-quality or incomplete input data may lead to inaccurate predictions.
* Recommendations for improving the model:
    * Perform hyperparameter tuning for better optimization.
    * Explore techniques to mitigate potential bias in the dataset.
