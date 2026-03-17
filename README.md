# Titanic Survival Prediction — Logistic Regression Case Study

![Python](https://img.shields.io/badge/Python-3.x-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Algorithm](https://img.shields.io/badge/Algorithm-Logistic%20Regression-purple)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A complete end-to-end machine learning project that predicts whether a Titanic passenger survived or not, based on passenger details such as age, gender, fare, and embarkation point. The project uses **Logistic Regression** and includes a full data preprocessing pipeline before training.

---

## Problem Statement

Given passenger information from the Titanic dataset, can a machine learning model predict whether a passenger survived the disaster?

- Output 1 — Survived
- Output 0 — Did Not Survive

---

## Project Workflow

| Step | Function | Description |
|------|----------|-------------|
| 1 | `TitanicLogisticReg` | Load the dataset from CSV |
| 2 | `ShowData` | Display raw dataset info — shape, columns, missing values |
| 3 | `CleanTitanicData` | Full preprocessing pipeline (details below) |
| 4 | `TrainTitanicModel` | Split data, train model, display coefficients |
| 5 | `PreserveModel` | Save the trained model to a `.pkl` file |

---

## Data Preprocessing Pipeline

The raw Titanic dataset requires significant cleaning before training. The following preprocessing steps are applied inside `CleanTitanicData`:

| Task | Details |
|------|---------|
| Drop unnecessary columns | Removes `PassengerId`, `Name`, `Cabin`, `zero` |
| Handle Age column | Converts to numeric, fills missing values with median |
| Handle Fare column | Converts to numeric, fills missing values with median |
| Handle Embarked column | Fills missing values with mode (most frequent value) |
| Handle Sex column | Converts to numeric format |
| Encode Embarked column | Applies One-Hot Encoding using `pd.get_dummies` |
| Convert boolean columns | Converts True/False columns to 1/0 integers |

---

## Dataset

**File:** `MarvellousTitanicDataset.csv`

**Features (Independent Variables - X):**
- `Pclass` — Passenger class (1st, 2nd, 3rd)
- `Sex` — Gender (encoded as numeric)
- `Age` — Age of passenger
- `SibSp` — Number of siblings/spouses aboard
- `Parch` — Number of parents/children aboard
- `Fare` — Ticket fare
- `Embarked` — Port of embarkation (One-Hot Encoded)

**Target (Dependent Variable - Y):**
- `Survived` — 1 (Survived) or 0 (Did Not Survive)

---

## Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | Logistic Regression |
| Max Iterations | 1000 |
| Train/Test Split | 80% / 20% |
| Model Saved As | `marvellousTitanic.pkl` |
| Library | scikit-learn |

---

## Model Preservation

After training, the model is saved to disk using `joblib` as a `.pkl` file. This allows the model to be reloaded later for making predictions on new passenger data without retraining.

```python
import joblib
model = joblib.load("marvellousTitanic.pkl")
result = model.predict([[3, 0, 22, 1, 0, 7.25, 0, 1]])
```

---

## Evaluation Metrics

- **Accuracy Score** — overall percentage of correct predictions
- **Confusion Matrix** — breakdown of correct vs incorrect predictions
- **Model Coefficients** — shows how much each feature influences the survival prediction

---

## Tech Stack

- Python 3
- pandas — data loading, cleaning, and encoding
- numpy — handling invalid/missing numeric values
- scikit-learn — model training and evaluation
- joblib — saving and loading the trained model

---

## How to Run

1. Clone this repository
2. Place `MarvellousTitanicDataset.csv` in the same folder as the script
3. Install the required libraries:
   ```bash
   pip install pandas numpy scikit-learn joblib
   ```
4. Run the script:
   ```bash
   python TitanicLogasticRegression.py
   ```
5. The trained model will be saved as `marvellousTitanic.pkl` in the same folder

---

## Key Concepts Covered

- Supervised Machine Learning
- Binary Classification
- Data Cleaning and Preprocessing
- Handling Missing Values (median and mode imputation)
- Feature Encoding (One-Hot Encoding)
- Logistic Regression
- Model Coefficients and Interpretability
- Model Persistence using joblib

---

## Author

**Raviraj Aade**

Built as part of a Machine Learning Case Study series to understand binary classification, real-world data preprocessing, and model saving techniques.
