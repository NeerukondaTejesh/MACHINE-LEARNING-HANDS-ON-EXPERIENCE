# Multiple Linear Regression

## Overview

Multiple Linear Regression is a supervised machine learning algorithm used to model the relationship between one dependent variable and multiple independent variables.

In this project, Multiple Linear Regression is used to predict the **Profit of a startup** based on several factors such as research and development spending, administration spending, marketing spending, and the startup's state.

## Problem Statement

The objective of this project is to build a Multiple Linear Regression model that predicts a startup's profit using multiple input features.

The model uses:

* **Independent variables (X):** R&D Spend, Administration, Marketing Spend, and State
* **Dependent variable (y):** Profit

## Dataset

The dataset used in this project is:

`50_Startups.csv`

The dataset contains information about different startups and their corresponding expenditures and profit.

## Implementation

The complete implementation is available in:

`multiple_linear_regression.ipynb`

The following steps were performed in the notebook.

### 1. Importing the Libraries

The following Python libraries were used:

* NumPy
* Pandas
* Matplotlib
* Scikit-learn

### 2. Importing the Dataset

The dataset was loaded using Pandas:

```python
dataset = pd.read_csv('50_Startups.csv')
```

The independent variables and dependent variable were then separated into `X` and `y`.

```python
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

### 3. Encoding Categorical Data

The `State` column contains categorical values, so it was converted into numerical form using **One-Hot Encoding**.

`ColumnTransformer` and `OneHotEncoder` from Scikit-learn were used for this preprocessing step.

```python
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [3])],
    remainder='passthrough'
)

X = np.array(ct.fit_transform(X))
```

This converts the categorical state information into numerical features that can be used by the regression model.

### 4. Splitting the Dataset

The dataset was divided into training and test sets using `train_test_split`.

An **80% training and 20% test split** was used with:

```python
test_size = 0.2
random_state = 0
```

The training set was used to learn the relationship between the input features and profit, while the test set was used to evaluate predictions on unseen data.

### 5. Training the Multiple Linear Regression Model

A `LinearRegression` model from Scikit-learn was created and trained using the training data.

```python
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```

### 6. Predicting the Test Set Results

The trained model was used to predict the profits for the test set:

```python
y_pred = regressor.predict(X_test)
```

The predicted values and actual test values were then printed side by side for comparison.

### 7. Making a Single Prediction

The trained model was also used to make a prediction for a single startup using specified input values.

```python
regressor.predict([[1, 0, 0, 160000, 130000, 300000]])
```

This demonstrates how the trained regression model can be used to predict the expected profit for a new startup based on its input features.

### 8. Obtaining the Regression Equation

The coefficients and intercept of the trained regression model were obtained using:

```python
print(regressor.coef_)
print(regressor.intercept_)
```

These values represent the learned parameters of the Multiple Linear Regression equation.

The general form of the model is:

**Profit = b₀ + b₁X₁ + b₂X₂ + ... + bₙXₙ**

where:

* `b₀` is the intercept
* `b₁, b₂, ..., bₙ` are the learned coefficients
* `X₁, X₂, ..., Xₙ` are the input features

## Workflow

The project follows this machine learning workflow:

```text
Import Libraries
       ↓
Import Dataset
       ↓
Separate Features and Target
       ↓
Encode Categorical Data
       ↓
Split Dataset
       ↓
Train Multiple Linear Regression Model
       ↓
Predict Test Results
       ↓
Make a Single Prediction
       ↓
Obtain Coefficients and Intercept
```

## Files

| File                               | Description                                           |
| ---------------------------------- | ----------------------------------------------------- |
| `multiple_linear_regression.ipynb` | Complete implementation of Multiple Linear Regression |
| `50_Startups.csv`                  | Dataset used for training and testing                 |
| `README.md`                        | Documentation for this project                        |

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Google Colab

## Learning Outcomes

Through this implementation, I practiced:

* Loading a dataset using Pandas
* Separating independent and dependent variables
* Handling categorical data
* Applying One-Hot Encoding
* Using `ColumnTransformer`
* Splitting data into training and testing sets
* Training a Multiple Linear Regression model
* Making predictions on test data
* Making predictions for new input data
* Understanding regression coefficients and intercept
* Understanding the basic workflow of supervised machine learning

## Conclusion

This project demonstrates how Multiple Linear Regression can be used to predict a continuous target variable using multiple input features.

The implementation covers the complete process from data loading and categorical data encoding to model training, prediction, and interpretation of the learned coefficients and intercept.
