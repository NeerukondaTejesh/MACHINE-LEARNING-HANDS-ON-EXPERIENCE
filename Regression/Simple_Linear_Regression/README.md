# Simple Linear Regression

## Overview

Simple Linear Regression is a supervised machine learning algorithm used to model the relationship between one independent variable and one dependent variable.

In this project, Simple Linear Regression is used to understand the relationship between **Years of Experience** and **Salary** and to predict salary based on an individual's years of experience.

## Problem Statement

The objective of this project is to build a Simple Linear Regression model that predicts an employee's salary from their years of experience.

The model learns the relationship between:

* **Independent variable (X):** Years of Experience
* **Dependent variable (y):** Salary

## Dataset

The dataset used in this project is:

`Salary_Data.csv`

The dataset contains information about employees' years of experience and their corresponding salaries.

## Implementation

The complete implementation is available in:

`simple_linear_regression.ipynb`

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
dataset = pd.read_csv('Salary_Data.csv')
```

The independent and dependent variables were then separated into `X` and `y`.

### 3. Splitting the Dataset

The dataset was divided into training and test sets using `train_test_split`.

A test size of **1/3** was used, with `random_state = 0` to make the split reproducible.

### 4. Training the Model

A `LinearRegression` model from Scikit-learn was created and trained using the training data.

```python
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```

### 5. Predicting the Test Results

The trained model was used to predict salary values for the test data.

```python
y_pred = regressor.predict(X_test)
```

### 6. Visualizing the Training Set

A scatter plot was used to display the actual training observations along with the regression line.

### 7. Visualizing the Test Set

The test observations were plotted together with the fitted regression line to visualize the model's predictions on unseen data.

## Libraries and Technologies

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Google Colab

## Files

| File                             | Description                                         |
| -------------------------------- | --------------------------------------------------- |
| `simple_linear_regression.ipynb` | Complete implementation of Simple Linear Regression |
| `Salary_Data.csv`                | Dataset used for the project                        |
| `README.md`                      | Documentation for this project                      |

## Learning Outcome

Through this implementation, I practiced:

* Loading and exploring a dataset
* Separating features and target variables
* Splitting data into training and testing sets
* Training a linear regression model
* Making predictions
* Visualizing regression results
* Understanding the basic workflow of supervised machine learning

## Conclusion

This project demonstrates the complete workflow of a Simple Linear Regression model, from importing the dataset and preparing the data to training the model, making predictions, and visualizing the results.

The project provides a practical understanding of how a machine learning model can be used to predict a continuous numerical value from a single input feature.
