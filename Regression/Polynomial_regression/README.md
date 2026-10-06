# Polynomial Regression

## Overview

Polynomial Regression is a supervised machine learning technique used to model the relationship between an independent variable and a dependent variable when the relationship is non-linear.

In this project, Polynomial Regression is used to model the relationship between **Position Level** and **Salary**. The project also compares the Polynomial Regression model with a Simple Linear Regression model.

## Problem Statement

The objective of this project is to predict the salary associated with a particular position level and determine whether a polynomial model can represent the relationship between position level and salary more effectively than a simple linear model.

The model uses:

* **Independent variable (X):** Position Level
* **Dependent variable (y):** Salary

## Dataset

The dataset used in this project is:

`Position_Salaries.csv`

The dataset contains information about different position levels and their corresponding salaries.

For the model, the **Position Level** column is used as the input feature and **Salary** is used as the target variable.

## Implementation

The complete implementation is available in:

`Polynomial_regression.ipynb`

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
dataset = pd.read_csv('Position_Salaries.csv')
```

The independent and dependent variables were then extracted:

```python
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
```

The position level is used as the independent variable and salary is used as the dependent variable.

### 3. Training the Linear Regression Model

A Simple Linear Regression model was trained using the complete dataset.

```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)
```

This model provides a baseline for comparison with the Polynomial Regression model.

### 4. Training the Polynomial Regression Model

Polynomial features were generated using `PolynomialFeatures`.

A polynomial degree of **4** was selected:

```python
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
```

A Linear Regression model was then trained on the transformed polynomial features:

```python
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
```

This allows the model to capture a non-linear relationship between position level and salary.

### 5. Visualizing the Linear Regression Results

The actual data points were plotted together with the predictions from the Linear Regression model.

```python
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
```

This visualization shows how a straight-line model represents the relationship between position level and salary.

### 6. Visualizing the Polynomial Regression Results

The Polynomial Regression predictions were plotted against the actual data points:

```python
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg_2.predict(X_poly), color='blue')
```

The polynomial curve provides a more flexible representation of the non-linear relationship in the dataset.

### 7. Creating a Higher-Resolution Polynomial Curve

To obtain a smoother and more detailed polynomial curve, a grid of closely spaced values was created:

```python
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
```

The transformed grid values were then passed to the trained Polynomial Regression model:

```python
plt.plot(
    X_grid,
    lin_reg_2.predict(poly_reg.fit_transform(X_grid)),
    color='blue'
)
```

This produces a smoother visualization of the polynomial regression curve.

### 8. Predicting a New Result with Linear Regression

The Linear Regression model was used to predict the salary corresponding to a position level of `6.5`:

```python
lin_reg.predict([[6.5]])
```

### 9. Predicting a New Result with Polynomial Regression

The Polynomial Regression model was also used to predict the salary for a position level of `6.5`:

```python
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
```

The two predictions can be compared to observe the difference between the linear and polynomial models.

## Workflow

The project follows this machine learning workflow:

```text
Import Libraries
       ↓
Import Dataset
       ↓
Separate Features and Target
       ↓
Train Linear Regression Model
       ↓
Generate Polynomial Features
       ↓
Train Polynomial Regression Model
       ↓
Visualize Linear Regression
       ↓
Visualize Polynomial Regression
       ↓
Create Smooth Polynomial Curve
       ↓
Predict New Results
       ↓
Compare Linear and Polynomial Predictions
```

## Files

| File                          | Description                                      |
| ----------------------------- | ------------------------------------------------ |
| `Polynomial_regression.ipynb` | Complete implementation of Polynomial Regression |
| `Position_Salaries.csv`       | Dataset used for training the models             |
| `README.md`                   | Documentation for this project                   |

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
* Selecting features and target variables
* Training a Linear Regression model
* Understanding the limitations of a linear model for non-linear relationships
* Generating polynomial features using `PolynomialFeatures`
* Training a Polynomial Regression model
* Choosing and applying a polynomial degree of 4
* Visualizing linear and polynomial regression results
* Creating a smoother regression curve using a prediction grid
* Making predictions for new input values
* Comparing Linear Regression and Polynomial Regression predictions
* Understanding the workflow of supervised regression problems

## Conclusion

This project demonstrates how Polynomial Regression can be used when the relationship between the input feature and target variable is non-linear.

A Simple Linear Regression model was first trained as a baseline, followed by a Polynomial Regression model using degree-4 polynomial features. The results were visualized using Matplotlib, and predictions were made for a new position level.

The project provides practical experience in feature transformation, regression modeling, visualization, and comparing different regression approaches for a prediction problem.
