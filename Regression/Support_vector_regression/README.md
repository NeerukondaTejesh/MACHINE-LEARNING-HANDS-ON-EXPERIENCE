# Support Vector Regression (SVR)

## Overview

Support Vector Regression (SVR) is a supervised machine learning algorithm used to predict continuous numerical values. It applies the principles of Support Vector Machines to regression problems by finding a function that models the relationship between input features and target values.

In this project, Support Vector Regression is used to model the relationship between **Position Level** and **Salary**. An SVR model with a Radial Basis Function (RBF) kernel is trained to predict salary based on position level.

## Problem Statement

The objective of this project is to predict the salary associated with a particular position level using Support Vector Regression.

The model uses:

- **Independent variable (X):** Position Level
- **Dependent variable (y):** Salary

## Dataset

The dataset used in this project is:

`Position_Salaries.csv`

The dataset contains position levels and their corresponding salaries.

The position level is used as the input feature, and salary is used as the target variable.

## Implementation

The complete implementation is available in:

`Support_vector_regression.ipynb`

The following steps were performed in the notebook.

### 1. Importing the Libraries

The following Python libraries were used:

- NumPy
- Pandas
- Matplotlib
- Scikit-learn

### 2. Importing the Dataset

The dataset was loaded using Pandas:

```python
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
```

The input feature and target variable were extracted from the dataset.

The target variable was reshaped into a two-dimensional array before feature scaling:

```python
y = y.reshape(len(y), 1)
```

### 3. Feature Scaling

Feature scaling was applied to both the input feature and target variable using `StandardScaler`.

```python
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
```

Scaling is particularly important for SVR because the algorithm's results can be affected by differences in feature scales.

Separate scalers were used for the input feature and target variable.

### 4. Training the SVR Model

An SVR model with the **RBF (Radial Basis Function) kernel** was created and trained on the scaled dataset.

```python
from sklearn.svm import SVR

regressor = SVR(kernel='rbf')
regressor.fit(X, y)
```

The RBF kernel allows the model to represent non-linear relationships between position level and salary.

### 5. Predicting a New Result

The model was used to predict the salary corresponding to position level `6.5`.

The input value was first transformed using the input scaler. The predicted salary was then converted back to the original salary scale using the target scaler.

```python
sc_y.inverse_transform(
    regressor.predict(
        sc_X.transform([[6.5]])
    ).reshape(-1, 1)
)
```

This demonstrates how the trained SVR model can make predictions for new input values.

### 6. Visualizing the SVR Results

A scatter plot was used to display the actual position levels and salaries, while a line plot showed the predictions made by the SVR model.

The scaled data and predictions were transformed back to their original scales for visualization.

```python
plt.scatter(
    sc_X.inverse_transform(X),
    sc_y.inverse_transform(y),
    color='red'
)

plt.plot(
    sc_X.inverse_transform(X),
    sc_y.inverse_transform(
        regressor.predict(X).reshape(-1, 1)
    ),
    color='blue'
)
```

The red points represent the actual observations, and the blue curve represents the model's predictions.

### 7. Creating a Higher-Resolution SVR Curve

To obtain a smoother visualization, a grid of closely spaced position-level values was generated.

```python
X_grid = np.arange(
    min(sc_X.inverse_transform(X)),
    max(sc_X.inverse_transform(X)),
    0.1
)

X_grid = X_grid.reshape((len(X_grid), 1))
```

The grid values were scaled before being passed to the trained SVR model. The predicted values were then inverse-transformed to the original salary scale.

This produces a smoother curve that makes the model's non-linear predictions easier to visualize.

## Workflow

The project follows this machine learning workflow:

```text
Import Libraries
       ↓
Import Dataset
       ↓
Separate Features and Target
       ↓
Reshape Target Variable
       ↓
Apply Feature Scaling
       ↓
Train SVR Model with RBF Kernel
       ↓
Predict Salary for a New Position Level
       ↓
Convert Prediction to Original Scale
       ↓
Visualize SVR Results
       ↓
Generate a Smooth Prediction Curve
```

## Files

| File | Description |
|---|---|
| `Support_vector_regression.ipynb` | Complete implementation of Support Vector Regression |
| `Position_Salaries.csv` | Dataset used for training the model |
| `README.md` | Documentation for this project |

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Google Colab
- Jupyter Notebook

## Learning Outcomes

Through this implementation, I practiced:

- Loading datasets using Pandas
- Extracting independent and dependent variables
- Reshaping a target variable
- Applying feature scaling with `StandardScaler`
- Understanding the importance of scaling in SVR
- Training an SVR model using the RBF kernel
- Predicting results for new input values
- Converting scaled predictions back to their original scale
- Visualizing actual observations and model predictions
- Creating a higher-resolution prediction curve
- Understanding non-linear regression using Support Vector Machines

## Conclusion

This project demonstrates how Support Vector Regression can be applied to a salary prediction problem involving a non-linear relationship between position level and salary.

The implementation covers dataset preparation, feature scaling, model training with an RBF kernel, prediction, inverse transformation, and visualization.

It provides practical experience with SVR and demonstrates how scaling and kernel selection contribute to building a regression model.
