# Machine Learning Algorithms

This repository contains implementations of various Machine Learning algorithms from scratch, demonstrating the core concepts and mathematics behind them.

## Currently Implemented Algorithms

1. **Linear Regression** (Salary Prediction)
2. **Logistic Regression** (Heart Disease Prediction)

*More ML algorithms are coming soon!*

## 1. Linear Regression (Salary Prediction)

Located in `Linear_Regression/`.

### Description
A simple linear regression model to predict salary based on years of experience using Gradient Descent.

### Dataset
- `Linear_Regression/Salary_dataset.csv`
- **Features**: `YearsExperience`
- **Target**: `Salary`

### Implementation Details
The model predicts $y$ (Salary) using the equation:
$$ y = mx + c $$

Parameters $m$ (slope) and $c$ (intercept) are optimized by minimizing the Mean Squared Error (MSE) through Gradient Descent:
$$ m = m - \alpha \frac{\partial J}{\partial m} $$
$$ c = c - \alpha \frac{\partial J}{\partial c} $$

## 2. Logistic Regression (Heart Disease Prediction)

Located in `Logistic_Regression/`.

### Description
A logistic regression model to predict the presence of heart disease. It uses the sigmoid function to map predictions to probabilities.

### Dataset
- `Logistic_Regression/Heart_Disease_Prediction.csv`
- **Features**: Various health metrics (normalized in the notebook).
- **Target**: Presence of Heart Disease (Binary: 0 or 1).

### Implementation Details
The model predicts the probability using the Sigmoid function:
$$ g(z) = \frac{1}{1 + e^{-z}} $$
where $z = wX + b$.

The cost function used is Binary Cross-Entropy (Log Loss):
$$ J(w, b) = -\frac{1}{n} \sum [y \log(g(z)) + (1-y) \log(1-g(z))] $$

Weights are updated using Gradient Descent.

## Prerequisites

- Python 3.x
- `numpy`
- `pandas`
- `matplotlib`
- `sklearn` (used for `train_test_split`)

Install dependencies:
```bash
pip install numpy pandas matplotlib scikit-learn
```

## Usage

1. Navigate to the algorithm's directory (e.g., `Linear_Regression` or `Logistic_Regression`).
2. Open the `.ipynb` notebook.
3. Run the cells to train the models and visualize results.
