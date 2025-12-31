# Linear Regression Salary Prediction

This project demonstrates a simple linear regression model to predict salary based on years of experience. It uses gradient descent to optimize the model parameters.

## Project Structure

- `Linear_Regression/`: Directory containing the project files.
  - `LinearRegression.ipynb`: Jupyter notebook implementing the linear regression model.
  - `Salary_dataset.csv`: Dataset containing years of experience and salary data.
  - `output.png`: Sample output plot of the regression line.

## Dataset

The dataset `Salary_dataset.csv` contains the following columns:
- `YearsExperience`: Number of years of experience.
- `Salary`: Salary corresponding to the experience.

## Prerequisites

To run the notebook, you need the following Python libraries installed:
- `matplotlib`
- `numpy`
- `pandas`

You can install them using pip:
```bash
pip install matplotlib numpy pandas
```

## Usage

1. Navigate to the `Linear_Regression` directory.
2. Open the `LinearRegression.ipynb` notebook using Jupyter Notebook or JupyterLab.
3. Run all the cells to train the model and visualize the results.

## Implementation Details

The linear regression model is implemented from scratch using Gradient Descent.

### Model
The model predicts salary ($y$) based on years of experience ($x$) using the equation:
$$ y = mx + c $$
where:
- $m$ is the slope (weight).
- $c$ is the intercept (bias).

### Gradient Descent
The model parameters ($m$ and $c$) are updated iteratively to minimize the Mean Squared Error (MSE) loss.
The gradients are calculated as:
$$ \frac{\partial J}{\partial m} = \frac{-2}{n} \sum (y - (mx + c)) \cdot x $$
$$ \frac{\partial J}{\partial c} = \frac{-2}{n} \sum (y - (mx + c)) $$

The parameters are updated using the learning rate $\alpha$:
$$ m = m - \alpha \frac{\partial J}{\partial m} $$
$$ c = c - \alpha \frac{\partial J}{\partial c} $$

## Results

After training for 10,000 iterations, the model learns the relationship between experience and salary. The notebook produces a plot showing the original data points and the fitted regression line.
