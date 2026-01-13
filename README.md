# Machine Learning Algorithms

This repository contains implementations of various Machine Learning algorithms from scratch, focusing on the core concepts and mathematical foundations behind each model.

## Currently Implemented Algorithms

1. **Linear Regression** (Salary Prediction)  
2. **Logistic Regression** (Heart Disease Prediction)  
3. **Decision Tree** (Iris Classification)  
4. **Random Forest** (Breast Cancer Classification)
5. **K-Nearest Neighbours** (Breast Cancer Classification)

*More machine learning algorithms will be added soon.*

---

## 1. Linear Regression (Salary Prediction)

Located in `Linear_Regression/`.

### Description
A simple linear regression model that predicts salary based on years of experience using Gradient Descent optimization.

### Dataset
- `Linear_Regression/Salary_dataset.csv`
- **Feature:** `YearsExperience`
- **Target:** `Salary`

### Implementation Details
The model predicts salary using the equation:

$$
y = mx + c
$$

The parameters \( m \) (slope) and \( c \) (intercept) are optimized by minimizing the Mean Squared Error (MSE) using Gradient Descent:

$$
m := m - \alpha \frac{\partial J}{\partial m},
\quad
c := c - \alpha \frac{\partial J}{\partial c}
$$

---

## 2. Logistic Regression (Heart Disease Prediction)

Located in `Logistic_Regression/`.

### Description
A logistic regression model that predicts the probability of heart disease using the sigmoid activation function.

### Dataset
- `Logistic_Regression/Heart_Disease_Prediction.csv`
- **Features:** Multiple health-related metrics (normalized)
- **Target:** Presence of Heart Disease (Binary: 0 or 1)

### Implementation Details
The predicted probability is computed using the sigmoid function:

$$
g(z) = \frac{1}{1 + e^{-z}}, \quad \text{where } z = wX + b
$$

The loss function used is Binary Cross-Entropy (Log Loss):

$$
J(w, b) = -\frac{1}{n} \sum_{i=1}^{n}
\left[
y_i \log(g(z_i)) + (1 - y_i)\log(1 - g(z_i))
\right]
$$

Model parameters are updated using Gradient Descent.

---

## 3. Decision Tree (Iris Classification)

Located in `Decision_Tree/`.

### Description
A decision tree classifier built from scratch to classify Iris flowers into their respective species.

### Dataset
- `Decision_Tree/Iris.csv`
- **Features:** `SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`
- **Target:** `Species` (Iris-setosa, Iris-versicolor, Iris-virginica)

### Implementation Details
The decision tree is built recursively by selecting feature splits that maximize **Information Gain**.

**Entropy:**

$$
E(S) = -\sum_{i} p_i \log_2(p_i)
$$

**Information Gain:**

$$
IG(S, A) = E(S) - \sum_{v} \frac{|S_v|}{|S|} E(S_v)
$$

---

## 4. Random Forest (Breast Cancer Classification)

Located in `Random_Forest/`.

### Description
An ensemble learning method using multiple Decision Trees to improve classification accuracy and control over-fitting.

### Dataset
- Breast Cancer Wisconsin (Diagnostic) Dataset (loaded from `sklearn.datasets`)
- **Features:** Computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.
- **Target:** Cancer Diagnosis (Malignant or Benign)

### Implementation Details
The model uses bootstrap aggregating (bagging) to create multiple decision trees on random subsets of data and features. The final prediction is determined by majority voting.

**Bootstrap Aggregating:**
Randomly sampling with replacement to train each tree.

**Prediction:**

$$
\hat{y} = \text{mode}(\{h_1(x), h_2(x), \dots, h_n(x)\})
$$

where \( h_i \) represents the output of the \( i \)-th base learner.

---

## 5. K-Nearest Neighbours (Breast Cancer Classification)

Located in `K_Nearest_Neighbours/`.

### Description
A K-Nearest Neighbours (KNN) classifier that classifies breast cancer cases as malignant or benign based on the similarity to labeled examples.

### Dataset
- `K_Nearest_Neighbours/data.csv` (Breast Cancer Wisconsin Diagnostic Dataset)
- **Features:** Computed from a digitized image of a fine needle aspirate (FNA) of a breast mass (e.g., radius, texture, perimeter).
- **Target:** `diagnosis` (M = Malignant, B = Benign)

### Implementation Details
The algorithm classifies new data points based on the majority class of their \( k \) nearest neighbors.

**Euclidean Distance:**
Used to measure similarity between data points:

$$
d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
$$

**Prediction:**
The class is determined by a majority vote of the neighbors:

$$
\hat{y} = \text{mode}(\{y_1, y_2, \dots, y_k\})
$$

The project includes both a from-scratch implementation in `KNN.ipynb` and an implementation using `scikit-learn` in `knn.py`.


## Prerequisites

- Python 3.x
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn` (used for `train_test_split` and dataset loading)

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```
