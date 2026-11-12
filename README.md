# Machine Learning Laboratory 1
## Linear Regression, Model Complexity and Ridge Regularisation

This repository contains the implementation of Machine Learning Laboratory 1, completed using Python and Google Colab as part of the Machine Learning course at Nanjing University of Information Science and Technology (NUIST).

The experiment focuses on understanding linear regression, the effect of increasing model complexity, overfitting, and the role of Ridge Regularisation in improving model generalisation.

## Objectives

The main objectives of this laboratory are:

- Implement Linear Regression using matrix operations and the Normal Equation.
- Generate a synthetic dataset and divide it into training and testing sets.
- Calculate Mean Squared Error (MSE) to evaluate model performance.
- Understand how increasing the number of input features affects model complexity.
- Identify overfitting through training and testing errors.
- Implement Ridge Regression and investigate the effect of different regularisation parameters.

## Technologies Used

- Python
- NumPy
- Matplotlib
- Google Colab

## Experimental Setup

A synthetic dataset was generated with the following configuration:

| Parameter | Value |
|---|---:|
| Number of observations | 80 |
| Number of input features | 40 |
| Training samples | 56 |
| Testing samples | 24 |
| Random seed | 7 |
| Noise standard deviation | 2 |

The target values were generated using the following equation:

\[
y_i = 1+3x_{i1}-2x_{i2}+1.5x_{i3}+0.5x_{i4}-x_{i5}+\epsilon_i
\]

Only the first five features contribute to the target values, while the remaining 35 features have zero coefficients.

## Experiments

### Task 1: Generate and Inspect the Data

- Generated an input matrix containing 80 observations and 40 features.
- Generated target values using the specified linear equation and Gaussian noise.
- Randomly shuffled the observations.
- Divided the dataset into 56 training samples and 24 testing samples.
- Implemented a Mean Squared Error (MSE) function using NumPy.
- Verified the dimensions of the generated datasets.

### Task 2: A Compact Linear Regression Model

- Selected only the first five input features.
- Added a bias column to the feature matrix.
- Implemented Linear Regression using the Normal Equation.
- Estimated the model parameters using `numpy.linalg.solve()`.
- Calculated training and testing MSE.
- Created a scatter plot comparing actual and predicted testing values.

### Task 3: Increase the Model Complexity

- Repeated the regression experiment using all 40 input features.
- Added a bias column, resulting in 41 model parameters.
- Estimated the parameters using the Normal Equation.
- Calculated training MSE, testing MSE, and the Euclidean norm of the parameter vector.
- Created a second scatter plot to examine prediction performance.
- Compared the five-feature and 40-feature models to investigate overfitting.

### Task 4: Ridge Regularisation

- Applied Ridge Regression using all 40 input features.
- Implemented the regularised Normal Equation.
- Evaluated three regularisation parameters:

  - λ = 0.01
  - λ = 0.1
  - λ = 10

- Calculated training MSE, testing MSE, and parameter norms for each value.
- Compared the results to investigate the effect of regularisation.
- Created graphs to visualise the relationship between lambda, prediction errors, and parameter magnitudes.

## Key Learning Outcomes

Through these experiments, I explored the following concepts:

1. How Linear Regression can be implemented using matrix operations.
2. Why training and testing datasets are important for evaluating machine learning models.
3. How increasing the number of features can lead to overfitting.
4. Why a lower training error does not necessarily indicate better performance on unseen data.
5. How Ridge Regularisation controls parameter magnitudes.
6. How the regularisation parameter λ affects training error, testing error, and model generalisation.

## How to Run

The experiment was implemented in Google Colab.

1. Clone or download this repository.
2. Open the Jupyter Notebook in Google Colab.
3. Make sure NumPy and Matplotlib are installed.
4. Execute the notebook cells in sequential order, starting from the environment setup.
5. Observe the numerical results and generated graphs for each task.

Required libraries:

```bash
pip install numpy matplotlib
```

## Conclusion

This laboratory demonstrates the relationship between model complexity, overfitting, and regularisation. The experiments show that increasing the number of features can reduce training error while increasing testing error. Ridge Regularisation helps control parameter magnitudes and can improve generalisation when an appropriate regularisation strength is selected.

The laboratory provided practical experience in implementing Linear Regression and Ridge Regression from scratch using NumPy and interpreting their results through numerical evaluation and visualisation.

---

**Course:** Machine Learning

**Laboratory:** Laboratory 1

**Institution:** Nanjing University of Information Science and Technology (NUIST)

**Academic Year:** 2026–2027
