# Machine Learning Experiments

This repository contains the implementations of the Machine Learning laboratory experiments, completed using Python and Google Colab as part of the Machine Learning course at Nanjing University of Information Science and Technology (NUIST).

| Experiment | Topic |
|---|---|
| [Laboratory 1](#laboratory-1-linear-regression-model-complexity-and-ridge-regularisation) | Linear Regression, Model Complexity and Ridge Regularisation |
| [Laboratory 2](#laboratory-2-binary-classification-logistic-regression-and-support-vector-machines) | Binary Classification, Logistic Regression and Support Vector Machines |

## Technologies Used

- Python
- NumPy
- Matplotlib
- scikit-learn (Laboratory 2, only for the SVM)
- Google Colab

---

# Laboratory 1: Linear Regression, Model Complexity and Ridge Regularisation

The experiment focuses on understanding linear regression, the effect of increasing model complexity, overfitting, and the role of Ridge Regularisation in improving model generalisation.

## Objectives

- Implement Linear Regression using matrix operations and the Normal Equation.
- Generate a synthetic dataset and divide it into training and testing sets.
- Calculate Mean Squared Error (MSE) to evaluate model performance.
- Understand how increasing the number of input features affects model complexity.
- Identify overfitting through training and testing errors.
- Implement Ridge Regression and investigate the effect of different regularisation parameters.

## Experimental Setup

A synthetic dataset was generated with the following configuration:

| Parameter | Value |
|---|---|
| Number of observations | 80 |
| Number of input features | 40 |
| Training samples | 56 |
| Testing samples | 24 |
| Random seed | 7 |
| Noise standard deviation | 2 |

The target values were generated using the following equation:

$$y_i = 1 + 3x_{i1} - 2x_{i2} + 1.5x_{i3} + 0.5x_{i4} - x_{i5} + \epsilon_i$$

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

## Key Results

| Model | Training MSE | Testing MSE | Parameter norm |
|---|---|---|---|
| 5 features | 3.5543 | 4.4627 | – |
| 40 features | 0.6666 | 13.5589 | 5.0131 |
| Ridge, λ = 0.01 | 0.6765 | 12.3476 | 4.8023 |
| Ridge, λ = 0.1 | 1.0965 | 8.3322 | 3.7392 |
| Ridge, λ = 10 | 11.9263 | 12.7410 | 0.3464 |

## Key Learning Outcomes

- How Linear Regression can be implemented using matrix operations.
- Why training and testing datasets are important for evaluating machine learning models.
- How increasing the number of features can lead to overfitting.
- Why a lower training error does not necessarily indicate better performance on unseen data.
- How Ridge Regularisation controls parameter magnitudes.
- How the regularisation parameter λ affects training error, testing error, and model generalisation.

## Conclusion

This laboratory demonstrates the relationship between model complexity, overfitting, and regularisation. The experiments show that increasing the number of features can reduce training error while increasing testing error. Ridge Regularisation helps control parameter magnitudes and can improve generalisation when an appropriate regularisation strength is selected.

The laboratory provided practical experience in implementing Linear Regression and Ridge Regression from scratch using NumPy and interpreting their results through numerical evaluation and visualisation.

---

# Laboratory 2: Binary Classification, Logistic Regression and Support Vector Machines

The experiment builds a binary classifier from first principles using the sigmoid function, binary cross-entropy and gradient descent, and then compares this probabilistic view of classification with the geometric maximum-margin view of a linear Support Vector Machine (SVM).

## Objectives

- Implement the sigmoid function and binary cross-entropy loss.
- Train logistic regression using gradient descent.
- Interpret predicted probabilities and a linear decision boundary.
- Use a linear SVM and identify its support vectors.
- Compare probabilistic and margin-based linear classification.

## Experimental Setup

A two-dimensional binary classification dataset was generated with the following configuration:

| Parameter | Value |
|---|---|
| Number of observations | 120 (60 per class) |
| Number of input features | 2 |
| Training samples | 84 |
| Testing samples | 36 |
| Random seed | 11 |
| Covariance of each class | Identity matrix |

The two classes were generated from Gaussian distributions:

$$x \mid y=0 \sim \mathcal{N}\left(\begin{bmatrix}-1.2\\-1.0\end{bmatrix}, I\right), \qquad x \mid y=1 \sim \mathcal{N}\left(\begin{bmatrix}1.2\\1.0\end{bmatrix}, I\right)$$

Logistic regression uses the labels $y \in \{0, 1\}$. For the SVM, the labels are converted to $y_{SVM} = 2y - 1 \in \{-1, +1\}$.

## Experiments

### Task 1: Generate and Inspect the Classification Data

- Generated 60 observations for each class and combined them into one dataset.
- Randomly permuted the observations using the same random generator.
- Used the first 84 observations for training and the remaining 36 for testing.
- Verified the dimensions of the training and testing arrays.
- Created a scatter plot of the complete dataset using a different colour and marker for each class.

### Task 2: Sigmoid Probabilities and Cross-Entropy

- Implemented the sigmoid function and checked σ(0) = 0.5, σ(2) ≈ 0.881 and σ(−2) ≈ 0.119.
- Implemented the binary cross-entropy loss with probabilities clipped to [10⁻¹², 1 − 10⁻¹²].
- Calculated the loss for p = 0.8 with y = 1 and with y = 0.
- Computed the training cross-entropy at w = 0 and b = 0, before any update.

### Task 3: Train Logistic Regression with Gradient Descent

- Implemented logistic regression directly in NumPy, without a pre-built classifier.
- Trained the model with gradient descent (learning rate 0.1, 1000 iterations).
- Recorded the training cross-entropy at every iteration and plotted it.
- Converted probabilities to class labels with the threshold 0.5 and calculated training and testing accuracy.
- Plotted the complete dataset with the learned decision boundary.
- Identified the test observations with the probability closest to 0.5, the largest probability and the smallest probability.

### Task 4: A Linear Support Vector Machine

- Converted the labels to {−1, +1}.
- Fitted a soft-margin linear SVM using `sklearn.svm.SVC` with `kernel='linear'` and `C=1`.
- Reported w, b, the number of support vectors, and the training and testing accuracy.
- Calculated the margin width 2 / ‖w‖₂.
- Plotted the training data, decision boundary, margin boundaries and support vectors.
- Compared the SVM boundary with the logistic regression boundary.

## Key Results

| Item | Logistic regression | Linear SVM (C = 1) |
|---|---|---|
| Learned w | [1.9078, 1.7761] | [0.9298, 1.0909] |
| Learned b | −0.1284 | −0.1302 |
| Training accuracy | 0.9524 | 0.9405 |
| Testing accuracy | 0.9444 | 0.9444 |
| Final training cross-entropy | 0.1857 | – |
| Support vectors | – | 19 |
| Margin width | – | 1.3953 |

The initial training cross-entropy at w = 0 and b = 0 was 0.6931, which equals ln(2).

## Key Learning Outcomes

- How the sigmoid function turns a linear score into a probability.
- Why the initial cross-entropy at zero parameters equals ln(2).
- How gradient descent trains logistic regression and how the loss curve shows convergence.
- How predicted probabilities relate to the distance from the decision boundary.
- How a linear SVM is determined by its support vectors and margin.
- Why logistic regression and a linear SVM can produce similar boundaries even though their learning principles differ.

## Conclusion

This laboratory shows that logistic regression and a linear SVM can solve the same binary classification problem with very similar decision boundaries. Logistic regression provides class probabilities and is trained using all observations through the cross-entropy loss, whereas the SVM is a geometric maximum-margin classifier determined mainly by its support vectors. On this dataset both methods reached the same testing accuracy of 0.9444, because the two classes are nearly linearly separable with only a small overlap.

---

## How to Run

The experiments were implemented in Google Colab.

1. Clone or download this repository.
2. Open the Jupyter Notebook of the experiment in Google Colab.
3. Make sure the required libraries are installed.
4. Execute the notebook cells in sequential order, starting from the environment setup.
5. Observe the numerical results and generated graphs for each task.

Required libraries:

```bash
pip install numpy matplotlib scikit-learn
```

---

**Course:** Machine Learning

**Laboratories:** Laboratory 1 and Laboratory 2

**Institution:** Nanjing University of Information Science and Technology (NUIST)

**Academic Year:** 2026–2027
