# STAT4609: Big Data Analytics
This is a repositoty for the course STAT4609: Big Data Analytics at HKU. It contains course Assignments and project.
The assignments are implementations of machine learning algorithms from scratch in Python through object oriented programming.

## Assignment 1
Predicting housing prices in Boston by implementing the following models from scratch:
<a href="https://github.com/alexsuakim/STAT4609/tree/main/A1">Repository for Assignment 1</a>
1. Linear regression
2. Ridge regression
- Multicolinearity & overfitting: Ridge regression produces a smaller test set MSE than the OLS result (linear regression). This is because OLS is sensitive to errors in the observed data when multicolinearity is present. This could lead to a large variance in the predicted parameters (beta hat), which means it could easily overfit. Ridge regression, on the other hand, mitigates multicolinearity by adding a penalty to these coefficients, which reduces the variance.
- Hyperparameter tuning: as lambda (the regularisation parameter) increases, the test MSE first decreases then increases again, making a U-shape. 
  
## Assignment 2
Classifying a breast cancer dataset by implementing the following models from scratch:
<a href="https://github.com/alexsuakim/STAT4609/tree/main/A2">Repository for Assignment 2</a>
1. Logistic regression
2. Naive bayes classifier
- Maximum likelihood estimation: log likelihood increases as the number of iterations increases.
- Gradient Descent: a logistic regression classifier has to go through an iterative optimisation process called gradient descent/ascent. Due to gradient descent, the accuracy of logistic regression is higher than that of naive bayes, but the training time of logistic regression is also higher than that of naive bayes. The time complexity of LR is O(ndk), and that of NB is O(nk), where n = samples, d = dimensions, k = iterations.

## Assignment 3
Classifying Iris dataset by implementing the following models from scratch:
<a href="https://github.com/alexsuakim/STAT4609/tree/main/A3">Repository for Assignment 3</a>
1. Decision tree
2.  Random forest
- Gini index vs. cost misclassification
- Decision tree vs. Random forest:
  - Decision tree: great interpretability, non parametric (no assumptions of linearity in the data & can handle both numerical & categorical data); prone to overfitting, often needs pruning of the tree.
  - Random forest: builds on top of a decision tree model; reduces variance & helps avoid overfitting; has higher complexity, lower interpretability due to the nature of an ensemble model.

## Assignment 4
Classifying images of hand written digits by implementing the following models from scratch:
<a href="https://github.com/alexsuakim/STAT4609/tree/main/A4">Repository for Assignment 4</a>
- K-means clustering
- Gaussian mixture model
- using Gibbs sampling
- comparison & insight?

## Course Project
