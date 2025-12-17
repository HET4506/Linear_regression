# Linear & Multiple Linear Regression from Scratch

This repository contains implementations of **Linear Regression and Multiple Linear Regression from scratch using NumPy**, without relying on machine learning libraries such as scikit-learn.

The goal of this project is to understand the internal mechanics of regression models, gradient descent, and data preprocessing.

---

## Datasets Used

- **Advertising Dataset (Kaggle)**
  - Simple Linear Regression
  - Multiple Linear Regression

- **Ames Housing Dataset**
  - Simple Linear Regression 
  - Multiple Linear Regression

---

## Key Concepts Implemented

- Feature normalization (mean–standard deviation scaling)
- Reuse of training statistics during inference
- Batch Gradient Descent
- Cost function (Mean Squared Error)
- Manual weight and bias updates
- End-to-end ML pipeline: data → preprocessing → training → prediction

---

## Project Structure

```md
- **advertising_regression/**  
  Implements simple and multiple linear regression using batch gradient descent, applied to the Advertising dataset to study feature influence and model convergence.

- **ames_housing_regression/**  
  Implements multiple linear regression with feature normalization and batch gradient descent on the Ames Housing dataset, demonstrating scalability and inference consistency.
