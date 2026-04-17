## Project Summary & Checklist

- **Project:** Recommendation System using Matrix Factorization.
- **Dataset:** MovieLens-1M.
- **Core Mathematical Algorithms:**
  - Optimization via Alterating Least Squares.
  - Solving OLS sub-problems using system of linear equations.

## 1. Algorithm

- **Objective:** Implement the mathematical core from scratch (focusing on performance and correct matrix operations).
- **Key Tasks:**
  - Implement optimized Python code for ALS mathematical formulas using ``numpy`` and ``scipy.sparse``.
  - Implement the Alternating Least Squares loop.
  - Solve the normal equations for the OLS step ($Ax = B$), using QR decomposition.
  - Collaborate to debug matrix dimension mismatches and ensure the loss function converges correctly.
  - Test the algorithm for the developing phase with a small data which is sliced from the original dataset.

- **Member:** Minh Đức, Trần Nhi.

## 2. Data Mining & Metrics Evaluator

- **Objective:** Prepare the dataset and evaluate the algorithm using RMSE.
- **Key Tasks:**
  - Preprocessing the dataset for the final phase.
  - Convert User IDs and Movie IDs into a usable matrix idnex format.
  - Split the dataset into Training Data (80%) and Testing Data (20%).
  - Write the evaluation function: Calculate the RMSE on the Test set to validate the model's accuracy.

- **Member:** Tuệ Nghi.

## 3. Report

- **Objective:** Ensure the right formatting and clearly articulate the math, and deliver a 10-minute presentation covering theory and practical results.
- **Key Tasks:**
  - Setup a standard LaTeX template including a proper Cover Page and Task Assignment Table.
  - Write the theoretical methodology section: Clearly typeset the math formulas, Loss Function, derivatives.
  - Design clean, professional slides, using Beam on LaTeX.
  - Ensure the 10-minute script: From Problem Statement $\rightarrow$ Mathematical Approach $\rightarrow$ Live Code Result/ RMSE Score (if possible) $\rightarrow$ Conclusion.

- **Member:** Thanh Huy, Ngọc Thạch.