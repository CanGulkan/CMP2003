# Recommender System: Predicting Ratings Using Matrix Factorization

This project implements a **recommender system** using **matrix factorization** to predict user ratings for items based on historical interaction data. The system is designed for accuracy and efficiency, evaluated using the **Root Mean Squared Error (RMSE)** metric. The project is part of a HackerRank challenge where the task is to predict hidden ratings and submit the results.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Objectives](#objectives)
- [Technologies](#technologies)
- [Getting Started](#getting-started)
- [Input Format](#input-format)
- [Output Format](#output-format)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Sample Input and Output](#sample-input-and-output)
- [Future Enhancements](#future-enhancements)
- [License](#license)

---

## Overview
Recommender systems help users discover items of interest by analyzing their preferences and interactions. In this project:
- **Matrix Factorization** is used to decompose the user-item interaction matrix into latent factors.
- Predictions are made for hidden ratings using **Stochastic Gradient Descent (SGD)**.
- Accuracy is measured using the RMSE metric:
  \[
  \text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (r_i - \hat{r}_i)^2}
  \]
  where \(r_i\) is the actual rating and \(\hat{r}_i\) is the predicted rating.

---

## Features
- **Matrix Factorization**: Efficiently predicts ratings by learning latent factors for users and items.
- **Bias Adjustment**: Includes user and item biases for better accuracy.
- **Stochastic Gradient Descent**: Optimizes the prediction model through iterative learning.
- **Performance Evaluation**: RMSE is computed to assess the model's predictive accuracy.

---

## Objectives
1. **Best Predictions**: Predict ratings as accurately as possible.
2. **Efficiency**: Minimize the time required for predictions while maintaining accuracy.

---

## Technologies
- **Programming Language**: C++
- **Compiler**: GCC or any standard C++ compiler
- **Build Tool**: Make or CMake
- **Libraries**: Standard Template Library (STL)

---

## Getting Started
### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/recommender-system.git
   cd recommender-system
   ```

2. Compile the program:
   ```bash
   g++ -std=c++11 -o recommender_system main.cpp
   ```

3. Run the executable:
   ```bash
   ./recommender_system
   ```

---

## Input Format
The program reads input in the following format:
- **Training Data**:
  ```
  train dataset
  UserID ItemID Rating
  0 0 4.0
  0 1 3.0
  1 0 5.0
  ```
- **Test Data**:
  ```
  test dataset
  UserID ItemID
  0 2
  1 1
  ```

---

## Output Format
The program outputs predicted ratings for test data in the following format:
```
3.75
4.25
2.50
```
Each line corresponds to a predicted rating for the given user-item pair.

---

## Usage
1. Prepare your dataset file (e.g., `input.txt`) with training and test data.
2. Run the program:
   ```bash
   ./recommender_system < input.txt > predictions.txt
   ```
3. Submit the `predictions.txt` file to the HackerRank platform.

---

## Evaluation
The RMSE metric is used to evaluate prediction accuracy. The formula for RMSE is:
\[
\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (r_i - \hat{r}_i)^2}
\]

---

## Sample Input and Output
### Input
```
train dataset
0 0 4.0
0 1 3.0
1 0 5.0
1 1 2.0

test dataset
0 2
1 1
```

### Output
```
3.75
2.25
```

---

## Future Enhancements
- **Hybrid Models**: Combine collaborative filtering with content-based approaches.
- **Advanced Metrics**: Add Precision@K, Recall@K, and F1-score for evaluation.
- **Real-time Recommendations**: Optimize the system for real-time prediction.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

