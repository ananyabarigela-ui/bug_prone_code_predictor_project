# Bug-Prone Code Prediction Using Machine Learning

## Overview

This project explores how Machine Learning can be used to analyze software complexity metrics and predict whether a piece of code may be bug-prone.

It extracts features such as lines of code, loops, conditions, variables, comment ratio, and nesting depth. These features are passed to trained Machine Learning models to generate a prediction.

The project also includes a Streamlit web application that allows users to analyze code, view complexity metrics, and identify selected suspicious code patterns.

## Key Features

- **Code Complexity Analysis:** Extracts basic metrics from submitted code.
- **Bug-Proneness Prediction:** Uses a trained Machine Learning model to classify code.
- **Static Pattern Checks:** Identifies selected suspicious coding patterns using rule-based checks.
- **Model Comparison:** Compares Logistic Regression, Decision Tree, and Random Forest.
- **Performance Evaluation:** Reports accuracy, precision, recall, and F1-score.
- **Interactive Dashboard:** Presents predictions, metrics, warnings, and model results through Streamlit.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib

## Machine Learning Workflow

1. Generate a dataset containing software complexity metrics.
2. Prepare the input features and target labels.
3. Split the dataset into training and testing sets.
4. Train multiple Machine Learning models.
5. Evaluate and compare model performance.
6. Save the best-performing model.
7. Use the trained model in the Streamlit application to analyze submitted code.

## Dataset

The current version uses a **synthetic dataset of 500 samples** with seven input features and a binary target label.

The labels are generated using a complexity-based formula with added noise. This makes the dataset useful for demonstrating the Machine Learning workflow, but it does not represent verified real-world software defects.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/bug-prone-code-predictor.git
cd bug-prone-code-predictor
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn streamlit matplotlib
```

### 3. Generate the dataset

```bash
python generate_dataset.py
```

### 4. Train the models

```bash
python train_model.py
```

### 5. Launch the application

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal to use the application.

## Project Structure

```text
bug-prone-code-predictor/
├── data/
│   └── bug_dataset.csv
├── models/
│   ├── best_model.pkl
│   └── model_comparison.csv
├── generate_dataset.py
├── train_model.py
├── app.py
└── README.md
```

## Limitations

- The model is trained on synthetic labels rather than confirmed software defects.
- Code metrics are estimated using basic text-based analysis.
- The prediction is an experimental classification result, not a guarantee that code is safe or contains a bug.
- Static pattern checks cover only a limited set of suspicious patterns.

## Future Improvements

- Train and evaluate models using real, labeled software defect datasets.
- Improve code parsing and support for multiple programming languages.
- Add more static analysis checks.
- Improve feature engineering and model evaluation.
- Package the application for easier deployment.

## Skills Demonstrated

Python programming, data preprocessing, feature engineering, supervised Machine Learning, model evaluation, model comparison, basic static code analysis, and Streamlit application development.
