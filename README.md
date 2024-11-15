# Employee Exit Prediction Model

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Dataset Description](#dataset-description)
- [Installation](#installation)

- [Data Preprocessing](#data-preprocessing)
- [Model Selection and Evaluation](#model-selection-and-evaluation)
- [Results](#results)

- [Contributing](#contributing)


## Project Overview
The Employee Exit Prediction Model aims to predict whether an employee will leave the company based on various features such as demographics, job satisfaction, and performance metrics. This project utilizes machine learning techniques to analyze employee data and provide insights that can help organizations improve retention strategies.

## Technologies Used
- **Programming Language:** Python
- **Libraries:**
  - `pandas` for data manipulation and analysis
  - `numpy` for numerical operations
  - `scikit-learn` for machine learning algorithms and model evaluation
  - `xgboost` for advanced boosting algorithms
  - `seaborn` and `matplotlib` for data visualization
  - `joblib` for model serialization
- **Environment:** Jupyter Notebook

## Dataset Description
The dataset consists of employee records with various features, including:
- **Demographic Information:** Age, gender, marital status, etc.
- **Job-related Features:** Job role, years at the company, job satisfaction, performance rating, etc.
- **Financial Metrics:** Monthly income, promotions count, etc.
- **Exit Status:** The target variable indicating whether the employee left the company (1) or stayed (0).

### Data Files
- `train.csv`: The training dataset used to build the model.
- `test.csv`: The test dataset used for predictions.

## Installation
To run this project, you need to have Python installed on your machine. You can install the required libraries using pip:
pip install pandas numpy scikit-learn xgboost seaborn matplotlib joblib




## Data Preprocessing
The data preprocessing steps include:
- **Loading Data:** Importing the training and test datasets.
- **Exploratory Data Analysis (EDA):** Understanding the data through visualizations and statistical summaries.
- **Handling Missing Values:** Filling missing values in continuous features with the mean and categorical features with the mode.
- **Encoding Categorical Features:** Using label encoding for binary features and one-hot encoding for multi-category features.
- **Feature Engineering:** Creating new features such as `remote_work_distance`, `promotion_rate`, and `income_per_dependent`.
- **Scaling Numerical Features:** Standardizing numerical features using `StandardScaler`.

## Model Selection and Evaluation
The following models were evaluated:
- Decision Tree
- Random Forest
- XGBoost
- Naive Bayes
- K-Nearest Neighbors (KNN)
- AdaBoost

### Hyperparameter Tuning
Grid Search was used to find the best hyperparameters for each model based on the weighted F1 score.

### Model Evaluation Metrics
- **Accuracy:** The proportion of correct predictions.
- **F1 Score:** The harmonic mean of precision and recall, providing a balance between the two.

## Results
The best model was saved as `best_model.joblib`. The results of the model evaluation, including the best parameters and scores for each model, are printed in the notebook.



## Contributing
Contributions are welcome! If you have suggestions for improvements or new features, please create an issue or submit a pull request.


