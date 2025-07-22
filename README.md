# Placement Prediction Using Linear Regression

## Overview

This project uses a **Logistic Regression** model to predict whether a student will be placed or not based on their performance in **SSC (10th grade)** and **HSC (12th grade)** exams. The model predicts the likelihood of a student being placed in a job based on these two key features, alongside other parameters such as degree percentage, MBA percentage, etc.

The main goal is to analyze the relationship between the **10th and 12th grade percentages** and the placement status of students. The dataset used in this project contains information about students' academic performance and placement status.

## Dataset

The dataset used in this project is a CSV file named `Job_Placement_Data.csv`. It contains the following columns:
- `gender`: The gender of the student.
- `ssc_percentage`: The percentage obtained in the 10th grade (SSC).
- `ssc_board`: The board from which the student completed their SSC.
- `hsc_percentage`: The percentage obtained in the 12th grade (HSC).
- `hsc_board`: The board from which the student completed their HSC.
- `hsc_subject`: The major subject in HSC.
- `degree_percentage`: The percentage obtained in undergraduate degree.
- `undergrad_degree`: The type of undergraduate degree the student pursued.
- `work_experience`: Whether the student has work experience or not.
- `emp_test_percentage`: The score in the employment test.
- `specialisation`: The area of specialization (marketing/operations) in MBA.
- `mba_percent`: The percentage obtained in MBA.
- `status`: Whether the student was placed (Placed/Not Placed).

## Project Workflow

1. **Data Loading**:
   The dataset is loaded into a Pandas DataFrame using `pd.read_csv()`.

2. **Data Exploration**:
   The dataset is explored to understand its structure, with `df.info()` and `df.head()` displaying initial data insights.

3. **Feature Selection**:
   We select relevant features from the dataset such as `ssc_percentage`, `hsc_percentage`, `degree_percentage`, `emp_test_percentage`, `mba_percent`, and `status` for further analysis and modeling.

4. **Data Visualization**:
   A scatter plot is created to visualize the relationship between **SSC Percentage** and **HSC Percentage** with respect to placement status.

5. **Data Preprocessing**:
   - Split the data into features (`X`) and target variable (`y`).
   - Standardize the features using `StandardScaler` to scale the data for the Logistic Regression model.

6. **Model Training**:
   - The data is split into training and testing sets using `train_test_split()`.
   - A **Logistic Regression** model is trained using the training data (`x_train`, `y_train`).

7. **Model Evaluation**:
   - The model's performance is evaluated by predicting the placement status on the test set (`y_pred`).
   - The **accuracy score** is computed using `accuracy_score()`.

8. **Decision Boundary Plot**:
   - A decision boundary plot is generated to visualize how the Logistic Regression model classifies the data into "Placed" and "Not Placed" categories based on `ssc_percentage` and `hsc_percentage`.

9. **Model Serialization**:
   The trained model is saved using **pickle** for future use. The model is serialized to a file named `model.pkl`.

## Requirements

To run the project, you need to install the following dependencies:
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `mlxtend`
- `pickle`

You can install them using pip:
```bash
pip install pandas numpy matplotlib scikit-learn mlxtend

## 📩 Contact & Contribution

For contributions, feedback, or collaborations, feel free to reach out:  
- **Author:** Shreya Gupta
- **Email:** shreyagupta119809@gmail.com
- **LinkedIn:** https://www.linkedin.com/in/shreya-gupta-2a6a292ab

