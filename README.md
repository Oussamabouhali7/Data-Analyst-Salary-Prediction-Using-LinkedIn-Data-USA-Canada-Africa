# 💰 Data Analyst Salary Prediction — Machine Learning

## 📌 Project Overview

This project aims to **predict the salaries of employees working in different Data Analyst-related positions** using various pieces of information collected from LinkedIn data.

The project covers a complete **Machine Learning workflow**, from data cleaning and preparation to exploratory data analysis, model training, and model evaluation.

The datasets used in this project cover three geographical regions:

* 🇺🇸 USA
* 🇨🇦 Canada
* 🌍 Africa

The three datasets are combined into a single dataset for analysis and machine learning.

---

## 🎯 Project Objectives

The main objectives of this project are:

* Collect and combine salary datasets.
* Clean and prepare the data.
* Handle missing values.
* Convert salaries into US dollars.
* Perform Exploratory Data Analysis (EDA).
* Transform categorical variables into numerical values.
* Train several Machine Learning regression models.
* Compare the performance of the different models.
* Identify the best model for salary prediction.

---

## 📊 Dataset

The project uses three CSV files representing three different regions:

```text
USA
Canada
Africa
```

The datasets are combined using the Pandas `concat()` function.

```python
import pandas as pd

df = pd.concat([df_usa, df_canada, df_africa])
```

This allows all observations from the three regions to be analyzed together.

---

## 🧹 Data Cleaning and Preparation

Before training the Machine Learning models, the dataset needs to be cleaned and prepared.

One of the main issues addressed during this step is the presence of missing values in the `Salary` column.

### 🔹 Missing Values — Mode

The first imputation method uses the **mode** to replace missing salary values.

The mode is calculated according to the type of job, allowing missing salary values to be replaced with a value corresponding to the specific job category.

---

### 🔹 Missing Values — Mean

Another method used in the project is **mean imputation**.

Missing values in the `Salary` column are replaced by the corresponding average salary.

---

### 💵 Salary Conversion

Since the datasets cover different geographical regions, salaries may be expressed using different currencies.

A salary conversion step is therefore performed to convert the salaries into **US dollars (USD)**, making the values easier to compare across regions.

---

## 📈 Exploratory Data Analysis

After preparing the dataset, an **Exploratory Data Analysis (EDA)** is performed.

The purpose of this stage is to better understand the characteristics and distribution of the data, particularly salary distribution.

### 📦 Boxplot

A **boxplot** is used to visualize the distribution of salaries.

It can help identify:

* The median salary.
* The spread of salary values.
* Potential outliers.
* The overall distribution of salaries.

---

### 📊 Histogram

A **histogram** is also used to analyze salary distribution.

It provides a visual representation of how salary values are distributed across different ranges.

---

## 🤖 Machine Learning Models

Four different Machine Learning regression models are implemented and compared:

1. **Linear Regression**
2. **KNN Regression**
3. **Decision Tree Regression**
4. **Ridge Regression**

These four models are trained using the same input variables and target variable.

---

## 1. 📈 Linear Regression

**Linear Regression** is used to model the relationship between the input variables and the target salary.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

---

## 2. 🔍 KNN Regression

**K-Nearest Neighbors Regression (KNN Regression)** predicts salary values based on the nearest observations in the dataset.

```python
from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

---

## 3. 🌳 Decision Tree Regression

**Decision Tree Regression** uses a tree-based structure to make predictions based on different features.

```python
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

---

## 4. 📐 Ridge Regression

**Ridge Regression** is a regularized version of Linear Regression.

```python
from sklearn.linear_model import Ridge

model = Ridge()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

---

## 🔄 Data Encoding

Machine Learning algorithms generally require numerical input data.

The project uses **`LabelEncoder()`** to transform categorical data into numerical values.

Example:

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

df["Job"] = encoder.fit_transform(df["Job"])
```

This transformation makes categorical variables usable by the Machine Learning models.

---

## 🧪 Train-Test Split

The dataset is divided into two subsets:

* **80% Training Data**
* **20% Testing Data**

```text
                Complete Dataset
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
      80% Training             20% Testing
          │                         │
          ▼                         ▼
     Model Training            Model Testing
```

The training dataset is used to train the models, while the testing dataset is used to evaluate their ability to make predictions on unseen data.

---

## 🎯 Features and Target

The variables used as inputs (`X`) are used by the four regression models.

The target variable (`y`) is the **salary that the models need to predict**.

```python
X = df[features]

y = df["Salary"]
```

The data is then divided into training and testing sets:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```

---

## 📊 Model Evaluation

After training the four models, their performance is evaluated and compared.

| Model                    | Type       |
| ------------------------ | ---------- |
| Linear Regression        | Regression |
| KNN Regression           | Regression |
| Decision Tree Regression | Regression |
| Ridge Regression         | Regression |

The objective is to determine which model provides the best salary predictions.

### 🏆 Best Model

According to the project evaluation, **KNN Regression is the best-performing model among the tested models**.

```text
🏆 Best Model: KNN Regression
```

---

## 🔬 Project Workflow

The complete Machine Learning workflow can be summarized as follows:

```text
                LinkedIn Data
                      │
                      ▼
            ┌──────────────────┐
            │   CSV Datasets   │
            │ USA / Canada /   │
            │     Africa       │
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────┐
            │ Data Combination │
            │   pd.concat()    │
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────┐
            │  Data Cleaning   │
            │ Missing Values   │
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────┐
            │ Salary Conversion│
            │      → USD       │
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────┐
            │       EDA        │
            │ Boxplot /        │
            │ Histogram        │
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────┐
            │ Data Encoding    │
            │  LabelEncoder    │
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────┐
            │ Train / Test     │
            │     80 / 20      │
            └────────┬─────────┘
                     │
                     ▼
       ┌──────────────────────────────┐
       │      Machine Learning        │
       │                              │
       │ • Linear Regression          │
       │ • KNN Regression             │
       │ • Decision Tree Regression   │
       │ • Ridge Regression           │
       └──────────────┬───────────────┘
                      │
                      ▼
             ┌─────────────────┐
             │    Evaluation   │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ 🏆 KNN Regression│
             │   Best Model    │
             └─────────────────┘
```

---

## 📁 Project Structure


```text
data-analyst-salary-prediction/
│
├── data/
│   ├── usa.csv
│   ├── canada.csv
│   └── africa.csv
│
├── notebooks/
│   └── salary_prediction.ipynb
│
├── images/
│   ├── boxplot.png
│   └── histogram.png
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/data-analyst-salary-prediction.git
```

### Navigate to the Project

```bash
cd data-analyst-salary-prediction
```

### Create a Virtual Environment

```bash
python -m venv venv
```

### Activate the Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🛠️ Technologies Used

The project uses Python-based Data Science and Machine Learning technologies.

### Programming Language

* Python 🐍

### Data Processing

* Pandas 🐼

### Data Visualization

* Boxplot
* Histogram

### Machine Learning

* Scikit-learn
* Linear Regression
* KNN Regression
* Decision Tree Regression
* Ridge Regression

### Data Preprocessing

* Missing value imputation
* Label Encoding
* Train/Test Split

---

## 🚀 How to Run the Project

The project can be executed following these main steps:

```text
1. Load the datasets
        ↓
2. Combine the datasets
        ↓
3. Clean the data
        ↓
4. Handle missing values
        ↓
5. Convert salaries to USD
        ↓
6. Perform EDA
        ↓
7. Encode categorical variables
        ↓
8. Split the dataset
        ↓
9. Train Machine Learning models
        ↓
10. Evaluate the models
        ↓
11. Select the best model
```

---

## 📌 Results

Four regression models were tested:

| Model                    | Result            |
| ------------------------ | ----------------- |
| Linear Regression        | Evaluated         |
| KNN Regression           | 🏆 **Best Model** |
| Decision Tree Regression | Evaluated         |
| Ridge Regression         | Evaluated         |

Based on the evaluation performed in the project, **KNN Regression achieved the best result** among the tested models.

---

## 💡 Conclusion

This project demonstrates a complete Machine Learning approach for **Data Analyst salary prediction**.

The project starts with datasets covering the **USA, Canada, and Africa**, which are combined into a single dataset. The data is then cleaned by handling missing values and converting salaries into USD.

An Exploratory Data Analysis phase is performed using visualizations such as **boxplots and histograms** to better understand the salary data.

Finally, four regression models are trained and evaluated:

* Linear Regression
* KNN Regression
* Decision Tree Regression
* Ridge Regression

Among these models, **KNN Regression is identified as the best-performing model** according to the project's evaluation.

---

## 👨‍💻 Author

**Bouhali Oussama**

---

## ⭐ Keywords

```text
Machine Learning
Data Science
Data Analysis
Salary Prediction
Data Analyst
LinkedIn Data
Regression
KNN Regression
Linear Regression
Decision Tree Regression
Ridge Regression
Python
Pandas
Scikit-learn
Data Cleaning
Exploratory Data Analysis
EDA
```
