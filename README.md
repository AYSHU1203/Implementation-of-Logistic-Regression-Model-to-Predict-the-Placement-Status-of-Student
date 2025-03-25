# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages.

2.Print the present data and placement data and salary data.

3.Using logistic regression find the predicted values of accuracy confusion matrices.

4.Display the results.

## Program:
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Developed by: Ayshwariya J

RegisterNumber: 212224230030
```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('Placement_Data.csv')
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no", "salary"], axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x = data1.iloc[:, :-1]
x
y = data1["status"]
y
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = (y_test, y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test, y_pred)
print(classification_report1)
lr.predict([[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]])
```
## Output:
Placement Data

![Image-1](https://github.com/user-attachments/assets/1d27e753-ce1f-40a1-8ed0-01fbc91bfaa9)

Checking the null() function

![Image-2](https://github.com/user-attachments/assets/8eeffc2a-2bd3-43e5-8099-a0df78941aa3)

Print Data:

![Image-3](https://github.com/user-attachments/assets/7ef13c18-3637-4f20-aa9f-2afc3db8ab39)

Y_prediction array

![Image-4](https://github.com/user-attachments/assets/4506ac35-8245-4f8e-a46f-5aec9a593771)

Accuracy value

![Image-5](https://github.com/user-attachments/assets/e4368f1b-3ae4-4b85-8900-de173bcf4c9c)

Confusion array

![Image-6](https://github.com/user-attachments/assets/1cb14fa5-27a8-4733-91c3-b4dfcb6cb008)

Classification Report

![Image-7](https://github.com/user-attachments/assets/85a687fd-29a7-4a0f-86b9-942d6c176237)

Prediction of LR

![Image-8](https://github.com/user-attachments/assets/b671ad50-da16-4f52-9c4e-38e5b942f04a)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
