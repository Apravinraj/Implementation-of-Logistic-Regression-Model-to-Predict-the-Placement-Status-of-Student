# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student
## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
Hardware – PCs Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Import the required packages and print the present data. Print the placement data and salary data. Find the null and duplicate values. Using logistic regression find the predicted values of accuracy , confusion matrices. Display the results.

## Program:
```
#Developed by: Pravin Raj A
#RegisterNumber: 212222240079

import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:
## Placement Data:
![272192788-43bf5574-366c-4aa8-8bfd-2480d17a20a5](https://github.com/Apravinraj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707879/6c5793c2-d93a-4fda-af90-3be2f4124553)


## Salary Data:

![272193458-4481c3dc-e014-4f48-89c2-a05e28f9449e](https://github.com/Apravinraj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707879/97302a2e-1103-4cdd-9055-6de610c8122d)

## Checking the null() function:
![272193502-f7bb1d3a-5411-4362-91b4-7017fdd1481d](https://github.com/Apravinraj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707879/7705c8b6-b98d-4692-a03e-f9e7b9cc9eec)

## Data Duplicate:

![272193555-6ac841e7-213e-4bc3-87a4-ff0f4818f536](https://github.com/Apravinraj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707879/9f949150-c2f2-4106-9ad2-1198f9eb553d)

## Print Data:

![272193766-90b83a36-7f3e-45bd-8eb6-4f60883f3079](https://github.com/Apravinraj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707879/b04f2f84-cf34-4aa5-b2d5-afaf5280d24e)


## Data-Status:

![272193815-6f57b4f8-1050-4e17-a634-b3e6e6f8a286](https://github.com/Apravinraj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707879/f424b5b1-8726-453b-a179-0e10afe963e3)


## Y_prediction array:

![272194118-390c1349-5449-4d2f-b9ec-27ce117d1395](https://github.com/Apravinraj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707879/d5a6330b-8002-47dd-b160-2964bee14da5)

## Accuracy value:

![272194223-e069c29d-699f-43e4-b302-1c2546e24a67](https://github.com/Apravinraj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707879/566f8128-dcea-4807-a3a1-b6f29e11e033)


## Confusion array:

![272194287-42bdc9e2-8e65-424d-a13d-914a10cb1e40](https://github.com/Apravinraj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707879/eabec26f-7b90-44b2-9b71-49a6b184e5bd)

## Classification Report:

![272194335-1cefb2d7-c140-48a1-ae8e-be95c75652ec](https://github.com/Apravinraj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707879/6aa48aca-4e6c-4de5-97de-5b6ccc3bdb3a)


## Prediction of LR:

![272195036-d09d023b-6312-411e-944d-c165ffa2dbb0](https://github.com/Apravinraj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707879/09cc0842-7c54-4b0b-bb69-943ce900b465)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
