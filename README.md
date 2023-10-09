# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

# AIM:

To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Equipments Required:

    Hardware – PCs
    Anaconda – Python 3.7 Installation / Jupyter notebook

# Algorithm

    Import the standard libraries.
    Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
    LabelEncoder and encode the dataset.
    Import LogisticRegression from sklearn and apply the model on the dataset.
    Predict the values of array.
    Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
    Apply new unknown values

# Program:

/*
# Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Agalya R
RegisterNumber:  212222040003
*/
```
import pandas as pd
data=pd.read_csv('/Placement_Data(1).csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
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

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy= accuracy_score(y_test,y_pred)#Accuracy Score = (TP+TN)/(TP+FN+TN+FP)
#accuracy_score(y_true,y_pred,normalize=False)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion #11+24=35 -correct predictions,5+3=8 incorrect predictions

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

# Output:
# Placement data
![image](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394395/15345c14-7043-4bf8-8739-7db73ed5acd9)

# Salary data
![image](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394395/4a0dd4dc-6f10-4dac-b7c2-953635f41770)

# Checking the null() function
![image](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394395/6f4b3e68-1d86-47f2-bec7-fca9ba358166)

# Data Duplicate
![image](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394395/c2dcffdf-37ca-4dbb-8fde-2c1aa98fe729)

# Print data
![image](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394395/9effff76-2f10-4c46-a493-74ae941e3dfa)

# Data-status
![image](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394395/6d666a50-1a3f-456f-b5d5-4ad1853f88da)

# y_prediction array
![image](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394395/d20c4e2d-f462-43b2-9763-169e22a49101)

# Accuracy value
![image](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394395/465f4382-0fea-45b9-950d-31d3a5b7640f)

# Confusion array
![image](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394395/bb850dd7-184a-408e-a4a1-6b4fd49157ee)

# Classification report
![image](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394395/c1fbfb4d-07c6-4f6f-852b-1148c5a9b11b)

# Prediction of LR
![image](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394395/340226f4-948c-4593-a27f-fe25fe7700d6)

# Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
