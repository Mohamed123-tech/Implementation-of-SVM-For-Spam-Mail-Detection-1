# Implementation-of-SVM-For-Spam-Mail-Detection
## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
Hardware – PCs

Google collab

# Algorithm
1.Import chardet and find the encoding of the dataset.

2.Import other necessary libraries and upload the csv file in the complier.

3.Find head,info and null elements of the dataset.

4.Using CounterVectorizer and SVC find the y prediction array and accuracy .

5.End the Program.

# Program:
/*
Program to implement the SVM For Spam Mail Detection..

Developed by: 212220040081

RegisterNumber:  logeshwaran s

*/

import chardet

file='/content/spam.csv'

with open(file,'rb') as rawdata:

result = chardet.detect(rawdata.read(100000))

result

import pandas as pd

data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer()

x_train=cv.fit_transform(x_train)

x_test=cv.transform(x_test)

from sklearn.svm import SVC

svc=SVC()

svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)

y_pred

from sklearn import metrics

accuracy=metrics.accuracy_score(y_test,y_pred)

accuracy


Output:
1. Result output
![image](https://github.com/ATHDY005/Implementation-of-SVM-For-Spam-Mail-Detection/assets/84709944/3785e0f4-10ec-466a-b1bf-94283bac0562)

2. data.head()
![image](https://github.com/ATHDY005/Implementation-of-SVM-For-Spam-Mail-Detection/assets/84709944/a3e14574-25c2-4d9d-b1df-a92f7292e07b)


3. data.info()
![image](https://github.com/ATHDY005/Implementation-of-SVM-For-Spam-Mail-Detection/assets/84709944/1caddfee-f990-43ec-a6fb-6c753c7197cd)


4. data.isnull().sum()
![image](https://github.com/ATHDY005/Implementation-of-SVM-For-Spam-Mail-Detection/assets/84709944/c18db785-480e-4a82-8044-b5a2543d6396)


5. Y_prediction value
![image](https://github.com/ATHDY005/Implementation-of-SVM-For-Spam-Mail-Detection/assets/84709944/f4d9070f-7291-4c24-b224-0cbab840954b)


6. Accuracy value
![image](https://github.com/ATHDY005/Implementation-of-SVM-For-Spam-Mail-Detection/assets/84709944/5fc8b022-ea77-4523-adc4-c13c13eef2ae)


# Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.

