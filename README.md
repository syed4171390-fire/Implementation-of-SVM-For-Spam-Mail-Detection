# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries
2. Create a sample email dataset
3. Convert labels to binary values
4. Split data into features and labels
5. Split into training and testing sets
6. Convert text data into numerical form using TF-IDF
7. Train SVM model
8. Make predictions
9. Evaluate the model

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: 
RegisterNumber:  
*/
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


data = pd.read_csv("C:/Users/acer/Downloads/spam (1).csv", encoding='latin-1')


data = data[['v1', 'v2']]
data.columns = ['label', 'message']


data['label'] = data['label'].map({'ham':0, 'spam':1})


X = data['message']
y = data['label']


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = SVC(kernel='linear')
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


## Output:
<img width="274" height="23" alt="ML IMAGESSSS" src="https://github.com/user-attachments/assets/da11c97c-d0bb-4b20-b2b8-464a4259a50e" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
