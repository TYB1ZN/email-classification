import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

data = pd.read_csv('mail_data.csv')
print(data)

data_head = data.head()
data_tail = data.tail()
data_info = data.info()
data_des = data.describe()
data_isnull = data.isnull()

print(data_head)
print(data_tail)
print(data_info)
print(data_des)
print(data_isnull)

# separating the data as texts and label
X = data['EmailContent'].values
Y = data['Label'].values

# spitting the data into training data & test data
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.2, random_state=0)

# Converting String to Integer
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

# print("Data training",X_train)
# print("Data test",X_test)

# Applying SVM algorithm
classifier = SVC(kernel = 'rbf', random_state = 10)
classifier.fit(X_train, Y_train)
print(classifier.score(X_test, Y_test))

# Predicting on test data
Y_pred = classifier.predict(X_test)

#Printing the results
for i in range(len(Y_pred)):
    print("Email Content:", X_test[i])
    print("Predicted Label:", Y_pred[i])
    print("Actual Label:", Y_test[i])
    print("------------")





