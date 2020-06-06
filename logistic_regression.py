# -*- coding: utf-8 -*-
"""
Logistic regression attempt at interpreting the titanic data set from kaggle.com for first attempt at comp submission

ensure data_cleaning.py is run first, else this will fail due to data not in correct format

@author: denis
"""

import pandas as pd
from sklearn.model_selection import train_test_split


#split the titanic training data to see what efficieny we are getting to be able to compare with random forest later on

X = train_data.drop(['Survived'],axis=1)
y = train_data['Survived']

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=13)


#%% time to create a logistical regression model

from sklearn.linear_model import LogisticRegression

titanic_log_mdl = LogisticRegression()

titanic_log_mdl.fit(X_train,y_train)

y_pred = titanic_log_mdl.predict(X_test)

#%% evaluate model performance

from sklearn.metrics import confusion_matrix,classification_report
print('Confusion Matrix:')
print(confusion_matrix(y_test,y_pred))
print('Classification Report:')
print(classification_report(y_test,y_pred))




