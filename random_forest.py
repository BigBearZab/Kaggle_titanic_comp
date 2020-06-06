# -*- coding: utf-8 -*-
"""
Random forest attempt at solving titanic data set using sklearn

@author: denis
"""
import pandas as pd
from sklearn.model_selection import train_test_split


#split the titanic training data to see what efficieny we are getting to be able to compare with random forest later on

X = train_data.drop(['Survived'],axis=1)
y = train_data['Survived']

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=13)


#%% time to create a logistical regression model

from sklearn.ensemble import RandomForestClassifier

titanic_rfc = RandomForestClassifier(n_estimators=100)

titanic_rfc.fit(X_train,y_train)

y_predrfc = titanic_log_mdl.predict(X_test)

#%% evaluate model performance

from sklearn.metrics import confusion_matrix,classification_report
print('RFC Confusion Matrix:')
print(confusion_matrix(y_test,y_predrfc))
print('RFC Classification Report:')
print(classification_report(y_test,y_predrfc))

#interestingly RFC gives exactly the same values as logistic regression
