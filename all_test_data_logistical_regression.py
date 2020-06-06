# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 23:04:00 2020

use logistical regression model to create the final answer to this issue using the whole data set. Processing of data will be done as before

@author: denis
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from data_cleaning import train_data

X = train_data.drop(['Survived'],axis=1)
y = train_data['Survived']

titanic_full_logistical = LogisticRegression()
titanic_full_logistical.fit(X,y)

#%% import and clean test data as was done for training data

test_data = pd.read_csv('test.csv')

#%%

test_data['class'] = test_data['Pclass'].apply(lambda x : str(x))
test_data['sexclass'] = test_data['Sex'] + test_data['class'] 


def impute_age(cols):
    Age = cols[0]
    sexclass = cols[1]
    if pd.isnull(Age):
        if sexclass == 'female1':
            return 35
        elif sexclass == 'female2':
            return 29
        elif sexclass == 'female3':
            return 22
        elif sexclass == 'male1':
            return 41
        elif sexclass =='male2':
            return 31
        else:
            return 27
    else:
        return Age
    
test_data['Age'] = test_data[['Age','sexclass']].apply(impute_age,axis=1)
test_data.drop(['sexclass'],axis=1,inplace=True)
test_data.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)

sex = pd.get_dummies(test_data['Sex'],drop_first=True)
embarked = pd.get_dummies(test_data['Embarked'],drop_first=True)
test_data = pd.concat([test_data,sex,embarked],axis=1)

#%% Don't forget to finish cleaning the data you potato

test_data.drop(['Sex','Embarked'],axis=1,inplace=True)

#%% NaN in fares, different to in the case of main data set. Need to replace with appropriate value

fareval = X.groupby('Pclass').mean()

def impute_fare(cols):
    Fare = cols[0]
    Pclass = cols[1]
    if pd.isnull(Fare):
        if Pclass == 1:
            return 84
        elif Pclass == 2:
            return 21
        else:
            return 14
    else:
        return Fare
    
test_data['Fare'] = test_data[['Fare','Pclass']].apply(impute_fare,axis=1)

#%%Now the data can be used to predict survival

final_pred = titanic_full_logistical.predict(test_data)

#%% Create the CSV for submission

outcomes = pd.DataFrame(final_pred)
outcomes = outcomes.rename(columns={0:'Survived'})

Titanic_predictions = pd.concat([test_data['PassengerId'],outcomes],axis=1)
Titanic_predictions.set_index('PassengerId',inplace=True)

Titanic_predictions.to_csv('Titanic_predictions.csv')



