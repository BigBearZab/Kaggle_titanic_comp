# -*- coding: utf-8 -*-
"""
Data cleaning for titanic data set

@author: denis
"""


import pandas as pd

import seaborn as sns

#ensure we are in correct directory

# os.chdir('C:\Users\denis\Documents\Python\DataScience\Titanic')


train_data = pd.read_csv('train.csv')

#%%
# time to start some visualisations

nan_df = train_data.isnull()
sns.heatmap(nan_df,cbar=False,cmap='plasma')

# cabin value is very poorly populated => remove. Also remove name and ticket as neither are beneficial to analysis

train_data.drop(['Cabin','Name','Ticket'],axis=1,inplace=True)

sns.heatmap(train_data.corr())

#%%

#to allow for the data to be useable average age required for male and femaleby class

aver_age = train_data.groupby(['Pclass','Sex']).mean().reset_index()
aver_age['class'] = aver_age['Pclass'].apply(lambda x : str(x))
aver_age['sexclass'] = aver_age['Sex'] + aver_age['class'] 
aver_age = aver_age[['Age','sexclass']]

train_data['class'] = train_data['Pclass'].apply(lambda x : str(x))
train_data['sexclass'] = train_data['Sex'] + train_data['class'] 


#%%

#replace NaN values with age


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
    
train_data['Age'] = train_data[['Age','sexclass']].apply(impute_age,axis=1)
train_data.drop(['sexclass'],axis=1,inplace=True)

#%%
    
# =============================================================================
# nan_df2 = train_data.isnull()
# sns.heatmap(nan_df2,cbar=False,cmap='plasma')   
# =============================================================================

#%% now need to change variables to a format readable to the machine



sex = pd.get_dummies(train_data['Sex'],drop_first=True)
embarked = pd.get_dummies(train_data['Embarked'],drop_first=True)

train_data.drop(['Sex','Embarked'],axis=1,inplace=True)
#%%

train_data = pd.concat([train_data,sex,embarked],axis=1)

    
    