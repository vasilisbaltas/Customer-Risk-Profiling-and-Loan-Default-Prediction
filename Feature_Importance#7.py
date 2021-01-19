# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

### import the preprocessed data
data = pd.read_csv('ML_Dataset.csv')


### random shuffle our dataset
data = data.reindex(np.random.permutation(data.index))


X_train, X_test, y_train, y_test = train_test_split(data.drop('DEFAULT',axis=1),
                                                    data['DEFAULT'], test_size = 0.15)


### we are going to deploy a random forest classifier because this class can provide
### us with feature importance - it is essential to have an interpretable model for
### various reasons; we can adjust/remove features in order to enhance performance, 
### understand potential underlying bias and of course be able to explain to other stakeholders
### our algorithms predictive behaviour

### decicion trees commonly do not require data scaling


def tree_feature_importance(X_train, y_train):
    

    forest = RandomForestClassifier( n_estimators = 300)
    forest.fit(X_train, y_train)


    importances = forest.feature_importances_
    indices = np.argsort(-importances)
    df_imp = pd.DataFrame(dict(feature = X_train.columns[indices],
                      importance = importances[indices]))


    ###plot top 10 most important features
    plt.figure(figsize = (10,5))
    plt.bar(df_imp.loc[:10,'feature'], df_imp.loc[:10,'importance'], color = 'darkblue', orientation = 'vertical')
    plt.xticks(rotation =45)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('10 most important features')
    plt.show()

    ### plot 10 least important features
    plt.figure(figsize = (10,5))
    plt.bar(df_imp.iloc[-10:,0], df_imp.iloc[-10:,1], color = 'darkblue', orientation = 'vertical')
    plt.xticks(rotation =45)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('10 least important features')    
    plt.show()
    
    return forest.score(X_test,y_test)



get_performance = tree_feature_importance(X_train, y_train)     ### the function also returns the accuracy
                                                                ### of the classfier - but this is not our objective
                                                                ### from this model







