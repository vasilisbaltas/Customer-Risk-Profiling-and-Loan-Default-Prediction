# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np



### importing the necessary datasets
client_features = pd.read_csv('Modified_Client_Features.csv')
client_quality = pd.read_csv('Client_Risk_Quality.csv')
loan_defaults = pd.read_csv('Loan_Defaults.csv')
loan_attributes = pd.read_csv('Modified_Loan_Attributes.csv')



#########################   FEATURE ENGINEERING    ###########################




### join the Modefied_Loan_Attributes and Loan_Defaults datasets and create a
### DEFAULT column indicating whether the loan has defaulted
loan_defaults['DEFAULT'] = 1
df = loan_attributes.merge(loan_defaults, how = 'outer', on= ['UID','RECORDNUMBER'])
df['DEFAULT'] = df['DEFAULT'].apply(lambda x:0 if x!=1 else x)



### create a list with clients UIDs
clients = list(np.unique(df.UID))
risky_clients = dict()


### we define a dictionary of risky clients - risky client is one that more 
### than 50% of his loans have defaulted
for client in clients:
    defaulted_loans = df[df.UID == client]['DEFAULT'].isin([1]).sum()
    total_loans = len(df[df.UID == client]['DEFAULT'])
                         
    if defaulted_loans/total_loans >= 0.50:   ### setting a 50% threshold gives
        risky_clients[client] = client        ### us 10.519 risky clients
    else:                                     ### out of 16.347 total clients
        pass



                                              ### 0 indicates a safe client
df['RISK DEFINITION'] = 0                     ### in the df dataset while 1
for i in range(df.shape[0]):                  ### indicates a risky one
    if df.loc[i,'UID'] in (risky_clients.values()):
        df.loc[i,'RISK DEFINITION'] = 1
    else:
        pass



### drop some columns to avoid data leakage
df = df.drop(columns = ['ACCSTARTDATE','LAST_MONTH','DVAL','DMON'])



### for some defaulted loans(7358) the OPENBALANCE AND REPAYPERIOD values are missing
### replace NaNs of OPENBALANCE and REPAYPERIOD with the respective mean feature values
df['OPENBALANCE'] = df['OPENBALANCE'].fillna(int(df['OPENBALANCE'].mean()))
df['REPAYPERIOD'] = df['REPAYPERIOD'].fillna(int(df['REPAYPERIOD'].mean()))





### merge df and Client_Risk_Quality with an inner join
df = df.merge(client_quality, on = 'UID')


### drop the SCORE column as well because our classifier's task is to give us a
### similar score
df = df.drop(['SCORE'],axis=1)

### encode CLASS variable from Client_Risk_Quality dataset
df['CLASS'] = df['CLASS'].apply(lambda x: 0 if x== 'STANDARD' else 1)


### merge df and Modified_Client_Features datasets - now we have all our datasets united
### after this merge df has 43993 rows and 129 columns
df = df.merge(client_features, on = 'UID')
df = df.drop(['UID'], axis=1)              ### no longer needed




### define categorical features to encode
encode_columns = ['F_23','F_43','F_44','F_114','F_125','F_128',
                  'F_143','F_149','F_154','F_173','F_183','F_186',
                  'F_189','F_193','F_196','F_197','F_202','F_217','F_218']


mappings = {'D':7, 'U':8, 'ND':9, '?':10,}

for col in encode_columns:
    df[col].replace(mappings, inplace=True)





### our data is ready for deployment of ML models - 43993 loans with 129 features
### 20000 of them have defaulted and 23993 have not  
    
#df.to_csv('ML_Dataset.csv',encoding='utf-8',index=False)






