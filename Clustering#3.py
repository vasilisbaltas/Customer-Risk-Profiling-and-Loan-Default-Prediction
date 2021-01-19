# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE




### importing the necessary datasets
client_features = pd.read_csv('Modified_Client_Features.csv')
loan_defaults = pd.read_csv('Loan_Defaults.csv')
loan_attributes = pd.read_csv('Modified_Loan_Attributes.csv')


### join the Modefied_Loan_Attributes and Loan_Defaults datasets and create a
### DEFAULT column indicating whether the loan has defaulted
loan_defaults['DEFAULT'] = 'Default'
d1 = loan_attributes.merge(loan_defaults, how = 'outer', on= ['UID','RECORDNUMBER'])
d1['DEFAULT'] = d1['DEFAULT'].apply(lambda x:'No Default' if x!='Default' else x)



### create a list with clients UIDs
clients = list(np.unique(d1.UID))
risky_clients = dict()


### we define a dictionary of risky clients - risky client is one that more 
### than 50% of his loans have defaulted
for client in clients:
    defaulted_loans = d1[d1.UID == client]['DEFAULT'].isin(['Default']).sum()
    total_loans = len(d1[d1.UID == client]['DEFAULT'])
                         
    if defaulted_loans/total_loans >= 0.50:   ### setting a 50% threshold gives
        risky_clients[client] = client        ### us 10.519 risky clients
    else:                                     ### out of 16.347 total clients
        pass




                                              ### 'Safe' indicates a safe client
client_features['RISK DEFINITION'] = 'Safe'   ### in the Client_Features dataset
for i in range(client_features.shape[0]):
    if client_features.loc[i,'UID'] in (risky_clients.values()):
        client_features.loc[i,'RISK DEFINITION'] = 'Risky'




#######################   CLUSTERING OF CLIENTS   ############################




### define categorical features to exclude in the clustering process
remove_columns = ['UID','F_23','F_43','F_44','F_114','F_125','F_128','F_131','F_133',
                  'F_134','F_143','F_148','F_149','F_154','F_159','F_173','F_183','F_186',
                  'F_189','F_193','F_196','F_197','F_202','F_210','F_212','F_217','F_218','F_230']


### finally 96 out of the 289 features of the initial dataset are going to be used
### in the clustering process


class MakeClusters():
    
    
    
    ### the class requires as inputs the modified Client_Features dataset and 
    ### the categorical features to remove
    def __init__(self,df,remove_columns):
        
        self.remove_columns = remove_columns
        self.df = df.drop(columns = remove_columns)
        self.imput = self.df.get_values()[:,:-1]



    ### this method performs a t-SNE dimensionality reduction and then provides
    ### us with with the 'elbow' diagramm in order to choose the optimal number
    ### of clusters based on the sum of squared error of all clusters(inertia)    
    def optimal_clusters_number(self):
        
        
        self.y = TSNE(n_components=2).fit_transform(self.imput)    ### prefer t-SNE algorithm to PCA because
                                                                   ### of its capacity to capture nonlinear
                                                                   ### relationships
        inertia = []
        for k in range(1,16):
            km = KMeans(n_clusters=k)
            km = km.fit(self.y)
            inertia.append(km.inertia_)
    
        plt.plot(range(1,16),inertia,'bx-',color='darkblue')
        plt.xlabel('Number of centroids')
        plt.ylabel('Inertia')
        plt.title('Elbow Method For Optimal k')
        plt.show()
        
    
    
    ### divide clients into clusters - we need to input the number of clusters
    ### which we can determine with the above method    
    def get_clusters(self,n_clusters):
        
        kmeans = KMeans(n_clusters = n_clusters, random_state=33).fit(self.y)
        self.df['CLUSTERS'] = kmeans.labels_       ### the assigned cluster labels
    
    
        palette = ['maroon','navy','blue','mediumpurple','darkviolet','purple','crimson','teal','lime']
    
        plt.figure(figsize=(12,8))
        sns.set_style("white")
        sns.scatterplot(
            x= self.y[:,0], y=self.y[:,1],
            hue="CLUSTERS",
            data = self.df,
            legend="full",
            alpha=0.3, palette = palette
            )

        return self.df    ### the method also returns the dataset with an extra
                          ### column indicating the assigned cluster
 
    
    
    ### this method gives us a t-SNE visualization for the 'risky' and 'safe' clients
    ### risky clients are denoted as '1'
    def get_defaulters(self):
        
        plt.figure(figsize=(12,8))
        sns.set_style("white")
        sns.scatterplot(
            x= self.y[:,0], y=self.y[:,1],
            hue="RISK DEFINITION",
            data = self.df,
            legend="full",
            alpha=0.3
            )




    ### this method returns a dictionary with the ratios of risky clients per cluster
    def defaulters_per_cluster(self):
         
         self.percentages = dict()
         self.df['RISK DEFINITION'] = self.df['RISK DEFINITION'].apply(lambda x: 0 if x=='Safe' else 1)
         
         perc = self.df['RISK DEFINITION'].sum()          ### the total number of risky clients
         x = np.unique(self.df['CLUSTERS'])
             
         self.percentages[x[0]] = self.df.groupby('CLUSTERS')['RISK DEFINITION'].sum()/ perc
             
        
         return self.percentages    
 
    
    ### this method returns a matrix with the average feature values per cluster -
    ### this means that we are going to have a mean value for every feature of the dataset
    ### and for every cluster 
    ### with this table we can compare tha characteristics of interesting clusters 
    ### the method also returns a dictionary with the minimum and maximum average
    ### values for every feature 

    def comparison_table(self):
        
        tablet = np.zeros((len(np.unique(self.df['CLUSTERS'])),self.imput.shape[1]))
        minmax = dict()
        
        cols = list(self.df.columns)
        cols.remove('CLUSTERS')
        cols.remove('RISK DEFINITION')
        
        for i in range(tablet.shape[0]):
            for j in range(tablet.shape[1]):
                
                tablet[i,j] = self.df[self.df['CLUSTERS'] == i][cols[j]].mean()
        
        for j in range(tablet.shape[1]):
            minmax[cols[j]] = np.min(tablet[:,j]), np.max(tablet[:,j])

        tablet = pd.DataFrame(tablet, columns = cols)
        
        
        return tablet, minmax       
    



##############################################################################



    
    
### defining clusters with the use of the above class    
    
check = MakeClusters(client_features,remove_columns)    
check.optimal_clusters_number()       ### 9 clusters were chosen as the optimal number
new_data = check.get_clusters(9)  
check.get_defaulters()    
percentages = check.defaulters_per_cluster()   ### clusters 1,6,0,2 seem to entail the  
print(percentages)                             ### biggest ratios of risky clients but this might
mean_values, min_max = check.comparison_table()    ### not always be reproduced in the same way     
#print(mean_values)                                 ### because of the stochastic nature of KMeans algorithm     
#print(min_max)                                     ### and t-SNE algorithm     


### after examining mean_values and min_max, features like F_1,F_3,F_7,F_21,F_272,F_236
### seem to have a common pattern for clusters 1 and 6 that represent the biggest ratios
### of risky clients
  

