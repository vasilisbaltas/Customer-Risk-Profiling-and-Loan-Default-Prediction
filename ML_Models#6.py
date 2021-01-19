# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
np.random.seed(33)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score,confusion_matrix
from statistics import mean,stdev
import random
random.seed(33)
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import h5py

### some tricks for reproducibility

os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['PYTHONHASHSEED']=str(33)
import tensorflow as tf
tf.compat.v1.set_random_seed(33)

from keras import backend as K
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)




### import the preprocessed data
data = pd.read_csv('ML_Dataset.csv')


### random shuffle our dataset
data = data.reindex(np.random.permutation(data.index))


### we have 20000 defaulted loans and 23993 not-defaulted loans -
### split in training and test set with a test size of 15% - 
### in a real life scenario with imbalanced data we should stratify the datasets
X_train, X_test, y_train, y_test = train_test_split(data.drop('DEFAULT',axis=1),
                                                    data['DEFAULT'], test_size = 0.15)



### it is important to bring our features to the same scale
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)






################  create a neural network classifier with Keras ###############



from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout



### this class creates a Neural Network Classifier and requires as inputs the
### training and test datasets
class ANN():
    
    
    def __init__(self, X_train, X_test, y_train, y_test):
        
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        model1 = Sequential()
        model1.add(Dense(2*self.X_train.shape[1],activation='relu',input_shape=(self.X_train.shape[1],)))
        model1.add(Dense(int(0.65*self.X_train.shape[1]),activation='relu'))
        model1.add(Dropout(0.25))
        model1.add(Dense(int(0.33*self.X_train.shape[1]),activation = 'relu'))
        model1.add(Dense(1,activation='sigmoid'))   
        
        
        ### we create an initial clasifier in order to examine the effect of training epochs
        ### on model's accuracy - we train for a total of 10 epochs
        model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.history = model1.fit(self.X_train, self.y_train,epochs = 10, validation_split = 0.2)
 
   
    
    
    ### this method plots the impact of training epochs on accuracy
    def examine_training_epochs(self):
        
        plt.figure()
        plt.plot(self.history.history['val_accuracy'], color = 'darkblue')
        plt.title('Model Accuracy')
        plt.ylabel('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.show()

        
     
        
        
    ### with this method we can obtain performance statistics like accuracy,
    ### f1-score, precision and recall - in a real case scenario of imbalanced data 
    ### the default class will probably be the minority class, so the most appropriate 
    ### metric to monitor would be the f1-score - additionally we can monitor precision
    ### and recall metrics in order to account for false positives and false negatives
    ### according to the business context - that is the reason we can adjust the 
    ### cutoff value in this method, so that we can intervene according to task's requirements    
    def model_performance(self, n_epochs = 8, cutoff_value = 0.5):
          
             
          
          accuracy = list()
          f1 = list()
          recall = list()
          precision = list()

          ### we want more robust statistics, so we run for 20 different seeds
          for i in range(20):

              model2 = Sequential()
              model2.add(Dense(2*self.X_train.shape[1],activation='relu',input_shape=(self.X_train.shape[1],)))
              model2.add(Dense(int(0.65*self.X_train.shape[1]),activation='relu'))
              model2.add(Dropout(0.25))
              model2.add(Dense(int(0.33*self.X_train.shape[1]),activation = 'relu'))
              model2.add(Dense(1,activation='sigmoid'))

                                                         ### adam is the usual 'go to' optimizer
              model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
              history = model2.fit(self.X_train, self.y_train, epochs = n_epochs, validation_split = 0.2)      ### train for n epochs

              
              predictions = model2.predict(self.X_test)
              
              predictions = np.where(predictions > cutoff_value, 1, predictions)     ### the labels are assigned according 
              predictions = np.where(predictions <= cutoff_value, 0, predictions)    ### to the cutoff value

              matrix = confusion_matrix(self.y_test,predictions)      ### the confusion matrix
              acc = (matrix[0,0]+matrix[1,1])/np.sum(matrix)          ### algorithm's accuracy

           
              f1.append(f1_score(self.y_test,predictions))
              accuracy.append(acc)
              recall.append(recall_score(self.y_test,predictions))
              precision.append(precision_score(self.y_test,predictions))
             
  
          print('ANN Mean test accuracy score is', round((mean(accuracy)),2),'+/-',round((np.std(accuracy)),2))
          print('ANN Mean test F1 score score is', round((mean(f1)),2),'+/-',round((np.std(f1)),2))
          print('ANN Mean test recall score is', round((mean(recall)),2),'+/-',round((np.std(recall)),2))
          print('ANN Mean test precision score is', round((mean(precision)),2),'+/-',round((np.std(precision)),2))
    

    
    
    
    
    ### with this method we can obtain the final classifier, previous methods
    ### did not return a model - we need to specify number of training epochs and training set
    ### normally this method should train the model on the entire dataset (43993 x 129) ,
    ### but since later we want to produce the comparison precision-recall curves
    ### for the ANN and SVM classifier we still need a test set.     
    def get_final_model(self, n_epochs = 8, X_train = X_train, y_train = y_train):
        
        
        
        self.model = Sequential()
        self.model.add(Dense(2*X_train.shape[1],activation='relu',input_shape=(X_train.shape[1],)))
        self.model.add(Dense(int(0.65*X_train.shape[1]),activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(int(0.33*X_train.shape[1]),activation = 'relu'))
        self.model.add(Dense(1,activation='sigmoid'))   
        
        
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = self.model.fit(X_train, y_train, epochs = n_epochs, validation_split = 0.2)
        
        

        return self.model





    ### this method performs risk profiling for potential loans(clients)
    ### we should input a test set and a cutoff value and then we can obtain a dictionary 
    ### with the predicted class(0-1), predicted profile (Low Risk, High Risk etc.)
    ### as well as the separate ouput probabilities of the model
    def make_predictions(self, X_test, cutoff_value):
        
        risk_profile = {}
        
        predictions = self.model.predict(X_test)
        probab = predictions
        
       
              
        for i in range(len(predictions)):
            
            if ((probab[i] > cutoff_value) and (probab[i] <= cutoff_value + (1-cutoff_value))):
                risk_profile[i] = 1,'Relatively High Risk'
            elif ((probab[i] > cutoff_value + (1-cutoff_value)) and (probab[i] <= 1)):
                risk_profile[i] = 1,'High Risk'
            elif ((probab[i] <= cutoff_value) and (probab[i] > cutoff_value/2)):
                risk_profile[i] = 0, 'Relatively Low Risk'
            else:
                 risk_profile[i] = 0, 'Low Risk'
                
                
        
        return risk_profile, probab


##############################################################################



 
my_model = ANN(X_train, X_test, y_train, y_test)    ### create and train an initial ANN model

my_model.examine_training_epochs()                  ### evaluate the impact of training epochs on the model

my_model.model_performance(8,0.5)                   ### monitor performance metrics - we need to imput number of epochs
                                                    ### and the desired model's cutoff value

### for 8 training epochs and a cutoff value 0.5 the ANN classifier provided us
### with 75% accuracy, 72% F1-score, 71% recall score and 73% precision score       
### and all these without any hyperparameter - tuning  .....  < IMPRESSIVE >


        
final = my_model.get_final_model(10)        ### train and obtain the final model - need to specify epochs and training set
                                            ### we trained the final model for 10 epochs according to the previous diagram
                                            ### we could also explore the impact for more than 10 epochs....

risk_profile, probabs = my_model.make_predictions(X_test, 0.5)    ### with this method we can obtain a risk
#print(risk_profile)                                              ### profiling dictionary and the output probabilities
                                                                  ### of the model
    








##############  create a Support Vector Machine Classifier  ###################


### we will also create a Support Vector Machine to compare it to the previous
### neural network model - SVM was chosen because of its capacity to perform
### more than adequately when we have a large number of features


### import a linear Support vector machine from scikit-learn
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV



class SVM():
    
    ### the class requires as inputs the training and test sets
    def __init__(self,X_train, X_test, y_train, y_test):
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test


    ### again, this method helps us monitor vital performance metrics
    def model_performance(self):
        
          accuracy = list()
          f1 = list()
          recall = list()
          precision = list()        

          for i in range(20):              ### 20 runs for robust statistics
              
              
              ### the class_weight argument accounts for data imbalance - in case
              ### we had imbalanced data the choice 'balanced' would transform the
              ### class weights, making the model more flexible with regards to the
              ### minority class - actually this means that it helps the model to 
              ### better predict the minority class  / now that we have balanced data
              ### it has minor influence on the predictive behaviour
              svc = LinearSVC(class_weight='balanced',random_state = i)
              svc.fit(self.X_train, self.y_train)
              predictions = svc.predict(self.X_test)
              
              accuracy.append(svc.score(X_test,y_test)) 
              f1.append(f1_score(y_test,predictions))            
              recall.append(recall_score(y_test,predictions)) 
              precision.append(precision_score(y_test,predictions))               


          print('SVM Mean test accuracy score is', round((mean(accuracy)),2),'+/-',round((np.std(accuracy)),2))
          print('SVM Mean test F1 score score is', round((mean(f1)),2),'+/-',round((np.std(f1)),2))
          print('SVM Mean test recall score is', round((mean(recall)),2),'+/-',round((np.std(recall)),2))
          print('SVM Mean test precision score is', round((mean(precision)),2),'+/-',round((np.std(precision)),2))
 
    
    
    
    
    ### this class gives us the final model - normally it should be trained on
    ### the entire dataset in order to be ready for production, however for our
    ### purpose we train only on the training set     
    def get_final_model(self, X_train = X_train, y_train = y_train):  
        
        svm = LinearSVC(class_weight='balanced',random_state = 33)
        self.svc = CalibratedClassifierCV(svm)                     ### normally the LinearSVC does not provide
        self.svc.fit(X_train, y_train)                             ### output probabilities - we had to use the
                                                                   ### CalibratedClassifierCV wrapper to obtain them
        return self.svc
        
   

    ### this method returns a risk_profiling dictionary and model's output probabilities
    def make_predictions(self, X_test):
        
        risk_profile = {}
        
        predictions = self.svc.predict_proba(X_test)
        probab = predictions[:,1]                      ### get the output probabilities
        

        
        for i in range(len(predictions)):
            
            if ((probab[i] > 0.5) and (probab[i] <= 0.75 )):          ### here we use the conventional
                risk_profile[i] = 1,'Relatively High Risk'            ### 0.5 cutoff value
            elif ((probab[i] > 0.75)  and (probab[i] <= 1)):
                risk_profile[i] = 1,'High Risk'
            elif ((probab[i] <= 0.5) and (probab[i] > 0.25)):
                risk_profile[i] = 0, 'Relatively Low Risk'
            else:
                 risk_profile[i] = 0, 'Low Risk'
                
                
        
        return risk_profile, probab        


     

###############################################################################



svc = SVM(X_train, X_test, y_train, y_test)    ### initialize our SVM class

svc.model_performance()                        ### the SVM model gave us 73% accuracy,70% f1-score,
                                               ### 69% recall and 71% precision score

svc.get_final_model()                          ### create the final SVM model

risk_profile_2, probabs_2 = svc.make_predictions(X_test)   ### obtain risk_profiling dictionary and 
#print(risk_profile_2)                                      ### output probabilities

        




### next up we are going to create a precision-recall curve for both the ANN and
### SVM classifiers - by using this curve stakeholders of the problem can examine
### possible precision-recall tradeoffs and then possibly adjust the cutoff value
### of the classifiers - when adjusting the cutoff value we can achieve different
### ratios of false positives and false negatives / alternatively we could utilize  
### the ROC curve but in a real life scenario we will probably have highly imbalanced
### data and therefore one should go with the precision-recall curve

from sklearn.metrics import precision_recall_curve, auc


def plot_precision_recall_curve(y_test, probabs, probabs_2):

      ann_precision, ann_recall, _ = precision_recall_curve(np.array(y_test), probabs)
      ann_auc = auc(ann_recall, ann_precision)

      svm_precision,svm_recall, _ = precision_recall_curve(np.array(y_test), probabs_2)
      svm_auc = auc(svm_recall, svm_precision)

      plt.plot(ann_recall, ann_precision,  label='ANN',color='darkblue')
      plt.plot(svm_recall,svm_precision, label='SVM',color='orange')


      plt.xlabel('Recall')
      plt.ylabel('Precision')

      plt.legend()
      plt.title('Precision-Recall tradeoff')
      plt.show()

      return ann_auc, svm_auc


ann_auc, svm_auc = plot_precision_recall_curve(y_test, probabs, probabs_2)
print(ann_auc)            ### 81.7% area under the curve for the neural network
print(svm_auc)            ### 72.4% area under the curve for the support vector machine
                          ### indicates dominance of the ANN
                          ### this result might not always be reproduced exactly in the 
                          ### same accuracy because of the stochastic nature of 
                          ### neural network's training
