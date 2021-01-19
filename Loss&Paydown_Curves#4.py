



from statistics import mean
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt





### Please transfer all the provided .csv files to the same folder with the existing code
loan_defaults = pd.read_csv('Loan_Defaults.csv')
loan_attributes = pd.read_csv('Modified_Loan_Attributes.csv')



###########################    data preprocessing   ###########################



###create a column that indicates that the loan has defaulted
loan_defaults['DEFAULT'] = 'Default'


### join the Loan_Defaults and Modified_Loan_Attributes datasets with a left outter join
data = loan_attributes.merge(loan_defaults, how = 'left', on=['UID','RECORDNUMBER'])


### fill the column that indicates whether the loan is defaulted
data['DEFAULT'] = data['DEFAULT'].apply(lambda x:'No Default' if x!='Default' else x)



### we have full information (starting date, Openbalance, Repayperiod) for 
### 12642 defaulted loans and 23993 healthy loans
print(len(data[data.DEFAULT == 'Default']))
print(len(data[data.DEFAULT == 'No Default']))


data['LAST_MONTH'] = pd.to_datetime(data['LAST_MONTH'])   
data['ACCSTARTDATE'] = pd.to_datetime(data['ACCSTARTDATE'])   



### replace missing values in the DVAL column with the 75% of the opening balance
### this is reasonable according to the histogramm of default values as a percentage
### of initial balance
### additionally replace missing DMON values with the difference between the 
### loan starting date and date of last log-in in the web service
for i in range(data.shape[0]):
    if np.isnan(data.loc[i,'DVAL']) and data.loc[i,'DEFAULT'] == 'Default' :
        data.loc[i,'DVAL'] = 0.75*(data.loc[i,'OPENBALANCE'])
        data.loc[i,'DMON'] = 12*(data.loc[i,'LAST_MONTH'].year - data.loc[i,'ACCSTARTDATE'].year) + abs(data.loc[i,'LAST_MONTH'].month - data.loc[i,'ACCSTARTDATE'].month)
    else:
        pass
   


### create a year column indicating the starting year of the loan
data['LOAN_START_YEAR'] = 0
for i in range(data.shape[0]):
  data.loc[i,'LOAN_START_YEAR'] = data.loc[i,'ACCSTARTDATE'].year



### create a default year column indicating the year of default 
### to create this year we exploit the DMON column and add it to the ACCSTARTDATE column
### of course healthy loans are going to have a default year erroneously but
### we can filter this afterwards  
data['DEFAULT_YEAR'] = 0
for i in range(data.shape[0]):
  
  if data.loc[i,'DEFAULT'] == 'Default':
     data.loc[i,'DEFAULT_YEAR'] = data.loc[i,'ACCSTARTDATE'].year + int((data.loc[i,'ACCSTARTDATE'].month + data.loc[i,'DMON'])/12)
  else:
     data.loc[i,'DEFAULT_YEAR'] = np.nan
  





#############################################################################

### we are going to create a nine-year-horizon loss curve starting from 2011 to
### 2019 - that is because from 1993 to 2010 we have data for only 926 loans
### thus the derived loss curve will not be representative of todays business environment
### as well as of Greek economy  /
     
### additionally the last method of this class('get_balance_paydown_curve') gives us the 
### annual balances, repayments and losses, but this time for the time window 1993-2019
### in order to build the balance paydown curve     



class Plot_Curves():
    
    
    ### this class requires as input the preprocessed dataset
    def __init__(self,data):
        
        self.data = data

    
    
    ### this method returns the sum of (expected) loan repayments for every year
    def annual_repayments(self):
        
       yearly_repayments = {1993:0,1994:0,1995:0,1996:0,1997:0,1998:0,1999:0,2000:0,2001:0,
                            2002:0,2003:0,2004:0,2005:0,2006:0,2007:0,2008:0,2009:0,2010:0,
                            2011:0,2012:0,2013:0,2014:0,2015:0,2016:0,2017:0,2018:0,2019:0,
                            2020:0,2021:0,2022:0,2023:0,2024:0,2025:0,2026:0}
        
                 
       for i in range(self.data.shape[0]):                         ### for every loan
           
           start_month = self.data.loc[i,'ACCSTARTDATE'].month
           start_year  = self.data.loc[i,'ACCSTARTDATE'].year
           installment = self.data.loc[i,'OPENBALANCE']/ data.loc[i,'REPAYPERIOD']   ### the monthly installment amount

           if self.data.loc[i,'DEFAULT'] == 'Default':
              months = self.data.loc[i,'DMON']               ### total loan repayment months
           else:                                             ### if the loan is defaulted we assume DMON number of monthly installments 
              months = self.data.loc[i,'REPAYPERIOD']        ### otherwise we assume REPAYPERIOd number of monthly installments

           
           s = 0                        ### some helpful counters
           initial_months = months
           while months > 0:
               
               x = 12 - start_month
               s += x
               months -= x
               
               yearly_repayments[start_year] += x * installment
               
               if months >= 12:
                   start_month = 0
               else:
                   start_month = 12 - (initial_months - s)
                   
               start_year += 1
             
               
       return yearly_repayments      
   
    
    
    
    ### this method returns the annual total balance of portfolio for years 2011-2019
    def annual_balance(self):
        
        
        self.balance = np.array([])
        
        
        ### make the assumption that in 2011 the total balance equals to the sum of opening
        ### balances of loans given in 2011 - we also ignore the repayments of principle
        ### capital made in 2011 since we chose to ignore loans before 2011
        self.balance = np.append(self.balance, self.data[self.data.LOAN_START_YEAR == 2011]['OPENBALANCE'].sum())
        repayments = self.annual_repayments()
        
        for i in range(1993,2011):        ### delete unnecessary information
           repayments.pop(i)
        for i in range(2020,2027):
           repayments.pop(i)
           
        
        for counter, year in enumerate(repayments):
            
             if counter == 0 :           ### in order to omit 2011 that is already incorporated
                 pass                    ### in the balance 
             else:                       ### each year's balance is calculated as Balance[t]= Balance[t-1] + New_Loans[t] - Repayments[t]
                 year_balance = self.balance[counter-1] + self.data[self.data.LOAN_START_YEAR == year]['OPENBALANCE'].sum() - repayments[year]
                 self.balance = np.append(self.balance,year_balance)
            

        return self.balance
    
    
    
    
    
    ### with this method we can plot the loss curve(2011-2019) and obtain the annual losses
    ### the year of each default was defined from the feature DMON earlier in the code
    def plot_loss_curve(self):
        
         ### from 2011 to 2019 --> 9 years
         years = ['Year 1','Year 2','Year 3','Year 4','Year 5','Year 6','Year 7','Year 8','Year 9']

         ### losses per  year
         losses = data.groupby('DEFAULT_YEAR')['DVAL'].sum()
         
         losses /= 2.5        ### it is stated in the case study that the sample was stratified
                              ### to transform the percentage of default loans from 20% to 50%
                              ### therefore we have to adjust the weight of losses so that we can
                              ### have an indicative loss curve
         
         self.losses = losses[(losses.index >= 2011) & (losses.index <= 2019)]


         ratio = list(round((self.losses/self.balance *100),1))
           
         
                 
         plt.figure(figsize = (10,5))
         plt.plot(years,ratio,color='red')
         plt.grid(axis='y')
         plt.xlabel('Year')
         plt.ylabel('Loss Rate (%)')
         plt.title('Nine-year-horizon loss curve')        
         plt.show()
         
         return self.losses
     
        
        
        
    ### this method return annual balances,repayments,losses for years 1993-2019
    def get_balance_paydown_curve(self):
         
          balance = np.array([])
        
          balance = np.append(balance, self.data[self.data.LOAN_START_YEAR == 1993]['OPENBALANCE'].sum())
          repayments = self.annual_repayments()
          
          for i in range(2020,2027):         ### keep data only for period 1993-2019
              repayments.pop(i)
          
          
          for counter, year in enumerate(repayments):
            
             if counter == 0 :           ### in order to omit 1993 that is already incorporated
                 pass                    ### in the balance array
             
             else:                       ### each year's balance is calculated as Balance[t]= Balance[t-1] + New_Loans[t] - Repayments[t]
                 year_balance = balance[counter-1] + self.data[self.data.LOAN_START_YEAR == year]['OPENBALANCE'].sum() - repayments[year]
                 balance = np.append(balance,year_balance)
            
          
            
          balance = balance[:28]                  ### keep data from 1993 to 2019
          
          losses = data.groupby('DEFAULT_YEAR')['DVAL'].sum()
          
          losses = losses / 2.5                   ### adjust for stratified sample
          losses = list(losses)
          
          for i in range(10):                               ### keep data from 1993 to 2019
              losses.insert(0,0)                            ### add 0s for years that we don't have
                                                            ### available losses 1993 - 2002
            
          return balance, repayments, losses[:-1]      
  
          




###############################################################################






curve = Plot_Curves(data)                     ### initialize the class 

annual_repay = curve.annual_repayments()      ### returns annual loan repayments

balances = curve.annual_balance()             ### returns annual portfolio balances

losses = curve.plot_loss_curve()              ### returns diagram and annual losses

x, y, z = curve.get_balance_paydown_curve()    ### we get the annual balances
                                               ### repayments and losses from 1993 to 2019


w = []
for item in list(y.keys()):
    w.append(y[item])

x = list(x)    
    
paydown_curve = pd.DataFrame({'balance':x, 'repayment':w, 'losses':z})

### tranfer to excel in order to make a suitable Paydown Curve
#paydown_curve.to_csv('Draw_Paydown_Curve.csv',encoding='utf-8',index=False)
