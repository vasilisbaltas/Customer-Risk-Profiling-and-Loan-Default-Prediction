# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


### Please transfer all the provided .csv files to the same folder with the existing code
loan_defaults = pd.read_csv('Loan_Defaults.csv')
loan_attributes = pd.read_csv('Modified_Loan_Attributes.csv')
risk_quality = pd.read_csv('Client_Risk_Quality.csv')



### merging the prepared loan_attributes and loan defaults datasets with an inner join
### they have 12642 loans in common that were defaulted
df = pd.merge(loan_attributes, loan_defaults, on = ['UID','RECORDNUMBER'])




### create 2 new columns for the df dataset - the year of the last log-in in the
### online service which we consider here as the date of actual default (assumption)
### as well as a percentage column for the percentage of the unpaid amount
df['LAST_MONTH'] = pd.to_datetime(df['LAST_MONTH'])
df['YEAR'] = 0
df['PERCENTAGE'] = 0
for i in range(df.shape[0]):
  df.loc[i,'YEAR'] = df.loc[i,'LAST_MONTH'].year
  df.loc[i,'PERCENTAGE'] = df.loc[i,'DVAL']/df.loc[i,'OPENBALANCE']


### create a YEAR column for the initial loan_attributes dataset - this column
### indicates the starting year of the loan  
loan_attributes['ACCSTARTDATE'] = pd.to_datetime(loan_attributes['ACCSTARTDATE'])
loan_attributes['YEAR'] = 0
for i in range(loan_attributes.shape[0]):
    loan_attributes.loc[i,'YEAR'] = loan_attributes.loc[i,'ACCSTARTDATE'].year





### checking how many loans were approved in the last decade - as well as the
### losses from defaulted loan

years = list(np.unique(df.YEAR))
losses = df.groupby('YEAR')['DVAL'].sum()
losses2 = losses/1000000           ### turn into millions - 17.241.680 euros charged off in the last decade

years2 = list(np.unique(loan_attributes.YEAR))
amounts = loan_attributes.groupby('YEAR')['OPENBALANCE'].sum()


x = years2[16:]                    ### extracting the data for the last nine years
y = amounts[16:]/1000000           

fig, ax = plt.subplots()
ax.plot(years, losses2, color='red')
ax.plot(x, y, color = 'darkblue')
ax.ticklabel_format(useOffset=False, style='plain')
plt.xlabel('Year')
plt.ylabel('Millions')
plt.title('New loans v. Defaulted loans')
plt.legend(['Defaulted loans','New loans'])
plt.show()







### exploring the percentage of unpaid loans - for what amount of the initial
### lended capital does the default value accounts for
df2 = df.dropna(axis=0)
perc = list(round((df2.PERCENTAGE*100),2))
plt.hist(perc, bins=40, range = [0,99], color = 'darkblue')
plt.xlabel('Percentage of unpaid loan (%)')
plt.ylabel('Number of loans')
plt.title('Default value as percentage of initial capital')
plt.show()


  


######################  continue data exploration     #########################



###create a column that indicates if the loan has defaulted
loan_defaults['DEFAULT'] = 'Default'


### join the Client_Risk_Quality, Loan_Defaults and Modified_Loan_Attributes datasets
data = loan_attributes.merge(loan_defaults, how = 'left', on=['UID','RECORDNUMBER'])
data = data.merge(risk_quality, on = 'UID')


### fill the column that indicates whether the loan is defaulted
data['DEFAULT'] = data['DEFAULT'].apply(lambda x:'No Default' if x!='Default' else x)


### create boxplots to explore the relationship between default and acquisition score,
### loan opening balance
data.boxplot(column = ['OPENBALANCE'], by='DEFAULT', grid = False, figsize=(10,5), showfliers = False)
data.boxplot(column = ['SCORE'], by='DEFAULT', grid = False, figsize=(10,5), showfliers = False)







### next-up we are going to create bins for the acquisition score variable and 
### repay-period variable

bins = [461,530,598,665,732]
bucket = ['High Risk','Relatively High Risk','Relatively Low Risk','Low Risk']
data['SCORE_RANGE'] = pd.cut(data['SCORE'], bins, labels=bucket)

bins2 = [11,24,36,48,61]
bucket2 = ['2 Years','3 Years','4 Years','5 Years']
data['REPAY_RANGE'] = pd.cut(data['REPAYPERIOD'], bins2, labels=bucket2)




### calculate the relative percentages of defaults for the 4 different 
### acquisition_score ranges - divide with the total number of default cases
### meaning 20.000 , that is the reason the percentages don't reach up to 100%  !!!!!
x1 = ['High Risk','Relatively High Risk','Relatively Low Risk','Low Risk']
defaulter_ratios = [(len(data[(data.SCORE_RANGE == interval) & (data.DEFAULT == 'Default')])/loan_defaults.shape[0])
                      for interval in x1]


### the same for different ranges of repay_period
x2 = ['2 Years','3 Years','4 Years','5 Years']
defaulter_ratios2 = [(len(data[(data.REPAY_RANGE == interval) & (data.DEFAULT == 'Default')])/loan_defaults.shape[0])
                      for interval in x2]


### barplot showing the relative ratios of defaulters with respect to the acquisitions score
plt.figure(figsize = (12,5))
plt.bar(x1,defaulter_ratios, color = 'darkblue')
plt.xticks(rotation =45)
plt.ylabel('Defaulters\'s ratio')
plt.xlabel('Acquisition score')
plt.title('Defaulters\'s percentage depending on acquisition score range')
plt.show()


### barplot showing the relative ratios of defaulters with respect to the repayment period
plt.figure(figsize = (10,5))
plt.bar(x2,defaulter_ratios2, color = 'darkblue')
plt.xticks(rotation =45)
plt.ylabel('Defaulters\'s ratio')
plt.xlabel('Repay period')
plt.title('Defaulters\'s percentage depending on repay period')
plt.show()






### we are going to create a barplot taking into account the repayment period
### and the acquistion score simultaneously
risks = []
for item1 in x1:
    for item2 in x2:
        risks.append(len(data[(data.SCORE_RANGE == item1) & (data.DEFAULT == 'Default') & (data.REPAY_RANGE == item2)])
                     /loan_defaults.shape[0])
        
cross = pd.DataFrame({'SCORE_RANGE':4*['High Risk'] + 4*['Relatively High Risk'] + 4*['Relatively Low Risk'] + 4*['Low Risk'],
                      'REPAY_RANGE':4*['2 Years','3 Years','4 Years','5 Years'],
                      'Defaulters\' ratio': risks})



plt.figure(figsize=(10,5))
sns.barplot('SCORE_RANGE','Defaulters\' ratio', hue = 'REPAY_RANGE', palette = ['darkblue','crimson','green','darkorange'], data=cross)
plt.title('Defaulters\'s percentage depending on acquisition score range and repayment range')
plt.show()


