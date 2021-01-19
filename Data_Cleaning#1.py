# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np




################    checking the Loan_Attributes csv file   ###################



loan_attributes = pd.read_csv('Loan_Attributes.csv')

### we will replace the NaNs in ACCSTARTDATE column with the values of FIRST_MONTH
### column since they are almost identical
loan_attributes['ACCSTARTDATE'] = loan_attributes['ACCSTARTDATE'].fillna(loan_attributes[loan_attributes['ACCSTARTDATE'].isna()]['FIRST_MONTH'])



### replace NaNs of OPENBALANCE and REPAYPERIOD with the respective mean feature values
loan_attributes['OPENBALANCE'] = loan_attributes['OPENBALANCE'].fillna(int(loan_attributes['OPENBALANCE'].mean()))
loan_attributes['REPAYPERIOD'] = loan_attributes['REPAYPERIOD'].fillna(int(loan_attributes['REPAYPERIOD'].mean()))



### removing outliers with respect to the OPENBALANCE and REPAYPERIOD attributes
low_balance = loan_attributes['OPENBALANCE'].quantile(0.05)             ### subjectively choose quantiles
high_balance = loan_attributes['OPENBALANCE'].quantile(0.99)
loan_attributes = loan_attributes[(loan_attributes['OPENBALANCE'] > low_balance) & (loan_attributes['OPENBALANCE'] < high_balance)]

loan_attributes = loan_attributes[(loan_attributes['REPAYPERIOD'] >= 12) & (loan_attributes['REPAYPERIOD'] <= 60)]         ### according to the case study



### drop the FIRST_MONTH column since it's almost identical to the ACCSTARTDATE column
### and the SEARCHDATE column because its data does not make sense and does not add value
### to the analysis
final_loan_attributes = loan_attributes.drop(['FIRST_MONTH','SEARCHDATE'],axis=1)



#final_loan_attributes.to_csv('Modified_Loan_Attributes.csv',encoding='utf-8',index=False)





################    checking the Client_Features csv file   ###################



client_features = pd.read_csv('Client_Features.csv')


### deleting duplicate columns
client_features = client_features.T.drop_duplicates().T


### defining columns to remove because of high correlation or low variance
columns_to_remove = ['F_2','F_22','F_29','F_42','F_41','F_45','F_46','F_47','F_48','F_49','F_50','F_51','F_52',
                     'F_54','F_55','F_56','F_57','F_58','F_60','F_61','F_63','F_64','F_66','F_73','F_76','F_86',
                     'F_93','F_96','F_99','F_105','F_106','F_111','F_110','F_113','F_115','F_116','F_117','F_118',
                     'F_119','F_120','F_121','F_122','F_124','F_24','F_25','F_26','F_27','F_28','F_126','F_127',
                     'F_129','F_130','F_139','F_140','F_145','F_150','F_151','F_152','F_153','F_155','F_156',
                     'F_157','F_158','F_167','F_170','F_171','F_174','F_175','F_176','F_177','F_178','F_179',
                     'F_180','F_181','F_182','F_184','F_185','F_187','F_188','F_190','F_191','F_192','F_194',
                     'F_195','F_198','F_201','F_203','F_204','F_205','F_206','F_215','F_216','F_219','F_220',
                     'F_221','F_222','F_223','F_224','F_225','F_226','F_227','F_228','F_231','F_232','F_233',
                     'F_234','F_235','F_237','F_243','F_249','F_255','F_256','F_261','F_267','F_273']

client_features = client_features.drop(columns = columns_to_remove)


### replace -999997 and -999999 that don't make sense with features' median values
### replacing with mean value would still give us a negative number making no sense
for col in client_features.columns:
    client_features[col] = client_features[col].apply(lambda x: int(client_features[col].median()) if x ==-999997 or x ==-999999 else x )





#client_features.to_csv('Modified_Client_Features.csv',encoding='utf-8',index=False)
