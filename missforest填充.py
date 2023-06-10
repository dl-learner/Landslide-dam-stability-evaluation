from missingpy import MissForest
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

path  = 'D:/pycharm2022/landslide-dam/缺失率/30%/NRMSE/'
#path = 'D:/pycharm2022/landslide-dam/GAIN-master/7因子/'
miss_data_x = pd.read_csv(path + 'miss_data_x.csv')
miss_data_x['0'] = miss_data_x['0'].astype(object)
miss_data_x['-1'] = miss_data_x['0'].astype(object)
#del miss_data_x['Y1']
print(miss_data_x.head())

print(miss_data_x.dtypes)
'''nan =float('NaN')


imputer = MissForest(random_state=1337)
it = imputer.fit_transform(miss_data_x)
it = pd.DataFrame(it)'''

data = miss_data_x
# Get column names for continuous variables
cont_vars = data.select_dtypes(include=['float64']).columns.tolist()

# Use MissForest to impute continuous variables
mf_con = MissForest(random_state=123)
cont_imputed = mf_con.fit_transform(data[cont_vars])

# Assign imputed values back to original data frame
data[cont_vars] = cont_imputed

# Get column names for categorical variables
cat_vars = data.select_dtypes(include=['object']).columns.tolist()
print(cat_vars)

# Use MissForest to impute categorical variables
mf_cat = MissForest()
cat_imputed = mf_cat.fit_transform(data[cat_vars])

# Assign imputed values back to original data frame
data[cat_vars] = cat_imputed

data.to_csv(path + 'imputed_data_missforest_noY.csv',index=False)
