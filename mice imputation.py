from missingpy import MissForest
import numpy as np
import pandas as pd
#from impyute.imputation.cs import mice
#import mice
from fancyimpute import IterativeImputer
#from sklearn.impute import SimpleImputer
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.ensemble import RandomForestClassifier

#from sklearn.experimental import enable_iterative_imputer
#from sklearn.impute import IterativeImputer,SimpleImputer
#from sklearn.linear_model import BayesianRidge

path  = 'D:/pycharm2022/landslide-dam/缺失率/30%/NRMSE/'
#path = 'D:/pycharm2022/landslide-dam/GAIN-master/7因子/'
miss_data_x = pd.read_csv(path + 'miss_data_x.csv')
#del miss_data_x['Y1']
print(miss_data_x.head())
miss_data_x['0'] = miss_data_x['0'].astype(object)
#nan =float('NaN')


df = miss_data_x


# 设定MICE的参数
num_iters = 30  # 迭代次数
imputer = IterativeImputer(random_state=123)
#imputed_data = imputer.fit_transform(df)

# 拟合数据并完成插补
imputed_df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

'''br = BayesianRidge()
imp = IterativeImputer(estimator=br,random_state=0,missing_values=np.nan,
                      sample_posterior=True,initial_strategy='most_frequent',
                      imputation_order='ascending',max_iter=30)
X_miss_reg = imp.fit_transform(miss_data_x)
#imputer = MissForest(random_state=1337)
#it = imputer.fit_transform(X_missing_reg)'''
#it = pd.DataFrame(X_miss_reg)
imputed_df.to_csv(path + 'imputed_data_MICE_noY.csv',index=False)

