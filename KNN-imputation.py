from sklearn.impute import KNNImputer
import numpy as np
import pandas as pd


imputer = KNNImputer(n_neighbors=6)
path  = 'D:/pycharm2022/landslide-dam/缺失率/30%/NRMSE/'
#path = 'D:/pycharm2022/landslide-dam/GAIN-master/7因子'
miss_data_x = pd.read_csv(path + '/miss_data_x.csv')
#del miss_data_x['Y1']
print(miss_data_x.head())

df_fill = imputer.fit_transform(miss_data_x)
it = pd.DataFrame(df_fill)
it.to_csv(path+'/imputed_data_KNN6_noY.csv',index=False)
