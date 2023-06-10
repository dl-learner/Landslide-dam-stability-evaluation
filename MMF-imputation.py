import numpy as np
import pandas as pd


path  = 'D:/pycharm2022/landslide-dam/缺失率/30%/NRMSE/'

miss_data_x = pd.read_csv(path + 'miss_data_x.csv')
#del miss_data_x['Y1']

# 使用均值填充缺失值
miss_data_x['1'].fillna(miss_data_x['1'].mean(), inplace=True)
miss_data_x['2'].fillna(miss_data_x['2'].mean(), inplace=True)
miss_data_x['3'].fillna(miss_data_x['3'].mean(), inplace=True)
miss_data_x['4'].fillna(miss_data_x['4'].mean(), inplace=True)
miss_data_x['5'].fillna(miss_data_x['5'].mean(), inplace=True)
#使用most-frequency
miss_data_x['0'].fillna(1, inplace=True)

miss_data_x.to_csv(path + 'mean_data_x.csv',index=False)

