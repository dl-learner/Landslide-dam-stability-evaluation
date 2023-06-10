import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#from deepforest import CascadeForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import xgboost
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from xgboost import plot_importance
from matplotlib import pyplot

#四个数据集：完整数据集，GAIN填充，missforest填充，KNN填充
#path = './GAIN-master/7因子'
#path2 = 'D:/pycharm2022/landslide-dam/created data based on 6'
path  = 'D:/pycharm2022/landslide-dam/缺失率/5%/'
ori_data_x = pd.read_csv(path + '/ori_data_x.csv')
#imputed_gain = pd.read_csv(path + '/imputed_data1.csv')
'''imputed_forest = pd.read_csv(path + '/imputed_data_missforest_noY1.csv')
imputed_knn = pd.read_csv(path + '/imputed_data_KNN3_noY1.csv')

imputed_RF = pd.read_csv(path + '/imputed_data_MICE_noY1.csv')
miss_data = pd.read_csv(path + '/mean_data_x.csv')'''
#miss_data1 = pd.read_csv(path + '/miss_data_x.csv')

def df_train1(x,a):

    x1 = x.copy()
    del x1['Y1']
    y1 = x['Y1']
    ss_x = StandardScaler()
    x1 = ss_x.fit_transform(x1)

    #clf = RandomForestClassifier(n_estimators=60, criterion='entropy', max_depth=4)
    clf = xgboost.XGBClassifier(learning_rate=0.05, objective='binary:logistic',n_estimators=80,max_depth = 6)
    #clf = CascadeForestClassifier(min_samples_leaf = 5,n_trees = 20,max_depth = 4,max_layers = 2,random_state=1255)
    x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, random_state=a)
    #score4 = cross_val_score(clf, x_train, y_train, cv=5, scoring='accuracy')
    clf.fit(x_train,y_train)

    y_pred = clf.predict_proba(x_test)
    y_pred = y_pred[:,1]
    fpr_tr1, tpr_tr1, threholds = metrics.roc_curve(y_test, y_pred)
    y_test_auc = metrics.auc(fpr_tr1, tpr_tr1)
    roc = pd.DataFrame()
    roc['fp_test'] = fpr_tr1
    roc['tp_test'] = tpr_tr1


    y_pred = clf.predict_proba(x_train)
    y_pred = y_pred[:, 1]
    fpr_tr1, tpr_tr1, threholds = metrics.roc_curve(y_train, y_pred)
    y_train_auc = metrics.auc(fpr_tr1, tpr_tr1)
    return y_train_auc,y_test_auc,roc

def model_train(x,model,para_dict,a):
    x1 = x.copy()
    del x1['Y1']
    y1 = x['Y1']
    ss_x = StandardScaler()
    x1 = ss_x.fit_transform(x1)

    x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, random_state=a)
    estimator = GridSearchCV(model, param_grid=para_dict, cv=5)
    estimator.fit(x_train, y_train)

    print("最佳参数：\n", estimator.best_params_)

    return estimator,estimator.best_params_

def df_train(x,a):
    XGBoost = xgboost.XGBClassifier(learning_rate=0.05, objective='binary:logistic')

    #para_path = r'./配置文件.txt '
    param_grid = {'n_estimators': [60,100],
                  'max_depth': [3, 5,8],
                  'learning_rate': [0.1],
                  'subsample': [0.7, 1.0],
                  'colsample_bytree': [0.7, 1.0],
                  'gamma': [0.1]}
    '''param_grid = {'n_estimators': [100],
                  'max_depth': [8],
                  'learning_rate': [0.1],
                  'subsample': [0.5],
                  'colsample_bytree': [0.8],
                  'gamma': [0.1]}'''

    clf, xgboostmodelpara = model_train(x, XGBoost, param_grid,a)

    x1 = x.copy()
    del x1['Y1']
    y1 = x['Y1']
    ss_x = StandardScaler()
    x1 = ss_x.fit_transform(x1)

    x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, random_state=a)
    clf.fit(x_train, y_train)

    y_pred = clf.predict_proba(x_test)
    y_pred = y_pred[:, 1]
    fpr_tr1, tpr_tr1, threholds = metrics.roc_curve(y_test, y_pred)
    y_test_auc = metrics.auc(fpr_tr1, tpr_tr1)
    roc = pd.DataFrame()
    roc['fp_test'] = fpr_tr1
    roc['tp_test'] = tpr_tr1

    y_pred = clf.predict_proba(x_train)
    y_pred = y_pred[:, 1]
    fpr_tr1, tpr_tr1, threholds = metrics.roc_curve(y_train, y_pred)
    y_train_auc = metrics.auc(fpr_tr1, tpr_tr1)
    y_pred1 = clf.predict_proba(x_test)
    tem = pd.DataFrame()
    tem['y_pre'] = y_pred1[:,0]
    tem['y_test'] = list(y_test)
    #y_pred1 = pd.DataFrame((y_pred1[:,0],y_test))
    return y_train_auc, y_test_auc, roc, tem

def df_train2(x,a):
    #XGBoost = xgboost.XGBClassifier(learning_rate=0.05, objective='binary:logistic')

    '''param_grid = {'n_estimators': [50, 100, 150],
                   'max_depth': [6, 10, 15],
                   'min_samples_split': [2, 3, 5],
                   'min_samples_leaf': [1, 2, 3],
                   'max_features': ['auto', 'sqrt']}'''

    et_clf = SVC(probability=True,class_weight={0:4, 1:5})

    param_grid = {'C': [0.5,0.8,1.2,1.5,1, 2,0.1],
                  'kernel': ['rbf'],
                  'gamma': ['scale', 'auto']}

    '''param_grid = {'C': [0.1],
                  'kernel': ['rbf'],
                  'gamma': ['auto']}'''

    '''param_grid = {'C': [0.4],
                  'penalty': ['l1'],
                  'solver': ['saga']}'''
    '''param_grid = {'n_estimators': [100],
                  'max_depth': [6],
                  'min_samples_split': [2],
                  'min_samples_leaf': [1],
                  'max_features': ['auto']}'''

    #et_clf = ExtraTreesClassifier(random_state=42)
    #et_clf = LogisticRegression()


    clf, xgboostmodelpara = model_train(x, et_clf, param_grid,a)

    x1 = x.copy()
    del x1['Y1']
    y1 = x['Y1']
    ss_x = StandardScaler()
    x1 = ss_x.fit_transform(x1)

    x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, random_state=a)
    clf.fit(x_train, y_train)

    y_pred = clf.predict_proba(x_test)
    y_pred = y_pred[:, 1]
    fpr_tr1, tpr_tr1, threholds = metrics.roc_curve(y_test, y_pred)
    y_test_auc = metrics.auc(fpr_tr1, tpr_tr1)
    roc = pd.DataFrame()
    roc['fp_test'] = fpr_tr1
    roc['tp_test'] = tpr_tr1

    y_pred = clf.predict_proba(x_train)
    y_pred = y_pred[:, 1]
    fpr_tr1, tpr_tr1, threholds = metrics.roc_curve(y_train, y_pred)
    y_train_auc = metrics.auc(fpr_tr1, tpr_tr1)
    y_pred1 = clf.predict_proba(x1)
    tem = pd.DataFrame()
    tem['y_pre'] = y_pred1[:, 0]
    tem['y_test'] = list(y1)
    return y_train_auc, y_test_auc, roc, tem


a, b, c, d,e,f,g = 0, 0, 0, 0,0,0,0
for i in range(11,16):

    print('************:',i)
    or_train_auc,or_test_auc,roc, y_pred1= df_train2(ori_data_x,i)
    #roc.to_csv(path+'/original_data_roc.csv')
    #y_pred1.to_csv(path+'/ori_data_pre.csv')
    print('original_data:',or_train_auc,or_test_auc)
    a += or_test_auc

    '''gain_train_auc,gain_test_auc,roc,y_pred1 = df_train(imputed_gain,i)
    #roc.to_csv(path+'/Imputed_gain_roc_train.csv')
    #y_pred1.to_csv(path + '/imputed_data_pre16.csv')
    print('imputed_gain:',gain_train_auc,gain_test_auc)
    b+= gain_test_auc'''

    '''or_train_auc,or_test_auc,roc, y_pred1 = df_train(imputed_knn,i)
    #roc.to_csv(path+'/imputed_knn_roc.csv')
    #y_pred1.to_csv(path + '/imputed_knn_pre.csv')
    print('imputed_knn:',or_train_auc,or_test_auc)
    c += or_test_auc

    or_train_auc,or_test_auc,roc, y_pred1 = df_train(imputed_forest,i)
    #roc.to_csv(path+'/imputed_missforest_roc.csv')
    #y_pred1.to_csv(path + '/imputed_missforest_pre.csv')
    print('imputed_missforest:',or_train_auc,or_test_auc)
    d += or_test_auc

    or_train_auc,or_test_auc,roc, y_pred1 = df_train(imputed_RF,i)
    #roc.to_csv(path+'/imputed_MICE_roc.csv')
    #y_pred1.to_csv(path + '/imputed_MICE_pre.csv')
    print('imputed_MICE:',or_train_auc,or_test_auc)
    e += or_test_auc

    or_train_auc,or_test_auc,roc, y_pred1 = df_train(miss_data,i)
    #roc.to_csv(path+'/mean_data_roc.csv')
    #y_pred1.to_csv(path + '/mean_pre.csv')
    print('mean_data:',or_train_auc,or_test_auc)
    f += or_test_auc'''

    '''or_train_auc, or_test_auc, roc, y_pred1 = df_train(miss_data1, i)
    #roc.to_csv(path+'/miss_data_roc.csv')
    #y_pred1.to_csv(path + '/miss_pre.csv')
    print('miss_data:', or_train_auc, or_test_auc)
    g += or_test_auc'''

print(a/5,b/5,c/5,d/5,e/5,f/5,g/5)
print(a,b,c,d,e,f,g)


x1 = ori_data_x.copy()
del x1['Y1']
y1 = ori_data_x['Y1']
clf = xgboost.XGBClassifier(learning_rate=0.1, objective='binary:logistic',n_estimators=80,max_depth = 6)
#clf = CascadeForestClassifier(min_samples_leaf = 5,n_trees = 20,max_depth = 4,max_layers = 2,random_state=1255)
x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, random_state=111)
#score4 = cross_val_score(clf, x_train, y_train, cv=5, scoring='accuracy')
clf.fit(x_train,y_train)
#plot_importance(clf)
#pyplot.show()