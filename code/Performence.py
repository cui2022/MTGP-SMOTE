import os

import numpy as np
import pandas as pd
from imblearn.metrics import geometric_mean_score
from sklearn import svm, preprocessing

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import recall_score, multilabel_confusion_matrix, confusion_matrix, f1_score
from sklearn.tree import DecisionTreeClassifier

from read_initial_data import delim_map
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import make_scorer
from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

pd.set_option('display.max_rows', None)


def Gmean(target, pred):
    target[target == 'positive'] = float(1)
    target[target == 'negative'] = float(0)
    pred[pred == 'positive'] = float(1)
    pred[pred == 'negative'] = float(0)
    pred = list(pred)
    target = list(target)
    AUC = roc_auc_score(target, pred)
    Gmean = geometric_mean_score(target, pred)
    return Gmean


def LR_save(data, file):
    file_name = '.\\result\\LR\\' + "LR_" + file + ".xlsx"
    # + time.strftime(' %Y.%m.%d %H:%M:%S ',time.localtime(time.time())).replace(":", "-")
    writer = pd.ExcelWriter(file_name)  # 写入Excel文件
    data.to_excel(writer, 'result')  # ‘less’是写入excel的sheet名
    writer.save()


def SVM_save(data, file):
    file_name = '.\\result\\SVM\\' + "SVM_" + file + ".xlsx"
    # + time.strftime(' %Y.%m.%d %H:%M:%S ',time.localtime(time.time())).replace(":", "-")
    writer = pd.ExcelWriter(file_name)  # 写入Excel文件
    data.to_excel(writer, 'result')  # ‘less’是写入excel的sheet名
    writer.save()


def DT_save(data, file):
    file_name = '.\\result\\DT\\' + "DT_" + file + ".xlsx"
    # + time.strftime(' %Y.%m.%d %H:%M:%S ',time.localtime(time.time())).replace(":", "-")
    writer = pd.ExcelWriter(file_name)  # 写入Excel文件
    data.to_excel(writer, 'result')  # ‘less’是写入excel的sheet名
    writer.save()


def RF_save(data, file):
    file_name = '.\\result\\RF\\' + "RF_" + file + ".xlsx"
    # + time.strftime(' %Y.%m.%d %H:%M:%S ',time.localtime(time.time())).replace(":", "-")
    writer = pd.ExcelWriter(file_name)  # 写入Excel文件
    data.to_excel(writer, 'result')  # ‘less’是写入excel的sheet名
    writer.save()


def ADA_save(data, file):
    file_name = '.\\result\\ADA\\' + "ADA_" + file + ".xlsx"
    # + time.strftime(' %Y.%m.%d %H:%M:%S ',time.localtime(time.time())).replace(":", "-")
    writer = pd.ExcelWriter(file_name)  # 写入Excel文件
    data.to_excel(writer, 'result')  # ‘less’是写入excel的sheet名
    writer.save()


def GBDT_save(data, file):
    file_name = '.\\result\\GBDT\\' + "GBDT_" + file + ".xlsx"
    # + time.strftime(' %Y.%m.%d %H:%M:%S ',time.localtime(time.time())).replace(":", "-")
    writer = pd.ExcelWriter(file_name)  # 写入Excel文件
    data.to_excel(writer, 'result')  # ‘less’是写入excel的sheet名
    writer.save()


def KNN_save(data, file):
    file_name = '.\\result\\KNN\\' + "KNN_" + file + ".xlsx"
    # + time.strftime(' %Y.%m.%d %H:%M:%S ',time.localtime(time.time())).replace(":", "-")
    writer = pd.ExcelWriter(file_name)  # 写入Excel文件
    data.to_excel(writer, 'result')  # ‘less’是写入excel的sheet名
    writer.save()


def GNB_save(data, file):
    file_name = '.\\result\\GNB\\' + "GNB_" + file + ".xlsx"
    # + time.strftime(' %Y.%m.%d %H:%M:%S ',time.localtime(time.time())).replace(":", "-")
    writer = pd.ExcelWriter(file_name)  # 写入Excel文件
    data.to_excel(writer, 'result')  # ‘less’是写入excel的sheet名
    writer.save()


def MLP_save(data, file):
    file_name = '.\\result\\MLP\\' + "MLP_" + file + ".xlsx"
    # + time.strftime(' %Y.%m.%d %H:%M:%S ',time.localtime(time.time())).replace(":", "-")
    writer = pd.ExcelWriter(file_name)  # 写入Excel文件
    data.to_excel(writer, 'result')  # ‘less’是写入excel的sheet名
    writer.save()


def file_name(path):
    path_list = os.listdir(path)
    path_list.sort(key=lambda x: int((x.split(' ')[-1]).split('.')[0]))
    print(path_list)
    return path_list


def excel_deal(filename):
    Train = pd.read_excel(filename, header=None)
    return Train


def dat_deal(filename):
    with open(filename) as f:
        first_line = f.readline()
        config = first_line.strip().split(",")

    classPos = config[0]
    num_feat = int(config[1])

    feat_labels = ['f' + str(x) for x in range(num_feat)]
    if classPos == "classFirst":
        feat_labels.insert(0, "class")
    elif classPos == "classLast":
        feat_labels.append("class")
    else:
        raise ValueError(classPos)

    delim = delim_map(config[3])

    rawData = pd.read_csv(filename, delimiter=delim, skiprows=1, header=None, names=feat_labels)
    labels = rawData['class']
    data = rawData.drop('class', axis=1)
    data = data.to_numpy()
    return data, labels


def lr_cross_validation(train_x, train_y):
    parameters = {
        'C': np.arange(0.02, 0.1, 0.02),
        'max_iter': range(10, 100, 10)
    }
    lr = LogisticRegression(penalty='l2', solver='liblinear')
    score = make_scorer(Gmean, greater_is_better=True)
    grid = GridSearchCV(lr, parameters, cv=3, scoring=score)
    grid.fit(train_x, train_y)

    clf = grid
    best_parameters = grid.best_estimator_.get_params()
    # clf.fit(train_x, train_y)
    return clf, best_parameters


def svm_cross_validation(train_x, train_y):
    parameters = {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100],
                  'gamma': [0.001, 0.01, 0.1, 1, 10]}
    svc = svm.SVC()
    # score = make_scorer(roc_auc, greater_is_better=True)
    grid_search = GridSearchCV(svc, parameters, cv=3, n_jobs=8, verbose=0, scoring="roc_auc")  #
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    model = grid_search  # svm.SVC(kernel=best_parameters['kernel'], C=best_parameters['C'], gamma=best_parameters[
    # 'gamma'])
    # model.fit(train_x, train_y)
    """
    model = svm.SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=8, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    model = svm.SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    """
    return model, best_parameters


# "Normal", "Tumor" "positive", "negative" "CMD", "DMD" "CO", "NO" "AML", "ALL"

def cau_index(target, pred, less, more):
    target[target == less] = float(1)
    target[target == more] = float(0)
    pred[pred == less] = float(1)
    pred[pred == more] = float(0)
    pred = list(pred)
    pred = list(map(float, pred))
    target = list(target)
    AUC = roc_auc_score(target, pred)
    G_Mean = geometric_mean_score(target, pred)
    F1 = f1_score(target, pred)
    return G_Mean, AUC, F1


def LR_Performance(Train, Test, less, more):  # X training samples   Y training target

    # 训练
    x_train = Train.iloc[:, :-1]
    y_train = Train.iloc[:, -1]
    x_test = Test.iloc[:, :-1]
    y_test = Test['class']
    min_max_scaler = preprocessing.MinMaxScaler()
    x_train = min_max_scaler.fit_transform(x_train)
    x_test = min_max_scaler.fit_transform(x_test)
    # scaler = preprocessing.StandardScaler()
    # x_train = scaler.fit_transform(x_train)
    # x_test = scaler.fit_transform(x_test)
    # X = preprocessing.scale(X)
    clf, parameters = lr_cross_validation(x_train, y_train)
    pred = clf.predict(x_test)
    target = y_test.to_numpy()

    # 普通数据集
    # 计算G_Mean、AUC、F1-score
    G_Mean, AUC, F1 = cau_index(target, pred, less, more)
    # print(target)
    # print(pred)
    return G_Mean, AUC, parameters['C'], parameters['max_iter'], F1


def DT_Performance(Train, Test, less, more):  # X training samples   Y training target

    clf = DecisionTreeClassifier(random_state=20)  # classfication

    x_train = Train.iloc[:, :-1]
    y_train = Train.iloc[:, -1]
    x_test = Test.iloc[:, :-1]
    y_test = Test['class']
    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()
    x_test = x_test.to_numpy()

    param = {'criterion': ['gini'], 'max_depth': range(3, 14, 2), 'min_samples_leaf': [2, 3, 5, 10],
             'min_impurity_decrease': [0.1, 0.2, 0.5]}
    score = make_scorer(Gmean, greater_is_better=True)
    grid = GridSearchCV(clf, param_grid=param, cv=5, scoring=score)
    grid.fit(x_train, y_train)
    clf = grid.best_estimator_
    best_parameters = grid.best_estimator_.get_params()
    clf.fit(x_train, y_train)  # training the  model
    pred = clf.predict(x_test)  # predict the target of testing samples
    target = y_test.to_numpy()
    # 普通数据集
    # 计算G_Mean、AUC、F1-score
    G_Mean, AUC, F1 = cau_index(target, pred, less, more)
    return G_Mean, AUC, best_parameters['max_depth'], best_parameters['min_samples_leaf'], best_parameters[
        'min_impurity_decrease'], F1


def SVM_Performance(Train, Test, less, more):  # X training samples   Y training target

    # x为数据集的feature，y为label.
    # 归一化
    x_train = Train.iloc[:, :-1]
    y_train = Train.iloc[:, -1]
    x_test = Test.iloc[:, :-1]
    y_test = Test['class']
    min_max_scaler = preprocessing.MinMaxScaler()
    x_train = min_max_scaler.fit_transform(x_train)
    x_test = min_max_scaler.fit_transform(x_test)

    # rbf 高斯（默认） linear 线性
    parameters = []
    clf, parameters = svm_cross_validation(x_train, y_train)  # classfication

    # pred = clf.predict(x_train)  # predict the target of testing samples
    # target = y_train.to_numpy()

    # 普通数据集
    # 计算G_Mean、AUC、F1-score

    pred = clf.predict(x_test)  # predict the target of testing samples
    target = y_test.to_numpy()

    # 普通数据集
    # 计算G_Mean、AUC、F1-score
    G_Mean, AUC, F1 = cau_index(target, pred, less, more)
    return G_Mean, AUC, parameters['kernel'], parameters['C'], parameters['gamma'], F1


def Knn_Performance(Train, Test, less, more):
    x_train = Train.iloc[:, :-1]
    y_train = Train.iloc[:, -1]
    x_test = Test.iloc[:, :-1]
    y_test = Test['class']
    # 归一化
    min_max_scaler = preprocessing.MinMaxScaler()
    x_train = min_max_scaler.fit_transform(x_train)
    x_test = min_max_scaler.fit_transform(x_test)
    k = 5
    # 创建k近邻实例
    neigh = KNeighborsClassifier(n_neighbors=k)
    # k近邻模型拟合
    neigh.fit(x_train, y_train)
    # k近邻模型预测
    pred = neigh.predict(x_test)
    # # 预测结果数组重塑
    # y_pred = y_pred.reshape((-1, 1))
    target = y_test.to_numpy()
    G_Mean, AUC, F1 = cau_index(target, pred, less, more)
    return G_Mean, AUC, k, F1


def GNB_Performance(Train, Test, less, more):
    x_train = Train.iloc[:, :-1]
    y_train = Train.iloc[:, -1]
    x_test = Test.iloc[:, :-1]
    y_test = Test['class']
    """
    # 归一化
    min_max_scaler = preprocessing.MinMaxScaler()
    x_train = min_max_scaler.fit_transform(x_train)
    x_test = min_max_scaler.fit_transform(x_test)
    """
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    # k近邻模型预测
    pred = gnb.predict(x_test)
    # # 预测结果数组重塑
    # y_pred = y_pred.reshape((-1, 1))
    target = y_test.to_numpy()
    G_Mean, AUC, F1 = cau_index(target, pred, less, more)
    return G_Mean, AUC, F1


def MLP_Performance(Train, Test, less, more):
    x_train = Train.iloc[:, :-1]
    y_train = Train.iloc[:, -1]
    x_test = Test.iloc[:, :-1]
    y_test = Test['class']
    # 归一化
    min_max_scaler = preprocessing.MinMaxScaler()
    x_train = min_max_scaler.fit_transform(x_train)
    x_test = min_max_scaler.fit_transform(x_test)
    mlp = MLPClassifier()
    mlp.fit(x_train, y_train)
    # k近邻模型预测
    pred = mlp.predict(x_test)
    # # 预测结果数组重塑
    # y_pred = y_pred.reshape((-1, 1))
    target = y_test.to_numpy()
    G_Mean, AUC, F1 = cau_index(target, pred, less, more)
    return G_Mean, AUC, F1


def RF_Performance(Train, Test, less, more):  # X training samples   Y training target

    clf = RandomForestClassifier(random_state=20)  # classfication

    x_train = Train.iloc[:, :-1]
    y_train = Train.iloc[:, -1]
    x_test = Test.iloc[:, :-1]
    y_test = Test['class']

    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()
    x_test = x_test.to_numpy()

    param = {'n_estimators': range(10, 70, 10), 'max_depth': range(3, 14, 2), 'min_samples_split': range(10, 60, 10)}
    # param = {'n_estimators': [10], 'max_depth': [10], 'min_samples_split': [10]}
    score = make_scorer(Gmean, greater_is_better=True)
    grid = GridSearchCV(clf, param_grid=param, cv=3, scoring=score)
    grid.fit(x_train, y_train)
    clf = grid.best_estimator_
    best_parameters = grid.best_estimator_.get_params()

    clf.fit(x_train, y_train)  # training the  model

    pred = clf.predict(x_test)  # predict the target of testing samples
    target = y_test.to_numpy()

    # 普通数据集
    # 计算G_Mean、AUC、F1-score
    G_Mean, AUC, F1 = cau_index(target, pred, less, more)

    return G_Mean, AUC, best_parameters['n_estimators'], best_parameters['max_depth'], best_parameters[
        'min_samples_split'], F1


def ADA_Performance(Train, Test, less, more):  # X training samples   Y training target
    clf = AdaBoostClassifier(base_estimator=None, learning_rate=0.5, algorithm='SAMME.R', random_state=20)
    # clf = RandomForestClassifier(random_state=20)  # classfication

    x_train = Train.iloc[:, :-1]
    y_train = Train.iloc[:, -1]
    x_test = Test.iloc[:, :-1]
    y_test = Test['class']

    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()
    x_test = x_test.to_numpy()

    param = {'n_estimators': range(10, 70, 10)}
    score = make_scorer(Gmean, greater_is_better=True)
    grid = GridSearchCV(clf, param_grid=param, cv=3, scoring=score)
    grid.fit(x_train, y_train)
    clf = grid.best_estimator_
    best_parameters = grid.best_estimator_.get_params()

    clf.fit(x_train, y_train)  # training the  model

    pred = clf.predict(x_test)  # predict the target of testing samples
    target = y_test.to_numpy()

    # 普通数据集
    # 计算G_Mean、AUC、F1-score
    G_Mean, AUC, F1 = cau_index(target, pred, less, more)

    return G_Mean, AUC, best_parameters['n_estimators'], F1


def GBDT_Performance(Train, Test, less, more):  # X training samples   Y training target

    clf = GradientBoostingClassifier(random_state=20)

    x_train = Train.iloc[:, :-1]
    y_train = Train.iloc[:, -1]
    x_test = Test.iloc[:, :-1]
    y_test = Test['class']

    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()
    x_test = x_test.to_numpy()

    param = {'n_estimators': range(10, 70, 10), 'max_depth': range(3, 14, 2), 'min_samples_split': range(10, 60, 10)}
    # param = {'n_estimators': [10], 'max_depth': [10], 'min_samples_split': [10]}
    score = make_scorer(Gmean, greater_is_better=True)
    grid = GridSearchCV(clf, param_grid=param, cv=3, scoring=score)
    grid.fit(x_train, y_train)
    clf = grid.best_estimator_
    best_parameters = grid.best_estimator_.get_params()

    clf.fit(x_train, y_train)  # training the  model

    pred = clf.predict(x_test)  # predict the target of testing samples
    target = y_test.to_numpy()

    # 普通数据集
    # 计算G_Mean、AUC、F1-score
    G_Mean, AUC, F1 = cau_index(target, pred, less, more)

    return G_Mean, AUC, best_parameters['n_estimators'], best_parameters['max_depth'], best_parameters[
        'min_samples_split'], F1
