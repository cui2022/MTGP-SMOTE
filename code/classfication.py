import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from imblearn.metrics import geometric_mean_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm, preprocessing
import sklearn.svm as svm


def LR(x_train, x_test, y_train, y_test, conname, less_class, more_class):
    scaler = preprocessing.StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    # # # 不能使用最大最小归一化！！！！！
    # min_max_scaler = preprocessing.MinMaxScaler()
    # x_train = min_max_scaler.fit_transform(x_train)
    # x_test = min_max_scaler.fit_transform(x_test)
    #
    x_train = pd.DataFrame(x_train, columns=conname)
    x_test = pd.DataFrame(x_test, columns=conname)
    lr = LogisticRegression(penalty='l2', max_iter=1000, solver='liblinear')
    lr = lr.fit(x_train, y_train)
    lr_pred = lr.predict(x_test)

    lr_pred[lr_pred == less_class] = float(1)
    lr_pred[lr_pred == more_class] = float(0)
    lr_pred = list(lr_pred)
    lr_pred = list(map(float, lr_pred))

    target = y_test.to_numpy()
    target[target == less_class] = float(1)
    target[target == more_class] = float(0)
    target = list(target)

    lr_G_mean = geometric_mean_score(target, lr_pred)
    return lr_G_mean


def SVM(x_train, x_test, y_train, y_test, conname, less_class, more_class):

    scaler = preprocessing.StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    # # # # 不能使用最大最小归一化！！！！！
    # min_max_scaler = preprocessing.MinMaxScaler()
    # x_train = min_max_scaler.fit_transform(x_train)
    # x_test = min_max_scaler.fit_transform(x_test)
    #
    x_train = pd.DataFrame(x_train, columns=conname)
    x_test = pd.DataFrame(x_test, columns=conname)

    model_SVM = svm.SVC()
    model_SVM = model_SVM.fit(x_train, y_train)
    svm_pred = model_SVM.predict(x_test)
    svm_pred[svm_pred == less_class] = float(1)
    svm_pred[svm_pred == more_class] = float(0)
    svm_pred = list(svm_pred)
    svm_pred = list(map(float, svm_pred))
    target = y_test.to_numpy()
    target[target == less_class] = float(1)
    target[target == more_class] = float(0)
    target = list(target)

    svm_G_mean = geometric_mean_score(target, svm_pred)
    return svm_G_mean


def DT(x_train, x_test, y_train, y_test, conname, less_class, more_class):
    x_train = pd.DataFrame(x_train, columns=conname)
    DT = DecisionTreeClassifier(random_state=20)
    DT = DT.fit(x_train, y_train)
    DT_pred = DT.predict(x_test)

    target = y_test.to_numpy()
    target[target == less_class] = float(1)
    target[target == more_class] = float(0)

    DT_pred[DT_pred == less_class] = float(1)
    DT_pred[DT_pred == more_class] = float(0)
    DT_pred = list(DT_pred)
    DT_pred = list(map(float, DT_pred))
    target = list(target)
    DT_G_mean = geometric_mean_score(target, DT_pred)

    return DT_G_mean


def RF(x_train, x_test, y_train, y_test, conname, less_class, more_class):
    x_train = np.nan_to_num(x_train.astype(np.float32))
    x_train = pd.DataFrame(x_train, columns=conname)
    rf = RandomForestClassifier(random_state=20)
    rf = rf.fit(x_train, y_train)
    rf_pred = rf.predict(x_test)

    target = y_test.to_numpy()
    target[target == less_class] = float(1)
    target[target == more_class] = float(0)
    target = list(target)

    rf_pred[rf_pred == less_class] = float(1)
    rf_pred[rf_pred == more_class] = float(0)
    rf_pred = list(rf_pred)
    rf_pred = list(map(float, rf_pred))

    rf_G_mean = geometric_mean_score(target, rf_pred)

    return rf_G_mean


def GBDT(x_train, x_test, y_train, y_test, conname, less_class, more_class):
    x_train = pd.DataFrame(x_train, columns=conname)
    gbdt = GradientBoostingClassifier(random_state=20)
    gbdt = gbdt.fit(x_train, y_train)
    gbdt_pred = gbdt.predict(x_test)

    target = y_test.to_numpy()
    target[target == less_class] = float(1)
    target[target == more_class] = float(0)
    target = list(target)

    gbdt_pred[gbdt_pred == less_class] = float(1)
    gbdt_pred[gbdt_pred == more_class] = float(0)
    gbdt_pred = list(gbdt_pred)
    gbdt_pred = list(map(float, gbdt_pred))

    gbdt_G_mean = geometric_mean_score(target, gbdt_pred)
    return gbdt_G_mean


def ADA(x_train, x_test, y_train, y_test, conname, less_class, more_class):
    x_train = pd.DataFrame(x_train, columns=conname)
    ADA = AdaBoostClassifier(random_state=20)
    ADA = ADA.fit(x_train, y_train)
    ADA_pred = ADA.predict(x_test)

    target = y_test.to_numpy()
    target[target == less_class] = float(1)
    target[target == more_class] = float(0)
    target = list(target)

    ADA_pred[ADA_pred == less_class] = float(1)
    ADA_pred[ADA_pred == more_class] = float(0)
    ADA_pred = list(ADA_pred)
    ADA_pred = list(map(float, ADA_pred))

    ADA_G_mean = geometric_mean_score(target, ADA_pred)
    return ADA_G_mean
