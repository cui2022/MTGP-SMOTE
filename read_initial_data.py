import math
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

def writeinto_excel(data):
    result_end = pd.DataFrame(data)
    file_name = ".\\Initial_data" + "initial less data" + time.strftime(' %Y.%m.%d %H:%M:%S ',
                                                                        time.localtime(time.time())).replace(":",
                                                                                                             "-") + ".xlsx "
    writer = pd.ExcelWriter(file_name)  # 写入Excel文件
    result_end.to_excel(writer, 'less', header=None, index=None)  # ‘less’是写入excel的sheet名
    writer.save()
    writer.close()


def delim_map(delim):
    switch = {
        "comma": ",",
        "space": " ",
        "comma-space": ", "
    }
    return switch.get(delim)


def calEuclidean(x, y):
    x = np.array(x)
    y = np.array(y)
    dist = np.linalg.norm(x - y)
    return dist


def cau_center(data):
    center = data.sum(axis=0) / data.shape[0]
    return center


def dis_order(center, data):
    dis = []
    for i in range(data.shape[0]):
        dis.append(calEuclidean(center, data[i, :]))
    data = pd.DataFrame(data)
    data["dis"] = dis
    res = data.sort_values(by='dis', ascending=False)
    return res


# 读.dat 与.arff文件
def read_data(filename, judge):
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

    num_classes = int(config[2])
    delim = delim_map(config[3])

    rawData = pd.read_csv(filename, delimiter=delim, skiprows=1, header=None, names=feat_labels)
    # print(rawData)

    # 将原始数据集分为训练集与测试集
    X = rawData.iloc[:, :-1]
    Y = rawData['class']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=22, stratify=Y)

    Train = pd.DataFrame(x_train)
    Train['class'] = y_train
    Test = pd.DataFrame(x_test)
    Test['class'] = y_test
    # 处理后返回处理过的少数类数据用于生成样本
    less_class = min(set(Train['class']), key=list(Train['class']).count)
    more_class = max(set(Train['class']), key=list(Train['class']).count)
    less_data_train = Train[Train['class'] == less_class]
    more_data_train = Train[Train['class'] == more_class]
    less_data_train = less_data_train.drop('class', axis=1)
    more_data_train = more_data_train.drop('class', axis=1)
    less_data_train = less_data_train.to_numpy()
    more_data_train = more_data_train.to_numpy()
    # num为需要生成的少数类数量
    num = Train.shape[0] - less_data_train.shape[0] * 2

    if judge == 0:
        # 计算出需要生成的少数类个数， 对多数类进行聚类找出聚类中心， 然后将每个少数类都与聚类中心进行三角形的计算
        kmeans = KMeans(n_clusters=math.ceil((num+0.0)/less_data_train.shape[0]), max_iter=300, n_init=10, init="k-means++", random_state=22)
        kmeans.fit(more_data_train)
        print(kmeans.cluster_centers_)
        return less_data_train, kmeans.cluster_centers_, less_data_train.shape[0]*(math.ceil((num+0.0)/less_data_train.shape[0])), less_class, more_class, Train
    elif judge == 1:
        return Train, Test
    # elif judge == 2:
    #     return all_data

