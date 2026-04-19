import math
import operator
import random
import sys
import time
import numpy as np
import pandas as pd
from evolutionary_forest.component.evaluation import quick_evaluate
from sklearn.svm import SVC

import Genetic_processes
from deap import gp
from deap import base
from deap import creator
from deap import tools
from multitree import xmut2, xmate2, xmate1, xmut1, copy
from read_initial_data import read_data, calEuclidean
import multitree
from Performence import Gmean

file = "ecoli-0-1-4-6_vs_5"
data_name = '.\\initial_dataset\\1\\' + file + '.dat'
less_data, more_data, NUMS, less_class, more_class, Train = read_data(data_name, 0)

#
NUM_TREES = NUMS
# Population size
POP_SIZE = 512
# Number of generations
NGEN = 1
# Crossover probability
CXPB = 0.8
# Mutation probability, should sum to one together with CXPB if varOr (varAnd doesn't care)
MUTPB = 0.2
ELITISM = 10
# LIMIT = 10
LIMIT = 50

# 基于分类器的多树gp生成不平衡数据的少数类样本，其evalute通过分类器分类结果进行评价，多树gp一次产生多个少数类样本，直接与原训练集样本组合，然后再次划分训练集与测试集进行结果评估
def np_protectedDiv(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = left
    return x


def writeinto_excel(data, i):
    result_end = pd.DataFrame(data)
    file_name = '.\\GP_dataset\\' + file + '\\produce_set\\' + str(NGEN) + '\\' + file + " NEGN " + str(
        NGEN) + " seed " + str(
        i) + ".xlsx"
    # + time.strftime(' %Y.%m.%d %H:%M:%S ',time.localtime(time.time())).replace(":", "-")
    writer = pd.ExcelWriter(file_name)  # 写入Excel文件
    result_end.to_excel(writer, 'less', header=None, index=None)  # ‘less’是写入excel的sheet名
    writer.save()


# 保存表达式
def write_excel(data1, data2, i, b_time, n_time):
    data1 = pd.DataFrame(data1)
    data2.reset_index(drop=True, inplace=True)
    result_end = pd.concat([data1, data2], axis=1)
    result_end['TRAIN_TIME'] = n_time - b_time
    file_name = '.\\GP_dataset\\' + file + '\\expression_set\\' + str(
        NGEN) + '\\' + "data and expression " + file + " NEGN " + str(
        NGEN) + " seed " + str(
        i) + ".xlsx"
    # 保存至GP_rough-initial_pop file_name = '.\\GP_rough\\initial_pop\\' + file + '\\' + "random_data and expression "
    # + file + " NEGN 50 seed " + str( i) + ".xlsx"
    writer = pd.ExcelWriter(file_name)  # 写入Excel文件
    result_end.to_excel(writer, 'less', header=None, index=None)  # ‘less’是写入excel的sheet名
    writer.save()


# def np_if(a, b, c):
#     return np.where(a < 0, b, c)


def evaluate1_nparray(individual, toolbox, less_data, more_data):
    initial_lessdata = less_data.T
    result = 0
    distance = 0
    value_list = []
    dis_list = []
    vec = np.zeros(shape=(NUMS, less_data.shape[0]))
    for i, expr in enumerate(individual):
        #func = toolbox.compile(expr=expr)
        #pgout = func(*initial_lessdata)
        # 下面语句可以加速
        pgout = quick_evaluate(expr, pset, initial_lessdata.T, prefix='x')
        # 目标多数类少数类与产生的样本构成三个支点，计算角度，期望钝角并且多数类与目标样本的边要更长
        a = calEuclidean(pgout, initial_lessdata[i % initial_lessdata.shape[0], :])  # 目标少数类与样本边长
        # print(more_data[int(i / more_data.shape[0]), :])
        b = calEuclidean(pgout, more_data[int(i / initial_lessdata.shape[0]), :])  # 目标多数类与样本边长
        c = calEuclidean(initial_lessdata[i % initial_lessdata.shape[0], :], more_data[int(i / initial_lessdata.shape[0]), :])  # 目标多数类与少数类边长
        judge = 0
        for kk in range(less_data.T.shape[0]):
            if (less_data.T[kk, :] == pgout).all():
                    judge = 1
                    break
        # 计算角度
        if a == 0 or b == 0 or judge == 1:
            C = 0
        else:
            co = (c * c - a * a - b * b) / (-2 * a * b)
            if co > 1:
                co = 1
            if co < -1:
                co = -1
            C = math.degrees(math.acos(co))  # 夹角3
        distance = b - a

        value_list.append(C)
        dis_list.append(distance)
        result += C * b/(b+a)
        vec[i, ::] = pgout
    individual.tree_value = value_list
    individual.dis = dis_list
    fitness1 = result

    return [fitness1, ]

print(NUMS)
less_data = less_data.T
num_instances = less_data.shape[0]
num_features = less_data.shape[1]

toolbox = base.Toolbox()
pset = gp.PrimitiveSet("MAIN", num_features, prefix="x")

# deap you muppet
pset.context["array"] = np.array
pset.addPrimitive(np.add, 2)
pset.addPrimitive(np.subtract, 2)
pset.addPrimitive(np.multiply, 2)
pset.addPrimitive(np_protectedDiv, 2)
# pset.addPrimitive(np.maximum, 2)
# pset.addPrimitive(np.minimum, 2)
# pset.addPrimitive(np_if, 3)
pset.addEphemeralConstant("rand", ephemeral=lambda: random.uniform(0, 1))

creator.create("FitnessMax", base.Fitness, weights=(+1.0,))

# minimisation so -1.0
# creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax, pset=pset, tree_value=list, dis=list)
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("tree", tools.initIterate, gp.PrimitiveTree, toolbox.expr)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.tree, n=NUM_TREES)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate1", evaluate1_nparray, toolbox=toolbox, less_data=less_data, more_data=more_data)
toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("mate1", xmate1)
toolbox.register("expr_mut", gp.genFull, min_=1, max_=5)
toolbox.register("mutate1", xmut1, expr=toolbox.expr_mut)
toolbox.register("mate2", xmate2)
toolbox.register("mutate2", xmut2, expr=toolbox.expr_mut)

# 新环境这句话需要修改gp.py里的 def wrapper(*args, **kwargs): 才能够正常限制树高
toolbox.decorate("mate1", gp.staticLimit(key=operator.attrgetter("height"), max_value=LIMIT))
# toolbox.decorate("mutate1", gp.staticLimit(key=operator.attrgetter("height"), max_value=LIMIT))
toolbox.register("copy", copy)


def main(i):
    random.seed(i)
    np.random.seed(i)

    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = Genetic_processes.eaSimple(i, pop, toolbox, CXPB, MUTPB, NGEN, stats, halloffame=hof, verbose=True)
    result = np.zeros(shape=(NUMS, less_data.shape[0]))
    result_f = pd.DataFrame(columns=['expression'])
    print(hof[0].fitness)

    best = hof[0]
    for i in range(NUMS):
        # 确保没有范围外的生成个体（这种个体很少）
        if best.tree_value[i] < 90 or best.dis[i] < 0:
            choice1 = random.randint(0, less_data.T.shape[0] - 1)
            result[i, :] = less_data.T[choice1, :]
            result_f = pd.concat([result_f, pd.DataFrame({'expression': ["x"+str(choice1)]})])
            continue
        print(best.tree_value[i])
        print(pd.DataFrame({'expression': [str(best[i])]}))
        result_f = pd.concat([result_f, pd.DataFrame({'expression': [str(best[i])]})])
        func = toolbox.compile(expr=best[i])
        pgout = func(*less_data.T)
        result[i, :] = pgout
    # 不再直接保存hof[0],而是保存所有个体中，对应每棵树的最小的那一个
    # summ = 0
    # for kkkk in range(NUMS):  # 第一层循环表示遍历树的个数
    #     max_tree = 0
    #     max_num = 0
    #     judger = 0
    #     for j in range(POP_SIZE):  # 第二层循环遍历所有的个体，找出最大对应树且不与原数据中的树相同
    #         # print(POP_SIZE, j, kkkk)
    #         if pop[j].tree_value[kkkk] >= max_tree and pop[j].dis[kkkk] >= 0:
    #             func = toolbox.compile(expr=pop[j][kkkk])
    #             pre_re = func(*less_data.T)
    #             judge = 0
    #             # for kk in range(i):
    #             #     if (result[kk, :] == pre_re).all():
    #             for kk in range(less_data.T.shape[0]):
    #                 if (less_data.T[kk, :] == pre_re).all():
    #                     judge = 1
    #                     break
    #             if judge == 0:
    #                 judger = 1
    #                 max_tree = pop[j].tree_value[kkkk]
    #                 max_num = j
    #     if judger == 0 or max_tree < 90:
    #         # result[kkkk, :] = less_data.T[kkkk % less_data.T.shape[0], :]
    #         # result_f = pd.concat([result_f, pd.DataFrame({'expression': ["x"+str(kkkk % less_data.T.shape[0])]})])
    #         choice1 = random.randint(0, less_data.T.shape[0] - 1)
    #         choice2 = random.randint(0, less_data.T.shape[0] - 1)
    #         result[i, :] = (less_data.T[choice1, :] + less_data.T[choice2, :])/2
    #         result_f = pd.concat([result_f, pd.DataFrame({'expression': ["multiply(0.5, " + "add(x"+str(choice1)+", x"+str(choice2)+"))"]})])
    #     else:
    #         result_f = pd.concat([result_f, pd.DataFrame({'expression': [str(pop[max_num][kkkk])]})])
    #         func = toolbox.compile(expr=pop[max_num][kkkk])
    #         pgout = func(*less_data.T)
    #         result[kkkk, :] = pgout
    #         summ += max_tree
    # print(summ)
    return result, result_f, less_class, Train


