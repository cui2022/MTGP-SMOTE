import math
import random
import sys

from deap import tools
from deap.tools import selRandom
import numpy as np
from matplotlib import pyplot as plt


def select1(individuals, k, tournsize=7, fit_attr="fitness"):
    chosen = []
    for i in range(k):
        # 先随机挑选tournsize个 个体
        chose = []
        while len(chose) == 0:
            aspirants = selRandom(individuals, tournsize)
            # 首先distance要满足条件
            max_ = 0
            for j in range(tournsize):
                if aspirants[j].fitness.values[0] > max_ and aspirants[j].fitness.values[1] > 0.0:
                    max_ = aspirants[j].fitness.values[0]
            for j in range(tournsize):
                if aspirants[j].fitness.values[0] == max_:
                    chose.append(aspirants[j])
        if len(chose) > 1:  # 当角度相同时，根据第二个值选择，选择更接近目标少数类样本的个体
            max_num = 0
            max_ = chose[0].fitness.values[1]
            for j in range(len(chose)):
                if chose[j].fitness.values[1] > max_:
                    max_num = j
                    max_ = aspirants[j].fitness.values[1]
            chosen.append(chose[max_num])
        else:
            chosen.append(chose[0])

    return chosen


def varAnd1(population, toolbox, cxpb, mutpb):
    offspring = []
    offspring_ = [toolbox.clone(ind) for ind in population]

    # 改进精英策略保留最好的
    max_ = 0
    ind_best_num = 0
    for i, ind in zip(range(len(offspring_)), offspring_):
        if ind.fitness.values[0] > max_:
            ind_best_num = i
            max_ = ind.fitness.values[0]

    ind_best = offspring_[ind_best_num]
    num = range(len(ind_best))
    for i in num:
        max_ = 0
        max_num = 0
        for j, k in zip(offspring_, range(len(offspring_))):
            if j.tree_value[i] > max_ and j.dis[i] > 0:
                max_num = k
        ind_best[i] = offspring_[max_num][i]

    # 保存并删除最好的，其余个体进行交叉变异
    del offspring_[ind_best_num].fitness.values
    offspring.append(ind_best)
    offspring_ = [ind for ind in offspring_ if ind.fitness.valid]

    # 针对样本里的每个个体做交叉或者变异， 只进行一种操作
    offspring_2 = [toolbox.clone(ind) for ind in offspring_]
    offspring_fitness = [ind.fitness.values[0] for ind in offspring_2]
    fit_mean = np.mean(offspring_fitness)
    fit_max = np.max(offspring_fitness)

    for i in range(len(offspring_)):
        cxpbb = 0.7
        mutpbb = 0.2
        copybb = 0.1
        choice = random.random()
        # 自适应交叉变异策略
        # if offspring_fitness[i] > fit_mean:
        #     cxpbb = cxpb*abs((fit_max-offspring_fitness[i])/(fit_max-fit_mean))
        #     mutpbb = mutpb*abs((fit_max-offspring_fitness[i])/(fit_max-fit_mean))
        #     if cxpbb == 0 and mutpbb == 0:
        #         cxpbb = 0
        #         mutpbb = 0.1
        # else:
        #     cxpbb = cxpb
        #     mutpbb = mutpb
        # print(cxpbb, mutpbb)
        # 防止早熟，设置一个默认突变概率
        # if choice < 0.05:
        #     ind = offspring_[i]
        #     ind, = toolbox.mutate1(ind)
        #     del ind.fitness.values
        #     offspring.append(ind)
        # el
        if choice < cxpbb:
            ind1 = offspring_[i]
            num = random.randint(i, len(offspring_) - 1)
            while num == i:
                num = random.randint(0, len(offspring_) - 1)
            ind2 = toolbox.clone(offspring_[num])
            ind1, ind2 = toolbox.mate1(ind1, ind2)
            del ind1.fitness.values
            offspring.append(ind1)
        elif choice < cxpbb + mutpbb:
            ind = offspring_[i]
            ind, = toolbox.mutate1(ind)
            del ind.fitness.values
            offspring.append(ind)
        # 新的算子：复制，将球体外的树替换为当前最优结果，加快演化速度
        elif choice < cxpbb + mutpbb + copybb:
            ind = offspring_[i]
            ind, = toolbox.copy(offspring_, ind)
            del ind.fitness.values
            offspring.append(ind)
        else:
            offspring.append(offspring_[i])
    return offspring


def varAnd2(population, toolbox, cxpb, mutpb):

    offspring = []
    offspring_ = [toolbox.clone(ind) for ind in population]
    choice = random.random()
    # 精英策略保留最好的
    max = 0
    ind_best = offspring_[0]
    ind_best_num = 0
    for i, ind in zip(range(len(offspring_)), offspring_):
        if ind.fitness.values[0] > max:
            ind_best = ind
            ind_best_num = i
            max = ind.fitness.values[0]
    # 保存并删除最好的，其余个体进行交叉变异
    # offspring_[ind_best_num].fitness.vaild = False
    del offspring_[ind_best_num].fitness.values
    offspring.append(offspring_[ind_best_num])
    offspring_ = [ind for ind in offspring_ if ind.fitness.valid]
    # offspring_ = offspring_.remove(ind_best)
    # 针对样本里的每个个体做交叉或者变异， 只进行一种操作
    for i in range(len(offspring_)):
        if choice < cxpb:
            ind1 = offspring_[i]
            num = random.randint(0, len(offspring_)-1)
            while num == i:
                num = random.randint(0, len(offspring_) - 1)
            ind2 = toolbox.clone(offspring_[num])
            ind1, ind2 = toolbox.mate2(ind1, ind2)
            del ind1.fitness.values
            offspring.append(ind1)
        elif choice < cxpb + mutpb:
            ind = offspring_[i]
            ind, = toolbox.mutate2(ind)
            del ind.fitness.values
            offspring.append(ind)
    return offspring


def eaSimple(sed, population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate1, invalid_ind)
    # fitnesses = toolbox.map(toolbox.evaluate2, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # if halloffame is not None:
    #     halloffame.update(population)

    # record = stats.compile(population) if stats else {}
    # logbook.record(gen=0, nevals=len(invalid_ind), **record)
    # if verbose:
    #     print(logbook.stream)
    maxFitnessValues = []
    meanFitnessValues = []
    # Begin the generational process
    print(1)
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = varAnd1(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        fitnesses = toolbox.map(toolbox.evaluate1, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        # # Update the hall of fame with the generated individuals
        # if gen >= nums:
        #     if halloffame is not None:
        #         halloffame.update(offspring)

        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # statistics

        fitnessValues = [ind.fitness.values[0] for ind in population]

        maxFitnessValue = math.log(max(fitnessValues))
        meanFitnessValue = math.log(sum(fitnessValues) / len(population))
        print(sum(fitnessValues), len(population))
        maxFitnessValues.append(maxFitnessValue)
        meanFitnessValues.append(meanFitnessValue)

        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
        record = stats.compile(population)
        logbook = tools.Logbook()
        logbook.record(gen=gen, evals=30, **record)
        logbook.header = "gen", "avg", "min", "max"
        if verbose:
            print(logbook)
    plt.plot(maxFitnessValues, color="red")
    plt.plot(meanFitnessValues, color="green")
    plt.xlabel("Generation")
    plt.ylabel("Max / Average Fitness")
    plt.title("Max and Average fitness over Generation")
    plt.savefig('迭代趋势'+str(sed), dpi=1000)
    plt.close()
    return population, logbook