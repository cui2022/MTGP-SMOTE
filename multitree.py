# https://pastebin.com/QKMhafRq
import random
import statistics
from functools import partial
from operator import attrgetter
import numpy as np
from deap import gp, tools
from deap.gp import genGrow, genFull, __type__
import random
from collections import defaultdict, deque
from functools import partial, wraps



def cxOnePoint(ind1, ind2):
    """Randomly select crossover point in each individual and exchange each
    subtree with the point as root between each individual.

    :param ind1: First tree participating in the crossover.
    :param ind2: Second tree participating in the crossover.
    :returns: A tuple of two trees.
    """
    if len(ind1) < 2 or len(ind2) < 2:
        # No crossover on single node tree
        return ind1, ind2

    # List all available primitive types in each individual
    types1 = defaultdict(list)
    types2 = defaultdict(list)

    if ind1.root.ret == __type__:
        # Not STGP optimization
        types1[__type__] = range(1, len(ind1))
        types2[__type__] = range(1, len(ind2))
        common_types = [__type__]
    else:
        for idx, node in enumerate(ind1[1:], 1):
            types1[node.ret].append(idx)
        for idx, node in enumerate(ind2[1:], 1):
            types2[node.ret].append(idx)
        common_types = set(types1.keys()).intersection(set(types2.keys()))

    if len(common_types) > 0:
        type_ = random.choice(list(common_types))

        index1 = random.choice(types1[type_])
        index2 = random.choice(types2[type_])

        slice1 = ind1.searchSubtree(index1)
        slice2 = ind2.searchSubtree(index2)

        # type_1 = ['add', 'subtract']
        # type_2 = ['multiply', 'np_protectedDiv']
        # while ind1[slice1.start].name in type_1 and ind2[slice2.start].name in type_2 or ind1[slice1.start].name in type_2 and ind2[slice2.start].name in type_1:
        #     # if ind1[slice1.start].name in type_1 and ind2[slice2.start].name in type_1 or ind1[slice1.start].name
        #     # in type_2 and ind2[slice2.start].name in type_2 : break or (ind1[slice1.start].name not in type_1 and
        #     # ind2[slice2.start].name not in type_2 and  ind1[ slice1.start].name not in type_2 and ind2[
        #     # slice2.start].name not in type_1):
        #     index1 = random.choice(types1[type_])
        #     index2 = random.choice(types2[type_])
        #     slice1 = ind1.searchSubtree(index1)
        #     slice2 = ind2.searchSubtree(index2)
        ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

    return ind1, ind2


def xmate1(ind1, ind2):
    # 根据indiviual.tree_value做评估选出差的一半个体进行交叉
    num1 = range(len(ind1))
    num2 = range(len(ind2))
    list1 = []
    list2 = []
    # sorted_id = sorted(range(len(ind2.tree_value)), key=lambda k: ind2.tree_value[k], reverse=False)
    j = 0
    #
    # ind_fitness = ind1.tree_value
    # fit_mean = np.mean(ind_fitness)
    # fit_max = np.max(ind_fitness)
    # for i in num1:
    #     choice = random.random()
    #     if ind_fitness[i] > fit_mean:
    #         cxpbb = 0.8 * abs((fit_max - ind_fitness[i]) / (fit_max - fit_mean))
    #     else:
    #         cxpbb = 0.8
    #     if choice < cxpbb:
    #         i2 = random.randint(1, len(ind2) - 1)
    #         ind1[i], ind2[sorted_id[j]] = cxOnePoint(ind1[i], ind2[i2])
    #         j = j + 1
    ind_fitness = ind1.tree_value
    ind_distance = ind1.dis
    # i1 = random.randrange(len(ind1))
    # i2 = random.randrange(len(ind2))
    # ind1[i1], ind2[i2] = gp.cxOnePoint(ind1[i1], ind2[i2])
    # indx_mate2 = random.sample(num2, int(len(ind2) / 5))
    # indx_mate1 = random.sample(num1, int(len(ind1) / 5))
    # for i1, i2 in zip(indx_mate1, indx_mate2):
    #         ind1[i1], ind2[i2] = gp.cxOnePoint(ind1[i1], ind2[i2])
    for i in num1:
        # choice = random.random()
        if ind_fitness[i] < 90 or ind_distance[i] < 0:
            # i2 = random.randint(1, len(ind2) - 1)
            # ind1[i], ind2[sorted_id[j]] = cxOnePoint(ind1[i], ind2[sorted_id[j]])
            # ind1[i], ind2[i2] = cxOnePoint(ind1[i], ind2[i2])
            ind1[i], ind2[i] = cxOnePoint(ind1[i], ind2[i])
            j = j + 1
    # print(cxpbb)
    return ind1, ind2


def xmut1(ind, expr):
    num = range(len(ind))

    # ind_fitness = ind.tree_value
    # fit_mean = np.mean(ind_fitness)
    # fit_max = np.max(ind_fitness)
    # # 自适应交叉变异策略
    # for i in num:
    #     choice = random.random()
    #     if ind_fitness[i] > fit_mean:
    #         mutpbb = 0.2 * abs((fit_max - ind_fitness[i]) / (fit_max - fit_mean))
    #     else:
    #         mutpbb = 0.2
    #     if choice < mutpbb or ind_fitness[i] == 0:
    #         indx = gp.mutUniform(ind[i], expr, pset=ind.pset)
    #         ind[i] = indx[0]
    ind_fitness = ind.tree_value
    ind_distance = ind.dis
    # i1 = random.randrange(len(ind))
    # indx = gp.mutUniform(ind[i1], expr, pset=ind.pset)
    # ind[i1] = indx[0]
    # num = range(len(ind))
    # indx_mate = random.sample(num, int(len(ind) / 5))
    # for i1 in indx_mate:
    #     indx = gp.mutUniform(ind[i1], expr, pset=ind.pset)
    #     ind[i1] = indx[0]
    for i in num:
        # choice = random.random()
        if ind_fitness[i] < 90 or ind_distance[i] < 0:
            indx = gp.mutUniform(ind[i], expr, pset=ind.pset)
            ind[i] = indx[0]
    return ind,


def copy(offsprint_, ind):
    num1 = range(len(offsprint_))
    num = range(len(ind))
    ind_fitness = ind.tree_value
    ind_distance = ind.dis
    # 寻找最好的个体进行复制
    for i in num:
            max_ = 0
            max_num = 0
            for j, k in zip(offsprint_, range(len(offsprint_))):
                if j.tree_value[i] > max_ and j.dis[i] > 0:
                    max_num = k
            ind[i] = offsprint_[max_num][i]
    return ind,


# 第二阶段对百分之20个体进行交叉
def xmate2(ind1, ind2):
    num1 = range(len(ind1))
    num2 = range(len(ind2))
    data_sort = sorted(ind1.tree_value)
    value_3 = data_sort[int(len(ind1) / 2)]
    sorted_id = sorted(range(len(ind2.tree_value)), key=lambda k: ind2.tree_value[k], reverse=True)
    j = 0
    for i in num1:
        if ind1.tree_value[i] >= value_3:
            i2 = random.randint(1, len(ind2) - 1)
            ind1[i], ind2[sorted_id[j]] = gp.cxOnePoint(ind1[i], ind2[i2])
            j = j + 1

    return ind1, ind2


def xmut2(ind, expr):
    # i1 = random.randrange(len(ind))
    # indx = gp.mutUniform(ind[i1], expr, pset=ind.pset)
    # ind[i1] = indx[0]
    # num = range(len(ind))
    # indx_mate = random.sample(num, int(len(ind) / 5))
    # for i1 in indx_mate:
    #     indx = gp.mutUniform(ind[i1], expr, pset=ind.pset)
    #     ind[i1] = indx[0]
    num = range(len(ind))
    data_sort = sorted(ind.tree_value)
    value_3 = data_sort[int(len(ind) / 2)]
    for i in num:
        if ind.tree_value[i] >= value_3:
            indx = gp.mutUniform(ind[i], expr, pset=ind.pset)
            ind[i] = indx[0]

    return ind,


# Direct copy from tools - modified for individuals with GP trees in an array
def xselDoubleTournament(individuals, k, fitness_size, parsimony_size, fitness_first):
    assert (1 <= parsimony_size <= 2), "Parsimony tournament size has to be in the range [1, 2]."

    def _sizeTournament(individuals, k, select):
        chosen = []
        for i in range(k):
            # Select two individuals from the population
            # The first individual has to be the shortest
            prob = parsimony_size / 2.
            ind1, ind2 = select(individuals, k=2)

            lind1 = sum([len(gpt) for gpt in ind1])
            lind2 = sum([len(gpt) for gpt in ind2])
            if lind1 > lind2:
                ind1, ind2 = ind2, ind1
            elif lind1 == lind2:
                # random selection in case of a tie
                prob = 0.5

            # Since size1 <= size2 then ind1 is selected
            # with a probability prob
            chosen.append(ind1 if random.random() < prob else ind2)

        return chosen

    def _fitTournament(individuals, k, select):
        chosen = []
        for i in range(k):
            aspirants = select(individuals, k=fitness_size)
            chosen.append(max(aspirants, key=attrgetter("fitness")))
        return chosen

    if fitness_first:
        tfit = partial(_fitTournament, select=tools.selRandom)
        return _sizeTournament(individuals, k, tfit)
    else:
        tsize = partial(_sizeTournament, select=tools.selRandom)
        return _fitTournament(individuals, k, tsize)


def genMTHalfAndHalf(pset, min_, max_, type_=None):
    """Generate an expression with a PrimitiveSet *pset*.
    Half the time, the expression is generated with :func:`~deap.gp.genGrow`,
    the other half, the expression is generated with :func:`~deap.gp.genFull`.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: Either, a full or a grown tree.
    """
    method = random.choice((genGrow, genFull))
    return method(pset, min_, max_, type_), method(pset, min_, max_, type_)
