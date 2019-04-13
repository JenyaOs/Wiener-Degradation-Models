from Module.data import DataStorage
from Module.wienerModel import WienerModelWithPowerTrendWithCov, WienerModel, WienerModelWithExpTrend, WienerModelWithPowerTrend
from Module.modeling import ModelingData

import numpy as np

def ResearchPowerModel(x0):
    time, delta, value = [], [], []
    # Хранение данных в структуре
    data = DataStorage(time, delta, value, [])
    x = x0
    # print("истинные параметры:",  x)
    # создание  модели
    model = WienerModelWithPowerTrend(data,  x)
    # моделирования данных
    test = ModelingData(model)

    for k in range(10):
        time.append([l for l in range(10)])
        delta1 = test.GeneratorWienerProcessDelta(time[k], x)
        delta.append(delta1)
        value1 = test.GeneratorWienerProcessValues(delta1)
        value.append(value1)

    data.updateAll(time, delta, value)
    model = WienerModelWithPowerTrend(data, x)

    res = model.estimate_Parametrs(x)

    return res

def ResearchPowerModelWithCovariate(x0, type):
    time, delta, value = [], [], []
    # Хранение данных в структуре
    data = DataStorage(time, delta, value, [])
    x = x0
    # print("истинные параметры:",  x)
    # создание  модели
    model = WienerModelWithPowerTrendWithCov(data, x,  type)
    # моделирования данных
    test = ModelingData(model)
    c = [1, 2, 3]
    cov = []
    for k in range(3):
        for i in range(5):
            time.append([l for l in range(20)])
            delta1 = test.GeneratorWienerProcessDelta(time[i], x, c[k])
            delta.append(delta1)
            value1 = test.GeneratorWienerProcessValues(delta1)
            value.append(value1)
            cov.append(c[k])

    data.updateAll(time, delta, value, cov)
    model = WienerModelWithPowerTrendWithCov(data,  x, type)

    res = model.estimate_Parametrs(x)

    return res


def output(filename, N, A):
    file = open(filename, "a")
    for i in range(N):
        for j in range(len(A[i])):
            file.write(str(A[i][j]) + " ")
        file.write("\n")


def getAverage(A, x0, N):
    print(A)
    print("ср. тетта:")
    average = [sum(A[0]) / N, sum(A[1]) / N, sum(A[2]) / N, sum(A[3]) / N]
    print(average)

    print("смещение:")
    print(x0[0] - average[0], x0[1] - average[1], x0[2] - average[2], x0[3] - average[3])

    print("Выборочная дисперсия:")
    print(sum((A[0] - average[0])*(A[0] - average[0])) / N, sum((A[1] - average[1])**2) / N, sum((A[2] - average[2])**2) / N,  sum((A[3] - average[3])**2) / N)

    return


def function(N, x0):
    A = []
    for i in range(N):
        print(i)
        A.append(ResearchPowerModelWithCovariate(x0,1))

    B = (np.array(A).transpose())
    getAverage(B, x0, N)
    return A


"""
x0 = [0.5, 1, 1, 0.5]
N = 16600
function(N, x0)

time1 = [0,0.1,0.3,0.5,0.9,1.5,2.4,5,7,8,9]
time2 = [0,0.01,0.1,0.3,0.35,0.44,0.49, 0.5, 0.8,0.83,0.9,0.92,1.5,2.4,3,5,7,8,9]
"""
print(ResearchPowerModel([0.5, 1, 1]))