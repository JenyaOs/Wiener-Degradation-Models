from Module.data import DataStorage
from Module.wienerModel import WienerModelWithPowerTrend, WienerModel, WienerModelWithExpTrend
from Module.modeling import ModelingData
from Module.test import Kolmogorov_Test,CramerVonMisesSmirnov_Test,AndersonDarling_Test,Xi_2,ToCheckAdequacyOfModel, GetF, powerRight
from Module.paintingGraph import Graphics, demonstateModelWithoutCov
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

from scipy.optimize import minimize

"""def GenerationDataOFModel(x0, _time, model):
    time, delta, value = [], [], []
    # Хранение данных в структуре
    data = DataStorage(time, delta, value)
    x = x0
    #print("истинные параметры:", x)
    # создание  модели
    model = model(data, x[0], x[1], x[2])
    # моделирования данных
    test = ModelingData(model)
    for i in range(40):
        time.append(_time)
        delta1 = test.GeneratorWienerProcessDelta(time[i], x)
        delta.append(delta1)
        value1 = test.GeneratorWienerProcessValues(delta1)
        value.append(value1)

    data.updateAll(time, delta, value)
    demonstateModelWithoutCov([0, 10], [0, 15], "time", "Z(t)", model, data)
    return data

def output_array(filename, A):
    file = open(filename,"a")
    for i in range(len(A)):
        file.write(str(A[i])+"\n")
    file.close()
    return

def output_data(filename, ar):
    file = open(filename, "w")
    for i in range(len(ar)):
        for j in range(len(ar[i])):
            file.write(str(ar[i][j])+" ")
        file.write("\n")
    return

def input_Ar(filename):
    Ar = []
    filename = open(filename,"r")
    print(filename.readline())
    print(filename.readline())
    line = filename.readline()
    while(line):
        Ar.append(float(line))
        line = filename.readline()
    #graph(Ar, label, criterion)
    filename.close()
    return Ar

def input_Mat(filename):
    Ar = []
    filename = open(filename,"r")
    #print(filename.readline())
    #print(filename.readline())
    line = filename.readline()
    while(line):
        ll = line.split(" ")
        cur = [float(ll[i]) for i in range(len(ll)-1)]
        Ar.append(cur)
        line = filename.readline()
    #graph(Ar, label, criterion)
    #ecdf =ECDF(Ar)
    #ecdf =
    filename.close()
    return Ar


def GetMinimizeParanetrForChi2(x, data, model):
    F = GetF(data.delta, x, data.time, model)
    ArraySk3 = (Xi_2(5).get_Sk(F))
    return ArraySk3

def ResearchFunctionChi2(M, data):

    X=[]
    for i in range(M):
        print(i)
        x=[1, 1]
        model = WienerModel(data[i],x)
        
        #res = model.estimate_Parametrs(x)
        #model.updateModel(data[i], res)
        St_Chi2 = minimize(GetMinimizeParanetrForChi2, x, args=(data[i], model), method="nelder-mead")
        X.append(St_Chi2.fun)

    return X

def ResearchFunction(M, data):
    K=[]
    A=[]
    C=[]
    X=[]
    for i in range(M):
        print(i)
        x=[1, 1, 1.2]
        model = WienerModelWithPowerTrend(data[i], x)
        
        res = model.estimate_Parametrs(x)
        model.updateModel(data[i], res)
        F = GetF(data[i].delta, res, data[i].time, model)
        ArraySk = (Kolmogorov_Test().get_Sk(F))
        ArraySk1 = (AndersonDarling_Test().get_Sk(F))
        ArraySk2 = (CramerVonMisesSmirnov_Test().get_Sk(F))
        ArraySk3 = (Xi_2(5).get_Sk(F))
        K.append(ArraySk)
        A.append(ArraySk1)
        C.append(ArraySk2)
        X.append(ArraySk3)

    return K, A, C, X

def OutPut(K, A, C, X, H):
    output_array(Kolmogorov_Test().get_Name()+H+".dat", K)
    output_array(AndersonDarling_Test().get_Name()+H+".dat", A)
    output_array(CramerVonMisesSmirnov_Test().get_Name()+H+".dat", C)
    output_array(Xi_2(5).get_Name()+H+".dat", X)
    return


def output_line(file,line):
    file = open(file, "w")
    file.readline(line[0])
    file.readline(line[1])
    file.close()

    return

def OutPutFirstLine(H):
    output_array(Kolmogorov_Test().get_Name()+H+".dat", ["K "+H,"0 16600"],)
    output_array(AndersonDarling_Test().get_Name()+H+".dat", ["AD "+H,"0 16600"])
    output_array(CramerVonMisesSmirnov_Test().get_Name()+H+".dat",  ["KMC "+H,"0 16600"])
    output_array(Xi_2(5).get_Name()+H+".dat",  ["CHI "+H,"0 16600"])
    return

"""

#-----------------новые функции----------------
#генерация данных
def GenerationDataOFModel(x0, _time, K, model):

    time, delta, value = [], [], []

    # Хранение данных в структуре
    data = DataStorage(time, delta, value)
    x = x0

    # создание  модели
    model = model(data, x[0], x[1], x[2])

    # моделирования данных в соответствии выбранной модели
    test = ModelingData(model)

    for i in range(K):
        time.append(_time)
        delta1 = test.GeneratorWienerProcessDelta(time[i], x)
        delta.append(delta1)
        value1 = test.GeneratorWienerProcessValues(delta1)
        value.append(value1)

    #обновление структуры данных
    data.updateAll(time, delta, value)

    return data

#исследования 1
#получение распределения статистик для каждого критерия
def ResearchFunction(data, model, M=16600):
    K = []
    A = []
    C = []
    X = []
    for i in range(M):

        print(i)
        #создание модели гипотезы о виде которой мы проверяем
        x = [1, 1, 1.2]
        model = WienerModelWithPowerTrend(data[i], x)


        #оценивание параметров модели
        res = model.estimate_Parametrs(x)

        #обновление параметров модели
        model.updateModel(data[i], res)

        # приведение к нужному виду выборки (преобразование для проверки на равномерность)
        F = GetF(data[i].delta, res, data[i].time, model)

        #получение статистик по каждому из критериев
        ArraySk = (Kolmogorov_Test().get_Sk(F))
        K.append(ArraySk)

        ArraySk1 = (AndersonDarling_Test().get_Sk(F))
        A.append(ArraySk1)

        ArraySk2 = (CramerVonMisesSmirnov_Test().get_Sk(F))
        C.append(ArraySk2)

        ArraySk3 = (Xi_2(5).get_Sk(F))
        X.append(ArraySk3)

    return K, A, C, X


#исследования 2
#получение распределения статистик для критерия хи2 с минимизированной статистикой
def GetMinimizeParanetrForChi2(x, data, model):
    F = GetF(data.delta, x, data.time, model)
    ArraySk3 = (Xi_2(5).get_Sk(F))
    return ArraySk3

def ResearchFunctionChi2(data, _model, M=16600):
    X=[]
    for i in range(M):
        print(i)
        model = _model(data[i], 1, 1, 1)
        x=[1, 1, 1]
        #res = model.estimate_Parametrs(x)
        #model.updateModel(data[i], res)
        St_Chi2 = minimize(GetMinimizeParanetrForChi2, x, args=(data[i], model), method="nelder-mead")
        X.append(St_Chi2.fun)

    return X

#вывод массива
def OutPutArray(filename, A):
    file = open(filename,"a")
    for i in range(len(A)):
        file.write(str(A[i])+"\n")
    file.close()
    return

#вывод результатов исследований
def OutPutFirstLine(path, H):
    output_array(path+Kolmogorov_Test().get_Name()+H+".dat", ["K "+H,"0 16600"],)
    output_array(path+AndersonDarling_Test().get_Name()+H+".dat", ["AD "+H,"0 16600"])
    output_array(path+CramerVonMisesSmirnov_Test().get_Name()+H+".dat",  ["KMC "+H,"0 16600"])
    output_array(path+Xi_2(5).get_Name()+H+".dat",  ["CHI "+H,"0 16600"])
    return

#вывод результатов только для Колмогорова, Кр-Мизеса-Смирнов, Дарлига и простого Хи2
def OutPutResultsTests(path, H, K,A,C,X):
    output_array(path+Kolmogorov_Test().get_Name()+H+".dat", K)
    output_array(path+AndersonDarling_Test().get_Name()+H+".dat", A)
    output_array(path+CramerVonMisesSmirnov_Test().get_Name()+H+".dat", C)
    output_array(path+Xi_2(5).get_Name()+H+".dat", X)
    return

def output_array(filename, A):
    file = open(filename,"a")

    for i in range(len(A)):
        file.write(str(A[i])+"\n")
    file.close()
    return

def OutPutResultsChi2(path, H, X):
    output_array(path+Xi_2(5).get_Name()+H+".dat",  X)
    return

#считываение из файла
def InPutArray(filename):
    file = open(filename,"r")
    print(file.readline())
    print(file.readline())
    res = file.readline()
    st = []
    while(res):
        st.append(float(res))
        res = file.readline()
    file.close()
    return st

#исследование мощности
def ResearchOfPower(fileH0, fileH1, alpha):
    st = InPutArray(fileH0)
    H0 = ECDF(st)
    st1 = InPutArray(fileH1)
    H1 = ECDF(st1)
    print(powerRight(H0.y, H1.y, alpha))
    return

