from Module.data import DataStorage
from Module.test import GetF, Kolmogorov_Test, CramerVonMisesSmirnov_Test, AndersonDarling_Test, Xi_2
from Module.wienerModel import WienerModel, WienerModelWithPowerTrend, WienerModelWithExpTrend, WienerModelWithPowerTrendWithCov
from Module.modeling import  ModelingData
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from scipy import stats


#проверка ковариат, на генерировать первый параметр, и по двум другим функциям оценить beta
#рассмотреть все различные варианты(от сетки, модели(разные функции тренда и функции от ковариат), измением параметров(масштаб и сдвиг))
#оценить по методу минимум Хи_2(функцию хи_2 минимизировать по параметрам модели, фиксируя число интервалов)
#мощность


#Проверка модели на адекватность x - оцененные параметры без коварит
def ToCheckAdequacyOfModel(M ,x, model, data):
    time = data.get_time()
    cov = data.get_covariates()
    testData = ModelingData(model)
    # накпливаем статистику Sk путем моделирования аналогичных выборок по оцененным параметрам
    K  = []
    A = []
    C = []
    X = []

    #Генерируем N выборок аналогичных оцененной и каждый раз оцениваем
    for i in range(M):
        print(i)
        dataStorage = DataStorage(time, [], [], cov)
        for j in range(len(time)):
            dataStorage.delta.append(testData.GeneratorWienerProcessDelta(time[j], x))
            dataStorage.value.append(testData.GeneratorWienerProcessValues(dataStorage.delta[j]))

        model.updateModel(dataStorage, x)
        res = model.estimate_Parametrs(x)
        F = GetF(dataStorage.delta, res, dataStorage.time, model)

        ArraySk = (Kolmogorov_Test().get_Sk(F))
        K.append(ArraySk)

        ArraySk1 = (AndersonDarling_Test().get_Sk(F))
        A.append(ArraySk1)

        ArraySk2 = (CramerVonMisesSmirnov_Test().get_Sk(F))
        C.append(ArraySk2)

        ArraySk3 = (Xi_2(5).get_Sk(F))
        X.append(ArraySk3)

    return K,A,C,X

#Исследования №1
#Линейная модель с равномерной сеткой
def ResearchLinearModel(x0, _time, M):
    time, delta, value = [],[],[]
    # Хранение данных в структуре
    data = DataStorage(time, delta, value)
    x = x0
    print("истинные параметры:",  x)
    # создание  модели
    model = WienerModel(data,x)
    #моделирования данных
    test = ModelingData(model)

    for i in range(20):
        time.append(_time)
        delta1 = test.GeneratorWienerProcessDelta(time[i], x)
        delta.append(delta1)
        value1 = test.GeneratorWienerProcessValues(delta1)
        value.append(value1)
    print(time[0])


    data.updateAll(time, delta, value)
    model = WienerModel(data,x)

    res = [model.estimate_sigma(),model.estimate_mu()]
    print("оцененные параметры:", res)

    return  ToCheckAdequacyOfModel(M, res, model, data)

#Исследования №2
#Степенная модель с равномерной сеткой
def ResearchPowerModel(x0,_time, M):
    time, delta, value = [],[],[]
    # Хранение данных в структуре
    data = DataStorage(time, delta, value)
    x = x0
    # создание  модели
    model = WienerModelWithPowerTrend(data,x)
    #моделирования данных
    test = ModelingData(model)

    for i in range(20):
        time.append(_time)
        delta1 = test.GeneratorWienerProcessDelta(time[i], x)
        delta.append(delta1)
        value1 = test.GeneratorWienerProcessValues(delta1)
        value.append(value1)


    data.updateAll(time, delta, value)
    model = WienerModelWithPowerTrend(data, x)
    res = model.estimate_Parametrs(x)

    return  ToCheckAdequacyOfModel(M, res, model, data)

#Исследования №3
#Степенная модель с равномерной сеткой
def ResearchExpModel(x0,_time, M):
    time, delta, value = [],[],[]
    # Хранение данных в структуре
    data = DataStorage(time, delta, value)
    x = x0
    print("истинные параметры:",  x)
    # создание  модели
    model = WienerModelWithExpTrend(data,x)
    #моделирования данных
    test = ModelingData(model)

    for i in range(20):
        time.append(_time)
        delta1 = test.GeneratorWienerProcessDelta(time[i], x)
        delta.append(delta1)
        value1 = test.GeneratorWienerProcessValues(delta1)
        value.append(value1)


    data.updateAll(time, delta, value)
    model = WienerModelWithExpTrend(data, x)

    res = model.estimate_Parametrs(x)
    print("оцененные параметры:", res)



    return  ToCheckAdequacyOfModel(M, res, model, data)

#Исследования №4
#Степенная модель с равномерной сеткой
def ResearchPowerModelWithCovariate(x0,_time, M):
    time, delta, value = [],[],[]
    # Хранение данных в структуре
    data = DataStorage(time, delta, value, [])
    x = x0
    print("истинные параметры:",  x)
    # создание  модели
    model = WienerModelWithPowerTrendWithCov(data, x, 2)
    #моделирования данных
    test = ModelingData(model)
    c = [1,2,3]
    cov = []
    for k in range(3):
        for i in range(10):
            time.append(_time)
            delta1 = test.GeneratorWienerProcessDelta(time[i], x, c[k])
            delta.append(delta1)
            value1 = test.GeneratorWienerProcessValues(delta1)
            value.append(value1)
            cov.append(c[k])


    data.updateAll(time, delta, value,cov)
    model = WienerModelWithPowerTrendWithCov(data, x, 2)

    res = model.estimate_Parametrs(x)
    print("оцененные параметры:", res)

    return  ToCheckAdequacyOfModel(M, res, model, data)

def graph(Ar, name, criterion):
    e = ECDF(Ar)
    plt.plot(e.x,e.y, linestyle='--', label=name)
    plt.plot(e.x, criterion.get_LimitDistribution(e.x),label="limit")
    plt.legend()
    plt.show()
    return

def output_array(filename, A, add_Inf=""):
    file = open(filename,"w")
    if(add_Inf):
        file.write(add_Inf)
    for i in range(len(A)):
        file.write(str(A[i])+"\n")
    file.close()
    return

def CompleteAdd_inf(criterion, x0, time, M):
    string = ""

    string+=criterion.get_Name()+", x0=["
    for i in range(len(x0)):
        string += str(x0[i])+", "
    string += "], time=["
    for i in range(len(time)):
        string += str(time[i]) + ", "
    string += "]\n"
    string +="0, " +str(M)+"\n"

    return string

def input(filename, criterion, label):
    Ar = []
    filename = open(filename,"r")
    print(filename.readline())
    print(filename.readline())
    line = filename.readline()
    while(line):
        Ar.append(float(line))
        line = filename.readline()
    graph(Ar, label, criterion)
    return

#_M количество повторений для нахождения предельной функции распределения статистик
def PartCheck():
    time = [i for i in range(10)]
    M = 10000
    x0 = [1,1,1.5,1]
    #какую модель исследуем
    K,A,C,X = ResearchPowerModelWithCovariate(x0, time, M)
    output_array("Kolmogorov_exp.txt", K)
    output_array("Anderson_exp.txt", A)
    output_array("KMC_exp.txt", C)
    output_array("Ch2_exp.txt", X)
    graph(K, "label", Kolmogorov_Test())
    graph(A, "label", AndersonDarling_Test())
    graph(C, "label", CramerVonMisesSmirnov_Test())
    graph(X, "label", Xi_2(5))
    return #Ar, CompleteAdd_inf(criterion,x0, time, M)

PartCheck()

#input("LinearModel/Андерсон-Дарлинг/test1_Андерсон-Дарлинг.txt", AndersonDarling_Test(), "test")


