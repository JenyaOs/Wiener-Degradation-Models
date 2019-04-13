from Module.wienerModel import WienerModel, WienerModelWithExpTrend, WienerModelWithCovariate,\
    WienerModelWithPowerTrend, WienerModelWithPowerTrendWithCov
from Module.data import DataStorage
from Module.modeling import  ModelingData
from Module.paintingGraph import Graphics
from Module.design import designDirect_D
import numpy as np

#Тестирование модуля генерирования данных
def ModuleModeling(x0, _model, c=[0], type = 0, size=10):
    time, delta, value = [], [], []
    # Хранение данных в структуре
    data = DataStorage(time, delta, value, [])
    x = x0
    # print("истинные параметры:",  x)
    # создание  модели
    model = _model(data,  x, type)
    # моделирования данных
    test = ModelingData(model)
    cov = []
    for k in range(len(c)):

        for i in range(100):
            time.append(np.linspace(1e-6, 10, size))
            delta1 = test.GeneratorWienerProcessDelta(time[i], x, c[k])
            delta.append(delta1)
            value1 = test.GeneratorWienerProcessValues(delta1)
            value.append(value1)
            cov.append(c[k])

    print(len(time))
    data.updateAll(time, delta, value, cov)
    model = _model(data, x, type)
    return data, model

#Тестирование модуля оценивание параметров
def ModuleEstimate(x,model):
    return model.estimate_Parametrs(x)

#Тестирование модуля отображение графиков
def ModuleDemonstrated(_data, _x, _model, kIter, iIter, cov=[0]):
    Graph = Graphics('time', 'Z(t)','Fuction trend', _data, _model)
    for k in range(kIter):
        Graph.PaintValuesFunction([iIter*k, iIter*(k+1)])
        Graph.PaintFunctionTrend(_x, cov[k])
    Graph.ShowGraphic()
    #for k in range(kIter):

    #        Graph.PaintFunctionSurvival(_x, _model, 10, cov[k])
    #Graph.ShowGraphic()
    return

#Тестирование модуля планирования эксперимента
def ModulePlanning(_model, cov, _typeForMatrix,_bounds = [1,5]):

    print("D-optimal")
    return designDirect_D(_model, cov, _typeForMatrix, _bounds)

#Тестирование модуля проверки гипотез
def ModuleTestOfFit():
    return

def test_linear():
    kIter = 1#количество ковариационного показателя
    iIter = 5#количество объектов
    jIter = 10#количество измерения деградационного показателя

    #-----------------------------#
    #линейная модель
    xLinear= [1,1]
    dataLinear, modelLinear = ModuleModeling(xLinear, WienerModel)
    res = ModuleEstimate(xLinear, modelLinear)
    #ModuleDemonstrated(dataLinear, res, modelLinear, kIter, iIter)
    return ModulePlanning(modelLinear, [0], 1)

def test_power():
    kIter = 1  # количество ковариационного показателя
    iIter = 5  # количество объектов
    jIter = 10  # количество измерения деградационного показателя

    #степенная модель
    xPower = [1,1,2]
    dataPower, modelPower = ModuleModeling(xPower, WienerModelWithPowerTrend)
    resPower = ModuleEstimate(xPower, modelPower)
    ModuleDemonstrated(dataPower,resPower, modelPower, kIter, iIter)
    return ModulePlanning(modelPower, [0], 2)

def test_powerCov(size):

    #степенная с ковариатами
    kIter = 2
    iIter = 5
    jIter = 10

    xPowerWithCov = [.1, 1, 1, .5]
    cov = [2,5]
    type = 1
    dataPowerWithCov, modelPowerWithCov = ModuleModeling(xPowerWithCov, WienerModelWithPowerTrendWithCov, cov, type,size)
    resPowerWithCov = ModuleEstimate(xPowerWithCov, modelPowerWithCov)
    ModuleDemonstrated(dataPowerWithCov,resPowerWithCov, modelPowerWithCov, kIter, iIter, cov)
    return ModulePlanning(modelPowerWithCov,cov, 4, [2, 5])

def test_linearCov(size):
    #линейная с ковариатами
    kIter = 2
    iIter = 5
    jIter = 10


    cov = [2,5]
    xWithCov = [.1, 1, 2]
    dataWithCov, modelWithCov = ModuleModeling(xWithCov, WienerModelWithCovariate, cov, 1,size)
    resWithCov = ModuleEstimate(xWithCov, modelWithCov)
    ModuleDemonstrated(dataWithCov,resWithCov, modelWithCov, kIter, iIter, cov)
    return ModulePlanning(modelWithCov, cov, 3, [2, 5])

print("linear")
f = open("result.txt", "w+")
for i in range((1)):
    print((i+1)*10)
    s = test_powerCov((i+1)*10)
    f.write(str(s)+"\n")

f.close()

#print("power")
#test_powerCov()


