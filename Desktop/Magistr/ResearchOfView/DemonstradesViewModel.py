from Module.wienerModel import WienerModel, WienerModelWithExpTrend, \
    WienerModelWithPowerTrend, WienerModelWithPowerTrendWithCov
from Module.data import DataStorage
from Module.modeling import  ModelingData
from Module.paintingGraph import Graphics

#отрисовка без ковариатами
def demonstateModelWithoutCov(x_lim, y_lim, x_name, y_name, model, data):
    Graph = Graphics(x_lim, y_lim, x_name, y_name, data)
    Graph.PaintValuesFunction([0, len(data.get_delta())])
    Graph.PaintFunctionTrend(model, model.get_parameters())
    Graph.ShowGraphic()
    Graph.ShowGraphic()
    return

#отрисовка с ковариат ВНИМАНИЕ не универсальна, надо прописать ковариаты, ПЕРЕЕДЕЛАТЬ
def demonstateModelWithCov(x_lim, y_lim, x_name, y_name, model, data):
    Graph = Graphics(x_lim, y_lim, x_name, y_name, data)
    Critical_level = 10
    MaxTime = 1000
    print(len(data.get_delta()))
    cov = [1,2,3]
    for i in range(len(cov)):
        Graph.PaintValuesFunction([i*5, (i+1)*5])
        Graph.PaintFunctionTrend(model, model.get_parameters(), cov[i])
        #Graph.PaintCriticalLevel(Critical_level)
        #Graph.ShowGraphic()

    Graph.ShowGraphic()
    return

def demonstateFuncSurvivalWithCov(x_lim, y_lim, x_name, y_name, model, data, result):
    Graph = Graphics(x_lim, y_lim, x_name, y_name, data)
    Critical_level = 10
    MaxTime = 10000
    print(len(data.get_delta()))
    cov = [1,2,3]
    for i in range(len(cov)):
        for k in range(len(result)):
            Graph.PaintFunctionSurvival(MaxTime, result[k], model, 5, cov[i], "grey", "--")

        Graph.PaintFunctionSurvival(MaxTime, model.get_parameters(), model, 5, cov[i])
        Graph.UpdateSetting(x_lim, y_lim, x_name, y_name)
        #Graph.PaintCriticalLevel(Critical_level)
        #Graph.ShowGraphic()

    Graph.ShowGraphic()
    return

#Исследования №1
#Линейная модель с равномерной сеткой
def ResearchLinearModel(x0):
    time, delta, value = [],[],[]
    # Хранение данных в структуре
    data = DataStorage(time, delta, value)
    x = x0
    print("истинные параметры:",  x)
    # создание  модели
    model = WienerModel(data,x[0],x[1])
    #моделирования данных
    test = ModelingData(model)

    for i in range(20):
        time.append([i for i in range(10)])
        delta1 = test.GeneratorWienerProcessDelta(time[i], x)
        delta.append(delta1)
        value1 = test.GeneratorWienerProcessValues(delta1)
        value.append(value1)
    print(time[0])

    data.updateAll(time, delta, value)
    model = WienerModel(data,x[0],x[1])
    demonstateModelWithoutCov([0, 10], [-10, 15], "time", "Z(t)", model,data)
    return

#Исследования №2
#Степенная модель с равномерной сеткой
def ResearchPowerModel(x0):
    time, delta, value = [],[],[]
    # Хранение данных в структуре
    data = DataStorage(time, delta, value)
    x = x0
    print("истинные параметры:",  x)
    # создание  модели
    model = WienerModelWithPowerTrend(data,x)
    #моделирования данных
    test = ModelingData(model)

    for i in range(20):
        time.append([i for i in range(10)])
        delta1 = test.GeneratorWienerProcessDelta(time[i], x)
        delta.append(delta1)
        value1 = test.GeneratorWienerProcessValues(delta1)
        value.append(value1)
    print(time[0])

    model = WienerModelWithPowerTrend(data,x)
    demonstateModelWithoutCov([0, 10], [0, 15], "time", "Z(t)", model,data)
    return

#Исследования №3
#Экспоненциальная модель с равномерной сеткой
def ResearchExpModel(x0):
    time, delta, value = [],[],[]
    # Хранение данных в структуре
    data = DataStorage(time, delta, value)
    x = x0
    print("истинные параметры:",  x)
    # создание  модели
    model = WienerModelWithExpTrend(data,x)
    #моделирования данных
    test = ModelingData(model)

    for i in range(5):
        time.append([i for i in range(20)])
        delta1 = test.GeneratorWienerProcessDelta(time[i], x)
        delta.append(delta1)
        value1 = test.GeneratorWienerProcessValues(delta1)
        value.append(value1)

    model = WienerModelWithExpTrend(data,x)
    demonstateModelWithoutCov([0, 10], [0, 15], "time", "Z(t)", model,data)
    return

#Исследования №4
#Степенная модель с равномерной сеткой
#проверка ковариат, на генерировать первый параметр, и по двум другим функциям оценить beta
#относительно отрисовки функции тренда для функции с ковариатами?
def ResearchPowerModelWithCovariate(x0, type):

    res_array = []
    for item in range(50):
        time, delta, value = [],[],[]
        # Хранение данных в структуре
        data = DataStorage(time, delta, value, [])

        x = x0
        print("истинные параметры:",  x)
        # создание  модели
        model = WienerModelWithPowerTrendWithCov(data,x, type)
        #моделирования данных
        test = ModelingData(model)
        c = [1,2,3]
        cov = []
        for k in range(3):
            for i in range(5):
                time.append([l for l in range(10)])
                delta1 = test.GeneratorWienerProcessDelta(time[i], x, c[k])
                delta.append(delta1)
                value1 = test.GeneratorWienerProcessValues(delta1)
                value.append(value1)
                cov.append(c[k])


        data.updateAll(time, delta, value, cov)
        model = WienerModelWithPowerTrendWithCov(data,  x, type)
        res = model.estimate_Parametrs(x)
        print(res)
        res_array.append(res)
    demonstateFuncSurvivalWithCov([0, 20], [0, 1], "time", "P(t)", model, data, res_array)


    return

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
    filename.close()
    return Ar

ResearchPowerModelWithCovariate([1,1,2,1],1)

"""data  =  DataStorage(input_Mat("DataOfPower2/data_time.txt"), input_Mat("DataOfPower2/data_delta.txt"), input_Mat("DataOfPower2/data_value.txt"))
model = WienerModel(data, 0.2, 1, 2)
x=[0.2,1, 2]
res = model.estimate_Parametrs(x)
model.updateModel(data, res)
print("оцененные параметры:", res)
Graph = Graphics([0,20], [0,100], "time", "Z(t)", data)
Graph.PaintValuesFunction([0, len(data.get_delta())])
Graph.PaintFunctionTrend(model, model.get_parameters())
Graph.PaintFunctionTrend(WienerModelWithPowerTrend(data,0.5,1,1.5), [0.5,1,1.5])
Graph.ShowGraphic()"""