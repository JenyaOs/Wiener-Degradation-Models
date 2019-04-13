import numpy as np
from scipy import stats
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.special import beta
from Module.modeling import ModelingData
from Module.data import DataStorage
import pandas

#Вычисление мощности
#двусторонний критерий
def power2d(st_limit, st_crit, alpha):
    print("alpha:", alpha)
    # встроенная функция  import pandas
    s = pandas.Series(st_limit)

    m = 0
    for i in range(len(st_crit)):
        if(st_crit[i]>s.quantile(1-alpha/2) or st_crit[i]<s.quantile(alpha/2)):
            m += 1

    return m/len(st_crit)

#правосторонний критерий
def powerRight(st_limit, st_crit, alpha):
    print("alpha:", alpha)
    # встроенная функция  import pandas
    s = pandas.Series(st_limit)

    m = 0
    for i in range(len(st_crit)):
        if (st_crit[i] > s.quantile(1 - alpha)):
            m += 1

    return m / len(st_crit)

#левосторонний критерий
def powerLeft(st_limit, st_crit, alpha):
    print("alpha:", alpha)
    # встроенная функция  import pandas
    s = pandas.Series(st_limit)

    m = 0
    for i in range(len(st_crit)):
        if (st_crit[i] < s.quantile(alpha)):
            m += 1

    return m / len(st_crit)

#приведение выборки к нужному формату
def GetF(x_array, x, t, model, c = 0):
    F = []
    for j in range(len(x_array)):
        x_sort = (x_array[j])
        for i in range(len(x_sort)):
            m = model.function_M(t[j][i+1],x, c) -  model.function_M(t[j][i],x, c)
            s = x[0] * np.sqrt(model.function_trend(t[j][i+1],x, c) - model.function_trend(t[j][i],x, c))
            F.append(norm.cdf(x_sort[i], m, s))
    return np.sort(F)

#Колмогоров
class Kolmogorov_Test():

    def get_Name(self):
        return "Колмогоров"

    def get_Sk(self, F):
        n = len(F)
        ecdf = ECDF(F)
        DnP = np.zeros(n)
        DnM = np.zeros(n)
        for i in range(n):
            DnP[i] = (ecdf.y[i+1] - F[i])
            DnM[i] = (F[i] - ecdf.y[i])

        Dn = np.max([np.max(DnP), np.max(DnM)])
        Sk = (6* n * Dn + 1) / (6 * np.sqrt(n))
        return Sk

    def get_LimitDistribution(self, x):
        return stats.gamma.cdf(x, 6.472, 0.2620, 0.0589)

#Критерий Андерсона-Дарлинга
class AndersonDarling_Test():
    def get_Name(self):
        return "Андерсон-Дарлинг"

    def get_Sk(self, F):
        n = len(F)
        sum = 0
        for i in range(n):
            a = (((2*(i+1)-1)/(2*n)))
            b = (1-(2*(i+1)-1)/(2*n))
            sum += a * np.log(F[i] if F[i]>0 else 1)
            sum += b * np.log(1-(F[i] if F[i]!=1 else 0))
        Sk = -n - 2*sum
        return Sk

    def getBeta3(self, x, t0, t1, t2, t3,t4):
        part1 = (x-t4)/t3
        res = pow(t2,t0)/(t3*beta(t0,t1))
        res*= pow(part1,t0-1)
        res *= pow(1-part1, t1 - 1)
        res /= pow(1+(t2-1)*part1,t0+t1)
        return res

    def get_LimitDistribution(self, x):
        #бета третьего рода либо найти встроенную либо расписать
        ar = []
        for i in range(len(x)):
            ar.append(self.getBeta3(x,4.7262,4.6575,9.4958,2.717,0.0775))
        return ar

#Критерий Крамера-Мизеса-Смирнова
class CramerVonMisesSmirnov_Test():
    def get_Name(self):
        return "Крамер-Мизес-Смирнов"

    def get_Sk(self, F):
        n = len(F)
        sum = 0
        for i in range(n-1):
            sum +=  (F[i] - ((2*(i+1)-1)/(2*n)))**2
        Sk = 1.0/(12.0*n) + sum
        return Sk

    def getBeta3(self, x, t0, t1, t2, t3, t4):
        part1 = (x - t4) / t3
        res = pow(t2, t0) / (t3 * beta(t0, t1))
        res *= pow(part1, t0 - 1)
        res *= pow(1 - part1, t1 - 1)
        res /= pow(1 + (t2 - 1) * part1, t0 + t1)

        return res

    def get_LimitDistribution(self, x):
        # бета третьего рода либо найти встроенную либо расписать
        ar = []
        for i in range(len(x)):
            ar.append(self.getBeta3(x, 4.1153, 4.1748, 11.035, 0.5116, 0.009))
        return ar

#критерий Хи_2 Пирсона
class Xi_2():

    def __init__(self, _count_intervals=5):
        self.k = _count_intervals

    def get_Name(self):
        return "Хи_2-Пирсона"

    def get_Sk(self, F):
        border = [i/self.k for i in range(self.k+1)]
        P_teor = [1/self.k for i in range(self.k)]
        P_emp = np.zeros(self.k)
        for i in range(len(F)):
            j = 0
            while(j < self.k):
                if(border[j]<F[i]<=border[j+1]):
                    P_emp[j] += 1
                    break
                j+=1
        P_emp = P_emp/len(F)
        hi = 0
        for i in range(self.k):
            hi += ((P_emp[i] - P_teor[i]) ** 2) / P_teor[i]

        return hi*(len(F))

    def get_LimitDistribution(self, x):
        return stats.gamma.cdf(x, 1, 0, 2.)

#Проверка модели на адекватность x - оцененные параметры без коварит
def ToCheckAdequacyOfModel(x, model, data):
    time = data.get_time()
    cov = data.get_covariates()
    testData = ModelingData(model)

    #Генерируем N выборок аналогичных оцененной и каждый раз оцениваем
    for i in range(1):
        #print(i)
        dataStorage = DataStorage(time, [], [], cov)
        for j in range(len(time)):
            dataStorage.delta.append(testData.GeneratorWienerProcessDelta(time[j], x))
            dataStorage.value.append(testData.GeneratorWienerProcessValues(dataStorage.delta[j]))

        model.updateModel(dataStorage, x)
        res = model.estimate_Parametrs(x)
        #print(res)
        F = GetF(dataStorage.delta, res, dataStorage.time, model)
        ArraySk = (Kolmogorov_Test().get_Sk(F))
        ArraySk1 = (AndersonDarling_Test().get_Sk(F))
        ArraySk2 = (CramerVonMisesSmirnov_Test().get_Sk(F))
        ArraySk3 = (Xi_2(5).get_Sk(F))
    return ArraySk, ArraySk1, ArraySk2, ArraySk3
