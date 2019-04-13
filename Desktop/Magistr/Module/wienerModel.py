from Module.data import DataStorage
import numpy as np
from scipy.optimize import minimize
import math


# Wiener degradation model with linear trend
class WienerModel(DataStorage):

        def __init__(self, data, x, type=0):
            '''
            :param data: Data of Degradation Process consist of time, increments and value experiment.
            :param x:    Array included models parameters such as sigma(scale), mu(drift)
            :param type: Type is covariate function, which included in some special forms of WienerModel
            '''
            self.sigma = x[0]
            self.mu = x[1]
            DataStorage.__init__(self, data.time, data.delta, data.value, data.covariates)

        # function allows to update all main parameters of this model
        def updateModel(self, data, x):
            '''
            :param data: Data of Degradation Process consist of new time, increments and value experiment.
            :param x:   New array included models parameters such as sigma, mu
            '''
            self.sigma = x[0]
            self.mu = x[1]
            DataStorage.updateAll(self, data.time, data.delta, data.value)

        # function allows to get some information of this models
        def get_mu(self):
            return self.mu

        def get_sigma(self):
            return self.sigma

        def get_gamma(self):
            return 1

        def get_beta(self):
            return 0

        # function allows to get all information of this models
        def get_parameters(self):
            return [self.sigma, self.mu]

        # fucntion of trend shows the distribution of degradation data
        def function_trend(self, t, x=[], c=0):
            return t

        # function of M help to demonstrate function trend
        def function_M(self, t, x, c = 0):
            return x[1] * self.function_trend(t)

        # first derivative of function trend on gamma parameters
        def function_DerGamma_trend(self,t=0, x=0, c = 0):
            return 0

        # second derivative of function trend on gamma parameters
        def function_DerGamma2_trend(self, t=0, x=0, c=0):
            return 0

        # second derivative of function trend on gamma and beta parameters
        def function_DerGammaBeta_trend(self, t, x, c=0):
            return 0

        # first derivative of function trend on beta parameters
        def function_DerBeta_trend(self, t, x, c):
            return 0

        # second derivative of function trend on gamma and beta parameters
        def function_DerBeta2_trend(self, t, x, c):
            return 0

        # analytic estimation of parameters sigma and mu for Wiener degradation model with linear function trend
        def estimate_Parametrs(self, x=[]):
            return [self.estimate_sigma(), self.estimate_mu()]

        # analytic estimation parameters sigma
        def estimate_sigma(self):
            delta = DataStorage.get_delta(self)
            time = DataStorage.get_time(self)
            sigma = 0
            n = len(delta)
            m = self.estimate_mu()
            for i in range(n):
                sigma1 = 0
                for j in range(len(time[i]) - 1):
                    delta_t = self.function_trend(time[i][j + 1]) - self.function_trend(time[i][j])
                    cur1 = (delta[i][j] - delta_t * m)**2
                    cur1 /= delta_t
                    sigma1 += cur1
                sigma += sigma1

            delete_sum = 0
            for i in range(n):
                delete_sum += len(time[i])-1

            sigma /= delete_sum
            sigma = np.sqrt(sigma)
            return sigma

        # analytic estimation parameters mu
        def estimate_mu(self):
            delta = DataStorage.get_delta(self)
            time = DataStorage.get_time(self)

            delt_Z = []
            time_Z = []
            n = len(delta)

            for i in range(n):
                delt_Z.append(np.sum(np.array([delta[i][j] for j in range(len(time[i])-1)])))
                time_Z.append(np.sum(np.array([time[i][j+1] - time[i][j] for j in range(len(time[i])-1)])))

            m = np.sum(delt_Z) / np.sum(time_Z)
            return m

        # log-likelihood function for MLE estimation
        def LNf(self, x):
            delta = DataStorage.get_delta(self)
            time = DataStorage.get_time(self)
            n = len(delta)
            sumk = 0
            for i in range(n):
                sumk += len(time[i]) - 1
            c1 = (sumk / 2) * np.log(2 * np.pi)
            c2 = sumk * np.log(x[0])
            s1 = 0
            p1 = 0
            for i in range(n):
                for j in range(len(time[i]) - 1):
                     m1 = (self.function_trend(time[i][j + 1]) - self.function_trend(time[i][j]))
                     m = x[1] * m1
                     if m <= 0 or x[0] <= 0:
                         return 1000000000000

                     s1 += ((delta[i][j] - m) ** 2) / m1
                     p1 += np.log(m1)
            s1 /= 2 * (x[0] ** 2)
            p1 /= 2
            sum = c1 + c2 + p1 + s1
            return sum

# Wiener degradation model with covariate based on linear trend
class WienerModelWithCovariate(DataStorage):


    def __init__(self, data, x, type = 1):
        '''
        :param data: Data of Degradation Process consist of time, increments and value experiment.
        :param x:    Array included models parameters such as sigma(scale), mu(drift), beta(parameter of covariate function)
        :param type: Type is covariate function, which included in some special forms of WienerModel
        '''
        self.sigma = x[0]
        self.mu = x[1]
        self.beta = x[2]
        self.type_cov = type
        DataStorage.__init__(self, data.time, data.delta, data.value, data.covariates)

    # function allows to update all main parameters of this model
    def updateModel(self, data, x):
        self.sigma = x[0]
        self.mu = x[1]
        self.beta = x[2]
        DataStorage.updateAll(self, data.time, data.delta, data.value)
        return

    # function allows to get some information of this models
    def get_mu(self):
        return self.mu

    def get_sigma(self):
        return self.sigma

    def get_gamma(self):
        return 1

    def get_beta(self):
        return self.beta

    # function allows to get all information of this models
    def get_parameters(self):
        return [self.sigma, self.mu, self.beta]

    # function of M help to demonstrate function trend
    def function_M(self, t, x, c=0):
        return x[1] * self.function_trend(t, x, c)

    # fucntion of covariate shows impact the distribution of degradation data
    def function_covariate(self,beta,c):
        f = 1
        if (self.type_cov == 1):
            f = np.exp(beta * c)
        if (self.type_cov == 2):
            f = np.exp(beta * np.log(c))
        if (self.type_cov == 3):
            f = np.exp(beta / c)
        return f

    # fucntion of trend shows the distribution of degradation data
    def function_trend(self, t, x, c=0):
        return t/self.function_covariate(x[2], c)

    # first derivative of function trend on gamma parameters
    def function_DerGamma_trend(self, t, x, c=0):
        return 0

    # second derivative of function trend on gamma parameters
    def function_DerGamma2_trend(self, t, x, c=0):
        return 0

    # second derivative of function trend on gamma and beta parameters
    def function_DerGammaBeta_trend(self, t, x, c=0):
        return 0

    # first derivative of function trend on beta parameters
    def function_DerBeta_trend(self, t, x, c):
        if (self.type_cov == 1):
            f = -c * t / np.exp( c * x[2])
        if (self.type_cov == 2):
            f = - np.log(c) * t / np.exp(np.log(c) * x[2])
        if (self.type_cov == 3):
            f = - (1 / c) * t / np.exp(x[2] / c)
        return f

    # second derivative of function trend on beta parameters
    def function_DerBeta2_trend(self, t, x, c):
        if (self.type_cov == 1):
            f = pow(c, 2) * t / np.exp( c * x[2])
        if (self.type_cov == 2):
            f = pow(np.log(c), 2) * t / np.exp(np.log(c) * x[2])
        if (self.type_cov == 3):
            f = pow((1 / c), 2) * t / np.exp( x[2] / c)
        return f

    # MLE estimation of parameters sigma and mu for Wiener degradation model with linear function trend
    def estimate_Parametrs(self, x0):
        x = x0
        res = minimize(self.LNf, x, method="nelder-mead")
        res = res.x
        return res

    # log-likelihood function
    def LNf(self, x):
        delta = DataStorage.get_delta(self)
        time = DataStorage.get_time(self)
        covariates = DataStorage.get_covariates(self)

        k = len(covariates)

        n = len(delta)
        sumk = 0
        for i in range(n):
            sumk += len(time[i])-1
        c1 = (sumk / 2) * np.log(2 * np.pi)
        c2 = sumk * np.log(x[0])
        s1 = 0
        p1 = 0


        for i in range(n):
           for j in range(len(time[i]) - 1):
                m1 = (self.function_trend(time[i][j + 1], x, covariates[i]) - self.function_trend(time[i][j], x, covariates[i]))
                m = x[1] * m1
                if m <= 0 or x[0] <= 0:
                    return 1000000000000

                s1 += ((delta[i][j] - m) ** 2) / m1
                p1 += np.log(m1)

        s1 /= 2 * (x[0] ** 2)
        p1 /= 2
        sum = c1 + c2 + p1 + s1
        return sum


#винеровская деградационная модель со степенной функцией тренда
class WienerModelWithPowerTrend(DataStorage):

    def __init__(self, data, x, type=0):
        '''
        :param data: Data of Degradation Process consist of time, increments and value experiment.
        :param x:    Array included models parameters such as sigma(scale), mu(drift), gamma(parameter of trend function),beta(parameter of covariate function)
        :param type: Type is covariate function, which included in some special forms of WienerModel
        '''

        self.sigma = x[0]
        self.mu = x[1]
        self.gamma = x[2]
        DataStorage.__init__(self, data.time, data.delta, data.value, data.covariates)

    # function allows to update all main parameters of this model
    def updateModel(self, data, x):
        '''
        :param data: Data of Degradation Process consist of new time, increments and value experiment.
        :param x:   New array included models parameters such as sigma, mu
        '''
        self.sigma = x[0]
        self.mu = x[1]
        self.gamma = x[2]
        DataStorage.updateAll(self, data.time, data.delta, data.value)


    #функции, позволяющая получить значения элементов класса
    def get_mu(self):
        return self.mu

    def get_sigma(self):
        return self.sigma

    def get_gamma(self):
        return self.gamma

    def get_beta(self):
        return 0

    # функция, позволяющая получить все значения элементов класса
    def get_parameters(self):
        return [self.sigma,self.mu,self.gamma]

    # функция М позволяющая отоборазить тренд функции.
    def function_M(self, t, x, c = 0):
        return x[1]*self.function_trend(t, x)

    #первая производная функции тренда по гамма
    def function_trend(self, t, x, c = 0):
        return math.pow(t,x[2])

    # первая производная функции тренда по гамма
    def function_DerGamma_trend(self, t, x, c=0):
        return np.log(t)*math.pow(t, x[2])

    # вторая производная функции тренда по гамма
    def function_DerGamma2_trend(self, t, x, c=0):
        return math.pow(np.log(t),2)*math.pow(t, x[2])

    # вторая производная функции тренда по гамма и по бета
    def function_DerGammaBeta_trend(self, t, x, c=0):
        return 0

    # первая производная функции тренда по бета
    def function_DerBeta_trend(self, t, x, c):
        return 0

    # вторая производная функции тренда по бета
    def function_DerBeta2_trend(self, t, x, c):
        return 0

    # первая производная функции от ковариат по бета
    def derivateBeta_M(self, beta=0, c=0):
        return 0

    # вторая производная функции от ковариат по бета
    def derivateBeta2_M(self, beta=0, c=0):
        return 0

    # оценивание параметров масштаба, сдвига и параметра функции тренда для  винеровская деградационной модели со степенной функцией тренда
    def estimate_Parametrs(self, x0):
        x = x0
        res = minimize(self.LNf, x, method="nelder-mead")
        res = res.x
        return res

    # функция правдоподобия
    def LNf(self, x):
        delta = DataStorage.get_delta(self)
        time = DataStorage.get_time(self)

        n = len(delta)
        sumk = 0
        for i in range(n):
            sumk += len(time[i])-1
        c1 = (sumk / 2) * np.log(2 * np.pi)
        c2 = sumk * np.log(x[0])
        s1 = 0
        p1 = 0
        for i in range(n):
           for j in range(len(time[i]) - 1):
                m1 = (self.function_trend(time[i][j + 1], x) - self.function_trend(time[i][j], x))
                m = x[1] * m1
                if m <= 0 or x[0] <= 0:
                    return 1000000000000

                s1 += ((delta[i][j] - m) ** 2) / m1
                p1 += np.log(m1)
        s1 /= 2 * (x[0] ** 2)
        p1 /= 2
        sum = c1 + c2 + p1 + s1
        return sum


#винеровская деградационная модель со экспоненциальной функцией тренда
class WienerModelWithExpTrend(DataStorage):
    # конструктор
    """_data - структура, хранящая исследуемые данные
     _sigma - параметр модели, обозначающая мастштаб
     _mu - параметр модели, обозначающая сдвиг
     _gamma - параметр отвечающий за тренд функции, по умолчанию для линейной модели всегда 1
     _beta - параметр отвечающий за тренд функции от коварит, по умолчанию для линейной модели всегда 0"""
    def __init__(self, _data, _x, _type=0):
        self.sigma = _x[0]
        self.mu = _x[1]
        self.gamma = _x[2]
        DataStorage.__init__(self, _data.time, _data.delta, _data.value, _data.covariates)


    # функция, позволяющие обновить значения элементов класса
    def updateModel(self, data, x):
        self.sigma = x[0]
        self.mu = x[1]
        self.gamma = x[2]
        DataStorage.updateAll(self, data.time, data.delta, data.value)
        return

    #функции, позволяющая получить значения элементов класса
    def get_mu(self):
        return self.mu

    def get_sigma(self):
        return self.sigma

    def get_gamma(self):
        return self.gamma

    def get_beta(self):
        return 0

    # функция, позволяющая получить все значения элементов класса
    def get_parameters(self):
        return [self.sigma,self.mu,self.gamma]

    # функция М позволяющая отоборазить тренд функции.
    def function_M(self, t, x, c= 0):
        return x[1]*self.function_trend(t, x)

    # функция тренда
    def function_trend(self, t, x, c = 0):
        return np.exp(t/x[2])-1

    # первая производная функции тренда по гамма
    def function_DerGamma_trend(self, t,x,c=0):
        return - t*np.exp(t/x[2])/pow(x[2],2)

    # вторая производная функции тренда по гамма
    def function_DerGamma2_trend(self, t,x,c=0):
        return - (-pow(t,2)*np.exp(t/x[2]) + 2*x[2]*t**np.exp(t/x[2]))/pow(x[2],4)

    # вторая производная функции тренда по гамма и по бета
    def function_DerGammaBeta_trend(self, t, x, c=0):
        return 0

    # первая производная функции тренда по бета
    def function_DerBeta_trend(self, t, x, c):
        return 0

    # вторая производная функции тренда по бета
    def function_DerBeta2_trend(self, t, x, c):
        return 0

    # первая производная функции от ковариат по бета
    def derivateBeta_M(self, beta=0, c=0):
        return 0

    # вторая производная функции от ковариат по бета
    def derivateBeta2_M(self, beta=0, c=0):
        return 0

    # оценивание параметров масштаба, сдвига и параметра функции тренда для  винеровская деградационной модели с экспоненциальной функцией тренда
    def estimate_Parametrs(self, x0):
        x = x0
        res = minimize(self.LNf, x, method="nelder-mead")
        res = res.x
        return res

    # функция правдоподобия
    def LNf(self, x):
        delta = DataStorage.get_delta(self)
        time = DataStorage.get_time(self)
        n = len(delta)
        sumk = 0
        for i in range(n):
            sumk += len(time[i])
        c1 = (sumk / 2) * np.log(2 * np.pi)
        c2 = sumk * np.log(x[0])
        s1 = 0
        p1 = 0
        for i in range(n):
           for j in range(len(time[i]) - 1):
                m1 = (self.function_trend(time[i][j + 1], x) - self.function_trend(time[i][j],  x))
                m = x[1] * m1
                if m <= 0 or x[0] <= 0:
                    return 1000000000000

                s1 += ((delta[i][j] - m) ** 2) / m1
                p1 += np.log(m1)
        s1 /= 2 * (x[0] ** 2)
        p1 /= 2
        sum = c1 + c2 + p1 + s1
        return sum


#винеровская деградационная модель со степенной функцией тренда и ковариатами
class WienerModelWithPowerTrendWithCov(DataStorage):
    # конструктор
    """ _data - структура, хранящая исследуемые данные
     _sigma - параметр модели, обозначающая мастштаб
     _mu - параметр модели, обозначающая сдвиг
     _gamma - параметр отвечающий за тренд функции, по умолчанию для линейной модели всегда 1
     _beta - параметр отвечающий за тренд функции от коварит, по умолчанию для линейной модели всегда 0"""
    def __init__(self, _data, _x, _type):
        self.sigma = _x[0]
        self.mu = _x[1]
        self.gamma = _x[2]
        self.beta = _x[3]

        self.type_cov = _type

        DataStorage.__init__(self, _data.time, _data.delta, _data.value, _data.covariates)

    # функция, позволяющие обновить значения элементов класса
    def updateModel(self, data, x):
        self.sigma = x[0]
        self.mu = x[1]
        self.gamma = x[2]
        self.beta = x[3]
        DataStorage.updateAll(self, data.time, data.delta, data.value, data.covariates)
        return

    #функции, позволяющая получить значения элементов класса
    def get_mu(self):
        return self.mu

    def get_sigma(self):
        return self.sigma

    def get_gamma(self):
        return self.gamma

    def get_beta(self):
        return self.beta

    def get_covariates(self):
        return self.covariates

    # функция, позволяющая получить все значения элементов класса
    def get_parameters(self):
        return [self.sigma,self.mu, self.gamma,self.beta]

    def function_covariate(self,beta,c):
        f = 1

        if (self.type_cov == 1):
            f = np.exp(beta * c)
        if (self.type_cov == 2):
            f = np.exp(beta * np.log(c))
        if (self.type_cov == 3):
            f = np.exp(beta / c)

        return f

    # функция М позволяющая отоборазить тренд функции.
    def function_M(self, t, x, c):
        return x[1]*self.function_trend(t, x, c)

    #  функции тренда
    def function_trend(self, t, x, c):
        return math.pow(t/self.function_covariate(x[3],c),x[2])

    # первая производная функции тренда по гамма
    def function_DerGamma_trend(self, t, x, c=0):
         return np.log(t/self.function_covariate(x[3],c))*math.pow(t/self.function_covariate(x[3],c),x[2])

    # вторая производная функции тренда по гамма
    def function_DerGamma2_trend(self, t, x, c=0):
        return math.pow(np.log(t/self.function_covariate(x[3],c)),2)*math.pow(t/self.function_covariate(x[3],c),x[2])

    # вторая производная функции тренда по гамма и по бета
    def function_DerGammaBeta_trend(self, t, x, c=0):
        if (self.type_cov == 1):
            f =  c * pow(t/ np.exp(x[3] * c), x[2])*(x[2]*x[3]*c - x[2]*np.log(t) - 1)
        if (self.type_cov == 2):
            f =  np.log(c)*pow(t/ np.exp(x[3] * np.log(c)), x[2])*(x[2]*x[3]*np.log(c)- x[2]*np.log(t)-1)
        if (self.type_cov == 3):
            f = (1/c)* pow(t / np.exp(x[3] * (1/c)), x[2]) * (x[2] * x[3] * (1/c) - x[2] * np.log(t) - 1)
        return f

    # первая производная функции тренда по бета
    def function_DerBeta_trend(self, t, x, c):
        if (self.type_cov == 1):
            f = -x[2] * c * pow(t,x[2]) / np.exp(x[3] * c * x[2])  #f = (t/np.exp(beta * c))^gamma / #-gamma*c*(t/np.exp(b*c))**gamma = -[c*((t/np.exp(b*c))**gamma) + gamma*np.log(t/np.exp(b*c)*(t/np.exp(b*c))**gamma)]
        if (self.type_cov == 2):
            f = -x[2] * np.log(c) * pow(t,x[2]) / np.exp(x[3] * np.log(c) * x[2])
        if (self.type_cov == 3):
            f = -x[2] * (1 / c)  * pow(t,x[2]) / np.exp(x[3]  * x[2] / c)

        return f

    # вторая производная функции тренда по бета
    def function_DerBeta2_trend(self, t, x, c):
        if (self.type_cov == 1):
            f = pow(x[2] * c,2) * pow(t, x[2]) / np.exp(x[3] * c * x[2])
        if (self.type_cov == 2):
            f = pow(x[2] * np.log(c),2) * pow(t, x[2]) / np.exp(x[3] * np.log(c) * x[2])
        if (self.type_cov == 3):
            f = pow(x[2] * (1 / c),2) * pow(t, x[2]) / np.exp(x[3] * x[2] / c)
        return f

    # оценивание параметров масштаба, сдвига, параметра функции тренда и параметра функции от ковариат для винеровская деградационной модели со степенной функцией тренда
    def estimate_Parametrs(self, x0):
        x = x0
        res = minimize(self.LNf, x, method="nelder-mead")
        res = res.x
        return res

    # функция правдоподобия
    def LNf(self, x):
        delta = DataStorage.get_delta(self)
        time = DataStorage.get_time(self)
        covariates = DataStorage.get_covariates(self)

        k = len(covariates)

        n = len(delta)
        sumk = 0
        for i in range(n):
            sumk += len(time[i])-1
        c1 = (sumk / 2) * np.log(2 * np.pi)
        c2 = sumk * np.log(x[0])
        s1 = 0
        p1 = 0


        for i in range(n):
           for j in range(len(time[i]) - 1):
                m1 = (self.function_trend(time[i][j + 1], x, covariates[i]) - self.function_trend(time[i][j], x, covariates[i]))
                m = x[1] * m1
                if m <= 0 or x[0] <= 0:
                    return 1000000000000

                s1 += ((delta[i][j] - m) ** 2) / m1
                p1 += np.log(m1)

        s1 /= 2 * (x[0] ** 2)
        p1 /= 2
        sum = c1 + c2 + p1 + s1
        return sum
