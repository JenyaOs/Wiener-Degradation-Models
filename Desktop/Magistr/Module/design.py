from Module.data import DataStorage
import numpy as np
from Module.modeling import  ModelingData
import numpy as np
import math
from scipy.optimize import minimize


def FisherInformationMatrix(model, c, weight, _t, type):
    '''
    :param model:
    :param c:
    :param weight:
    :param _t:
    :param type:
    :return: result
    '''
    x = model.get_parameters()
    t = model.get_time()
    s = x[0]
    m = x[1]
    rr = int(len(t)*weight)
    t = t[0:len(t)*rr]

    def I11():
        n = len(t)
        k = len(t[0])
        return 2*n*k/pow(s,2)

    def I12():
        return 0

    def I13():
        n = len(t)
        k = len(t[0]) - 1

        p = [[model.function_trend(_t[j+1],x ,c)-model.function_trend(_t[j],x ,c) \
                                                                                    for j in range(k)]\
                                                                                    for i in range(n)]

        der_p = [[model.function_DerGamma_trend(_t[j + 1],x ,c)-model.function_DerGamma_trend(_t[j],x ,c) \
                                                                                    for j in range(k)]\
                                                                                    for i in range(n)]

        result = sum([sum([der_p[i][j]/p[i][j]
                                                                                    for j in range(k)])\
                                                                                    for i in range(n)])
        return result/s

    def I14():
        n = len(t)
        k = len(t[0]) - 1


        p = [[model.function_trend(_t[j + 1],x ,c) - model.function_trend(_t[j],x ,c) \
                                                                                    for j in range(k)]\
                                                                                    for i in range(n)]
        der_p = [[model.function_DerBeta_trend(_t[j + 1],x ,c) - model.function_DerBeta_trend(_t[j],x ,c) \
                                                                                    for j in range(k)]\
                                                                                    for i in range(n)]
        result = sum([sum([der_p[i][j]/p[i][j]
                                                                                    for j in range(k)])\
                                                                                    for i in range(n)])
        return result/s

    def I22():
        n = len(t)
        k = len(t[0]) - 1

        result = sum([sum([model.function_trend(_t[j + 1], x, c) - \
                           model.function_trend(_t[j], x, c) \
                           for j in range(k)]) \
                           for i in range(n)])

        return result/pow(s,2)

    def I23():
        n = len(t)
        k = len(t[0]) - 1


        result = sum([sum([model.function_DerGamma_trend(_t[j + 1], x, c) - \
                           model.function_DerGamma_trend(_t[j], x, c) \
                           for j in range(k)]) \
                           for i in range(n)])

        return result * m / pow(s,2)

    def I24():
        n = len(t)
        k = len(t[0]) - 1

        result = sum([sum([model.function_DerBeta_trend(_t[j + 1], x, c) - \
                           model.function_DerBeta_trend(_t[j], x, c) \
                           for j in range(k)]) \
                           for i in range(n)])

        return result * m / pow(s,2)

    def I33():
        n = len(t)
        k = len(t[0]) - 1
        m2 = pow(m, 2)
        s2 = pow(s, 2)

        p = [[model.function_trend(_t[j + 1], x, c) - model.function_trend(_t[j], x, c) \
              for j in range(k)] \
             for i in range(n)]

        p2 = [[pow(p[i][j], 2) \
               for j in range(k)] \
              for i in range(n)]

        p4 = [[pow(p[i][j], 4) \
               for j in range(k)] \
              for i in range(n)]

        der2_p = [[model.function_DerGamma2_trend(_t[j + 1], x, c) - model.function_DerGamma2_trend(_t[j], x, c) \
                   for j in range(k)] \
                  for i in range(n)]

        der_p2 = [
            [pow(model.function_DerGamma_trend(_t[j + 1], x, c) - model.function_DerGamma_trend(_t[j], x, c), 2) \
             for j in range(k)] \
            for i in range(n)]

        result = sum([sum([0.5 * ((der2_p[i][j] * p[i][j] - der_p2[i][j]) / p2[i][j]) + \
                           (m2 * der2_p[i][j] - (s2 * p[i][j] + m2 * p2[i][j]) * \
                            (der2_p[i][j] * p2[i][j] - 2 * p[i][j] * der_p2[i][j]) / p4[i][j]) / (2 * s2) \
                           for j in range(k)]) \
                           for i in range(n)])

        return result

    def I34():
        n = len(t)
        k = len(t[0]) - 1
        m2 = pow(m, 2)
        s2 = pow(s, 2)

        p = [[model.function_trend(_t[j + 1], x, c) - model.function_trend(_t[j], x, c) \
              for j in range(k)] \
             for i in range(n)]

        p2 = [[pow(p[i][j], 2) \
               for j in range(k)] \
              for i in range(n)]

        p4 = [[pow(p[i][j], 4) \
               for j in range(k)] \
              for i in range(n)]

        der2_p = [[model.function_DerGammaBeta_trend(_t[j + 1], x, c) - model.function_DerGammaBeta_trend(_t[j], x, c) \
                     for j in range(k)] \
                     for i in range(n)]


        der_p2 = [[((model.function_DerGamma_trend(_t[j + 1], x, c) - model.function_DerGamma_trend(_t[j], x, c))* \
                    (model.function_DerBeta_trend(_t[j + 1], x, c) - model.function_DerBeta_trend(_t[j], x, c))) \
                    for j in range(k)] for i in range(n)]


        result = sum([sum([0.5 * ((der2_p[i][j] * p[i][j] - der_p2[i][j]) / p2[i][j]) + \
                           (m2 * der2_p[i][j] - (s2 * p[i][j] + m2 * p2[i][j]) * \
                            (der2_p[i][j] * p2[i][j] - 2 * p[i][j] * der_p2[i][j]) / p4[i][j]) / (2 * s2) \
                           for j in range(k)]) \
                           for i in range(n)])

        return result

    def I44():
        n = len(t)
        k = len(t[0]) - 1
        m2 = pow(m, 2)
        s2 = pow(s, 2)

        p = [[model.function_trend(_t[j + 1], x, c) - model.function_trend(_t[j], x, c) \
              for j in range(k)] \
             for i in range(n)]

        p2 = [[pow(p[i][j], 2) \
               for j in range(k)] \
              for i in range(n)]

        p4 = [[pow(p[i][j], 4) \
               for j in range(k)] \
              for i in range(n)]

        der2_p = [[model.function_DerBeta2_trend(_t[j + 1], x, c) - model.function_DerBeta2_trend(_t[j], x, c) \
                   for j in range(k)] \
                  for i in range(n)]

        der_p2 = [[pow(model.function_DerBeta_trend(_t[j + 1], x, c) - model.function_DerBeta_trend(_t[j], x, c), 2) \
                   for j in range(k)] \
                  for i in range(n)]

        result = sum([sum([0.5 * ((der2_p[i][j] * p[i][j] - der_p2[i][j]) / p2[i][j]) + \
                           (m2 * der2_p[i][j] - (s2 * p[i][j] + m2 * p2[i][j]) * \
                            (der2_p[i][j] * p2[i][j] - 2 * p[i][j] * der_p2[i][j]) / p4[i][j]) / (2 * s2) \
                           for j in range(k)]) \
                           for i in range(n)])

        return result

    if(type == 1):
        if(rr != 0):
            return np.array([[I11(), I12()],[I12(), I22()]])
        else:
            return np.array([[0,0],[0,0]])
    if(type == 2):
        if(rr!= 0):
            return np.array([[I11(), I12(), I13()], [I12(), I22(), I23()], [I13(), I23(), I33()]])
        else:
            return np.array([[0,0,0],[0,0,0],[0,0,0]])
    if(type == 3):
        if(rr!=0):
            return np.array([[I11(), I12(), I14()],[I12(), I22(), I24()],[I14(), I24(), I44()]])
        else:
            return np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    if(type == 4):
        if(rr != 0):
            return np.array([[I11(), I12(), I13(), I14()],[I12(), I22(), I23(), I24()], [I13(), I23(), I33(), I34()],[I14(), I24(), I34(), I44()]])
        else:
            return np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    return "Error"

class desing1():
    # конструктор
    def __init__(self, _plan, _weight, _model, _type):
        self.plan = _plan
        self.weight = _weight
        self.model = _model
        self.type =_type
        self.delta = 0.01

    # получение нормированной матрицы
    def M_ksi(self, p, w, t):
        m_ksi = 0
        for i in range(len(p)):
            infMatrix = FisherInformationMatrix(self.model,p[i], w[i], t,self.type)
            m_ksi += np.dot((w[i]), infMatrix)
        return m_ksi

    # получение массива оптимальных точек плана
    def X_u_D(self, plan, weight, t):

        return -np.log(np.linalg.det(self.M_ksi(plan, weight, t)))

    # получение массива оптимальных точек плана
    def X_u_A(self, plan, weight, t):
        return (np.trace(np.linalg.inv(self.M_ksi(plan, weight, t))))

    def fine_weight(self, weight):
        fine = 0
        if (sum(weight)>1):
            fine = 10000
        if (sum(weight)<1):
            fine = 10000
        return fine

    # получение массива оптимальных точек весов
    def X_p_D(self, weight, plan, t):
        return -np.log(np.linalg.det(self.M_ksi(plan, weight, t))) + self.fine_weight(weight)

    def X_p_A(self, weight, plan, t):
        return (np.trace(np.linalg.inv(self.M_ksi(plan, weight, t)))) + self.fine_weight(weight)

    def fine_times(self, times):
        fine = 0
        for i in range(len(times)-1):
            if(times[i+1]-times[i]< 0.5):
                fine += 1000
        if(times[0]< 0):
            fine += 10000
        return fine

    # получение массива оптимальных точек весов
    def X_t_D(self, t, weight, plan):
        print(t)
        print(-np.log(np.linalg.det(self.M_ksi(plan, weight, t))))
        return -np.log(np.linalg.det(self.M_ksi(plan, weight, t))) + self.fine_times(t)

    def X_t_A(self, t,  weight, plan):
        return (np.trace(np.linalg.inv(self.M_ksi(plan, weight, t)))) + self.fine_times(t)

    # разница планов, получение невязки
    def get_z(self, p_init_new, u_init_new, u_init, p_init):
        p_diffs = p_init - p_init_new
        u_diffs = u_init - u_init_new
        s = 0
        for i in range(len(u_diffs)):
            s += np.linalg.norm(u_diffs[i]) ** 2
        for i in range(len(p_diffs)):
            s += np.linalg.norm(p_diffs[i]) ** 2
        return s

        # ограничение на сумму весов

    def eq_constraint(self, x):
        return np.sum(x) - 1

        # ограничение на веса

    def ineq_constraint(self, x):
        return x

    #функция ограничение стоимости
    def fine(self, times):
        z = 0
        #print(times)
        for i in range(len(times)-1):
            if(times[0]<0):
                z+=1000000
            if(times[i+1]-times[i] <= 0.5):
                z+=100000

        return z


#построение D - оптимального плана
def designDirect_D(_model, _plan0, _typeForMatrix,  _bounds):
    weight = [0.3,0.7]
    optDesign = desing1(_plan0, weight, _model, _typeForMatrix)

    z = optDesign.delta * 2
    p_init = optDesign.weight
    u_init = optDesign.plan
    times =  _model.get_time()[0]
    print(times)
    #print(p_init)
    print("Начальное значение функционала X: {0}".format(optDesign.X_u_D(u_init, p_init, times)))

    while (z > optDesign.delta ) :

        optimized_u = (minimize(optDesign.X_u_D, u_init, method='SLSQP',  bounds=[_bounds]*len(p_init), args=(p_init, times)))
        optDesign.plan = optimized_u.x


        optimized_p = ( minimize(optDesign.X_p_D, optDesign.weight, method='SLSQP',
                                    args=(u_init, times), constraints=[{'type': 'ineq', 'fun': optDesign.ineq_constraint},
                                                                  {'type': 'eq', 'fun': optDesign.eq_constraint}]))
        optDesign.weight = optimized_p.x

        z = optDesign.get_z(optDesign.weight, optDesign.plan, u_init, p_init)
        u_init = optDesign.plan
        p_init = optDesign.weight


    print("Спектор план:\n{0}".format(u_init.T))
    print("Веса плана:\n{0}".format(p_init))
    print("Значение функционала X: {0}".format(optDesign.X_u_D(u_init, p_init,times)))
    print(optDesign.X_u_D(u_init, p_init, times))

    times =  _model.get_time()[0]
    times = (minimize(optDesign.X_t_D, times, bounds=[[0, 50]] * len(times), method='SLSQP',
                      args=(optDesign.plan, optDesign.weight)))
    times = times.x
    print("Моменты измерения деградационного показателя: ", times)
    print("Значение функционала X: {0}".format(optDesign.X_u_D(u_init, p_init, times)))

    return optDesign.X_u_D(u_init, p_init, times)

def designDirect_A(_model, _plan0, _typeForMatrix, _bounds):
    weight = [1 / len(_plan0) for i in range(len(_plan0))]
    optDesign = desing1(_plan0, weight, _model, _typeForMatrix)

    z = optDesign.delta * 2
    p_init = optDesign.weight
    u_init = optDesign.plan
    print(p_init)
    print("Начальное значение функционала X: {0}".format(optDesign.X_u_A(u_init, p_init)))

    while (z > optDesign.delta):
        print(p_init)
        optimized_u = (
            minimize(optDesign.X_u_A, u_init, method='SLSQP', bounds=[_bounds] * len(p_init), args=(p_init)))
        optDesign.plan = optimized_u.x

        optimized_p = (minimize(optDesign.X_p_A, optDesign.weight, method='SLSQP',
                                args=(optDesign.plan), bounds=[[0, 1]] * len(p_init)))
        optDesign.weight = optimized_p.x

        z = optDesign.get_z(optDesign.weight, optDesign.plan, u_init, p_init)
        u_init = optDesign.plan
        p_init = optDesign.weight

    print("Спектор план:\n{0}".format(u_init.T))
    print("Веса плана:\n{0}".format(p_init))
    print("Значение функционала X: {0}".format(optDesign.X_u_A(u_init, p_init)))
    """times = (minimize(optDesign.X_time_D, times, bounds=[[0, 200]]*len(times), method='SLSQP', args=(optDesign.plan, optDesign.weight)))
    times = times.x
    print("Моменты измерения деградационного показателя: ", times)
    print("Значение функционала X: {0}".format(optDesign.X_u_D(u_init, p_init, times)))"""

