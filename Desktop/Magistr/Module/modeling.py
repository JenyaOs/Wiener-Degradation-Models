import numpy as np
import random

#моделирование данных
class ModelingData():
    def __init__(self, _model):
        self.model = _model

    # генерирование моментов измерения деградационного показателя на интервале с заданным шагом
    # _len - промежуток измерения _countIntervals - количество интервалов
    def GeneratorTime(self, _len, _countIntervals):
        # step - шаг, между моментами измерения
        step = _len / _countIntervals
        # arrayT - инициализация результирующего массив
        arrayT = np.zeros(_len + 1)
        # генерирования значений массива
        for i in range(_len + 1):
            arrayT[i] = (i * step)
        # возвращения результата
        return arrayT

    # генерирование значений функции тренда
    def GenetatorTrend(self, time, x, c):
        # arrayT - результирующий массив
        arrayT = np.zeros(len(time))
        # генерация значений
        for i in range(len(time)):

            arrayT[i] = self.model.function_M(time[i], x, c)
        return arrayT

    # генерирование значений приращений винеровского деградационного процесса
    def GeneratorWienerProcessDelta(self, time, x, c=0):
        # arrayD - результирующий массив
        arrayD = np.zeros(len(time) - 1)
        # генерация значений
        for i in range(len(time) - 1):
            # приращение между моментам времени
            delta = self.model.function_trend(time[i + 1], x, c) - self.model.function_trend(time[i], x, c)
            # сдвиг
            muZ = self.model.get_mu() * delta
            # масштаб
            sigmaZ = np.sqrt(delta) * self.model.get_sigma()
            # значение по нормальному распределению с заданными параметрами
            arrayD[i] = (random.normalvariate(muZ, sigmaZ))

        return arrayD

    # генерирование значений в соответствии с винеровской деградационной моделью
    def GeneratorWienerProcessValues(self, delta):
        # arrayV - результирующий массив
        arrayV = np.zeros(len(delta) + 1)
        arrayV[0] = 0
        arrayV[1] = delta[0]
        for i in range(len(delta) - 1):
            arrayV[i + 2] = (arrayV[i + 1] + delta[i + 1])
        return arrayV
