from Module.data import DataStorage
from Module.modeling import  ModelingData
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


class Graphics(DataStorage):
    def __init__(self, _nameX, _nameY, _title, _data, _model):

        self.nameX = _nameX
        self.nameY = _nameY
        self.title = _title
        self.model = _model

        DataStorage.__init__(self, _data.time, _data.delta, _data.value, _data.covariates)


    def SaveGraphic(self, title):
        plt.savefig(title, format = 'png', dpi = 100)
        return

    def ShowGraphic(self):
        plt.legend()
        plt.show()
        return

    def PaintFunctionTrend(self, x, c=0):
        # отрисовка функции тренда
        time = DataStorage.get_time(self)[0]
        test1 = ModelingData(self.model)
        trend = test1.GenetatorTrend(time, x, c)
        string = 'x='+ str(c)
        plt.plot(time, trend,  label = string)
        return

    def PaintValuesFunction(self, slice):
        self.UpdateSetting(self.nameX, self.nameY, self.title)
        value = DataStorage.get_value(self)[slice[0]:slice[1]]
        time = DataStorage.get_time(self)[slice[0]:slice[1]]
        for i in range(len(value)):
            plt.plot(time[i], value[i], color='grey', linestyle='--')

        return

    def PaintFunctionSurvival(self, x, model, Zcr, c =0, colors="red", style="-"):
        x_array = []
        y_array = []
        t1  = 0
        flag = True
        while flag:
            t = t1
            m = model.function_M(t,x,c)
            sigma1 = x[0] * np.sqrt(model.function_trend(t, x, c))
            x_array.append(t)
            y_array.append(norm.cdf(Zcr, m, sigma1))
            if (y_array[t1] < 1e-4 or t1 > 1000000):
                flag = False
                plt.xlim([0, t])
            else:
                t1 += 1
        plt.plot(x_array, y_array, color=colors,  linestyle=style)
        return

    def UpdateSetting(self, _nameX, _nameY, _title):

        plt.xlabel(_nameX)
        plt.ylabel(_nameY)
        plt.title(_title)
        return


