
#хранение данных
class DataStorage():
    # конструктор
    #_time - время измерения деградационного показателя.
    #_delta - приращение деградационного показателя.
    #_value - значение деградационного показателя в момент измерения.
    #_covariates - массив ковариат, соответствующих каждому измерению деградационного показателя.
    def __init__(self, _time, _delta, _value, _covariates=[]):
        self.time = _time
        self.delta = _delta
        self.value = _value
        self.covariates=_covariates

    # деструктор
    def __del__(self):
        self.time
        self.delta
        self.value
        self.covariates

    #функции, позволяющие получить значения элементов класса
    def get_time(self):
        return self.time

    def get_delta(self):
        return self.delta

    def get_value(self):
        return self.value

    def get_covariates(self):
        return self.covariates

    # функции, позволяющие изменить значения элементов класса
    def set_time(self, _time):
        self.time = _time
        return

    def set_delta(self,_delta):
        self.delta = _delta
        return

    def set_value(self,_value):
        self.value = _value
        return

    def set_covariates(self,_covariates):
        self.covariates= _covariates
        return

    # функция, позволяющие обновить значения элементов класса
    def updateAll(self,  _time, _delta, _value, _covariates=[]):
        self.set_time(_time)
        self.set_delta(_delta)
        self.set_value(_value)
        self.set_covariates(_covariates)