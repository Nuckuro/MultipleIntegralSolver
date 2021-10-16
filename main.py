import numpy as np


class Function:
    def __init__(self, func: str):
        self.__func = func

    def __call__(self, identifiers: dict):
        return eval(self.__func, {'np': np}, identifiers)

    @classmethod
    def li(cls, *args):
        return list(map(cls, args))


def cartesian_product(*arrays):
    la = len(arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=np.result_type(*arrays))
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


class Integral:
    def __init__(self, variables,
                 conditions: list[Function], function: Function,
                 min_, max_):
        self.__variables = variables
        self.__conditions = conditions
        self.__function = function
        self.__min = np.array(min_, dtype=np.float32)
        self.__max = np.array(max_, dtype=np.float32)

    def grid(self, n):
        delta = (self.__max - self.__min) / n
        v = cartesian_product(*([np.arange(n, dtype=np.float32)] * len(self.__variables)))
        return (v + .5) * delta, delta

    def sum_up(self, n):
        x, delta = self.grid(n)
        var_dict = {self.__variables[i]: x[:, i] for i in range(len(self.__variables))}
        y = self.__function(var_dict) * np.product(delta)
        mask = np.all([c(var_dict) for c in self.__conditions], axis=0)
        v = y[mask].sum()
        return v


with open('input.txt') as f:
    fun, *rest, min_, max_ = (l for l in map(str.strip, f) if l)
    integral = Integral(variables=['x', 'y'],
                        conditions=Function.li(*rest),
                        function=Function(fun),
                        min_=[0, 10],
                        max_=[0, 10])
    print(integral.sum_up(10000))
