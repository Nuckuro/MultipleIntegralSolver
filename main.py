import numpy as np
from tqdm import tqdm


class Function:
    def __init__(self, func: str):
        self.__func = func

    def __call__(self, identifiers: dict):
        return eval(self.__func, np.__dict__ | {'np': np}, identifiers)

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
        self.__min = np.array(min_, dtype=np.float16)
        self.__max = np.array(max_, dtype=np.float16)

    def grid(self, n):
        v = cartesian_product(*([np.linspace(begin, end, n, dtype=np.float16) for begin, end in
                                 zip(self.__min, self.__max)]))
        return v, (self.__max - self.__min) / n

    def sum_up(self, n=None):
        n = n or int(10000000 ** (1 / len(self.__variables)))
        x, delta = self.grid(n)
        var_dict = {self.__variables[i]: x[:, i] for i in range(len(self.__variables))}
        y = self.__function(var_dict) * np.product(delta)
        mask = np.all([c(var_dict) for c in self.__conditions], axis=0)
        v = y[mask].sum()
        return v


with open('input.txt') as f:
    vars_str, fun, *rest, min_, max_ = (l for l in map(str.strip, f) if l)
    integral = Integral(variables=[x.strip() for x in vars_str.split(',')],
                        conditions=Function.li(*rest),
                        function=Function(fun),
                        min_=[Function(x)({}) for x in min_.split(',')],
                        max_=[Function(x)({}) for x in max_.split(',')])
    print(integral.sum_up())
