from src.handler import AbstractHandler
from src.mediator import BaseComponent
import src.utils.statistics as stat
import numpy as np

class Analyzer(AbstractHandler, BaseComponent):
    def handle(self, data):
        pass

    def statistics(self, data, N=None, type=None):
        return {"mean":     stat.mean(data),
                "min" :     stat.min(data),
                "max" :     stat.max(data),
                "var" :     stat.var(data),
                "std" :     stat.std(data),
                "skew":     stat.skew(data),
                "kurtosis": stat.kurtosis(data),
                "mse" :     stat.mean_squared_error(data)
                }
     
    
    def stationary(self, data, N, M):
        chunks = np.array_split(data, M)
        mean_vals = np.array([stat.mean(chunk) for chunk in chunks])
        std_vals = np.array([stat.std(chunk) for chunk in chunks])

        # переписать через if или ещё как-нибудь, чтобы не считать лишнего
        mean_rel_measure = [abs(mean_vals[i] - mean_vals[j]) for i in range(M) for j in range(i+1, M)]
        std_rel_measure  = [abs(std_vals[i] - std_vals[j])   for i in range(M) for j in range(i+1, M)]
        range_interval = 0.05 * (stat.max(data) - stat.min(data))
        return all(map(lambda elem: elem < range_interval, mean_rel_measure)) and \
               all(map(lambda elem: elem < range_interval, std_rel_measure))
