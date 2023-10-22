from src.handler import AbstractHandler
from src.mediator import BaseComponent
import src.utils.statistics as stat
import numpy as np
from matplotlib import pyplot as plt

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
    

    def hist(self, data, N, M):
        offset = min(data)
        rangem = max(data) - min(data)

        bins = np.zeros(M)
        xi = np.linspace(min(data), max(data), M)
        iwidth = max([xi[i+1] - xi[i] for i in range(len(xi)-1)])
        for d in data:
            bin_index = int((d - offset) / iwidth)
            bins[bin_index] += 1
        
        
        # plt.bar(xi, bins, width=iwidth)
        plt.hist(data, bins=M)
        plt.show()
        return xi, bins, iwidth

    def acf(self, data, N, L, calc_cov = True):
        if calc_cov:
            return stat.covariance(data, data, L)
        return stat.autocorrelation(data, L)
    
    def ccf(self, x, y, L):
        return stat.covariance(x , y, L)
