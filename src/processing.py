from src.handler import AbstractHandler
from src.mediator import BaseComponent
import src.utils.statistics as stat
import numpy as np

class Processor(AbstractHandler, BaseComponent):
    def antiShift(self, data):
        return data - stat.mean(data)
    
    def antiSpike(self, data, N, R):
        outdata = np.zeros(N)
        for i in range(1, N-1):
            if abs(data[i]) > R:
                outdata[i] = (data[i-1] + data[i+1]) / 2
            else:
                outdata[i] = data[i]    
        outdata[0] = data[0]
        outdata[N-1] = data[N-1]
        return outdata

    def antiTrendLinear(self, data, N):
        return [data[i+1] - data[i] for i in range(N - 1)]
    
    def antiTrendNonLinear(self, data, N, W):
        return [1 / W * sum(data[n:n+W]) for n in range(N - W)]

    def antiNoise(self, data: list, N, M):
        return [1 / M * sum(data[:, i] )for i in range(N)]

    def handle(self, data):
        pass