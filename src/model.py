from src.handler import AbstractHandler
from src.mediator import BaseComponent
from src.utils.random_generator import RandomGeneratorType, RandomGeneratorFactory
from src.utils.math_functions import FuncType, FuncFactory
from functools import reduce
import numpy as np
import yfinance as yf

class Model(AbstractHandler, BaseComponent):

    def shift(self, data, N, C, N1, N2):
        if N2 > N:
            raise ValueError("N2 must be <= N")
        data[N1:N2]+=C
        

    def spikes(self, N, M, R, Rs, generatorType : RandomGeneratorType, data = None, hardcode=1):
        generator = RandomGeneratorFactory().getGenerator(generatorType)
        if data is None:
            data = np.zeros(N)
        positions = np.fromiter(generator.generate(a=0, b=N, N=M), int)

        positive_spikes_len = M // 2
        positive_spikes = np.fromiter(generator.generate(a=R - Rs, b=R + Rs, N=positive_spikes_len), float)
        negative_spikes = np.fromiter(generator.generate(a=-hardcode * R - Rs, b=-hardcode * R + Rs, N=M - positive_spikes_len), float)
        data[positions[0:positive_spikes_len]] = positive_spikes
        data[positions[positive_spikes_len:]] = negative_spikes

        return data

    @classmethod
    def _cast_to_interval(cls, data, R, shift_constant = 0.5, multiply_constant = 2, add_constant = 0):
        data_max = np.max(data)
        data_min = np.min(data)
        data_interval = ((data - data_min) / (data_max - data_min) - shift_constant) * multiply_constant * R  + add_constant
        return data_interval
    
    def generate_noise(self, N, R, delta=1):
        generator = RandomGeneratorFactory().getGenerator(RandomGeneratorType.NORMAL_GENERATOR)
        noise = np.fromiter(generator.generate(N=N, delta=delta), float)
        symm_interval_noise = self._cast_to_interval(noise, R)
        return symm_interval_noise
    

    def generate_custom_noise(self, N, R, delta=1):
        generator = RandomGeneratorFactory().getGenerator(RandomGeneratorType.CUSTOM_GENERATOR)
        noise = np.fromiter(generator.generate(N=N, delta=delta), float)
        symm_interval_noise = self._cast_to_interval(noise, R)
        return symm_interval_noise

    # TODO return func object
    def getFuncData(self, type : FuncType, *args, **kwargs):
        func = FuncFactory().getFunc(type) 
        return np.fromiter(func.calculate(*args, **kwargs), float)


    def handle(self, data):
        pass
    
    def addArrays(self, *args):
        return reduce(np.add, args)
    
    def multArrays(self, *args):
        return reduce(np.multiply, args)
    
    def convolModel(self, x, N, h, M):
        return [sum(x[i -m] * h[m] for m in range(M) if i -m >= 0 and i - m < N) for i in range(N + M - 1)]# [int(M/2) + 1: N- int(M/2)]






#-----------------------------------------------kursovaya----------------------------------------

    def get_stocks_data(self, ticker, period, interval):
        return yf.Ticker(ticker).history(period=period, interval=interval)
    

    def ito_process(self, t, a, b, c, d):
        return a + t * np.exp(b*t) + np.sin(c*t) * np.random.random() * d