from src.handler import AbstractHandler
from src.mediator import BaseComponent
from src.utils.random_generator import RandomGeneratorType, RandomGeneratorFactory
from src.utils.trend_functions import TrendFuncType, TrendFuncFactory
import numpy as np

class Model(AbstractHandler, BaseComponent):

#====================THIRD TASK===========================#
    def shift(self, data, N, C, N1, N2):
        if N2 > N:
            raise ValueError("N2 must be <= N")
        data[N1:N2]+=C
        

    def spikes(self, N, M, R, Rs, generatorType : RandomGeneratorType):
        generator = RandomGeneratorFactory().getGenerator(generatorType)
        data = np.zeros(N)
        positions = np.fromiter(generator.generate(a=0, b=N, N=M), int)

        positive_spikes_len = M // 2
        positive_spikes = np.fromiter(generator.generate(a=R - Rs, b=R + Rs, N=positive_spikes_len), float)
        negative_spikes = np.fromiter(generator.generate(a=-R - Rs, b=-R + Rs, N=M - positive_spikes_len), float)
        data[positions[0:positive_spikes_len]] = positive_spikes
        data[positions[positive_spikes_len:]] = negative_spikes

        return data


#====================SECOND TASK===========================#
    @classmethod
    def _cast_to_interval(cls, data, R, shift_constant = 0.5, multiply_constant = 2, add_constant = 0):
        data_max = np.max(data)
        data_min = np.min(data)
        data_interval = ((data - data_min) / (data_max - data_min) - shift_constant) * multiply_constant * R  + add_constant
        return data_interval
    
    def generate_noise(self, N, R, delta=1):
        generator = RandomGeneratorFactory().getGenerator(RandomGeneratorType.UNIFORM_GENERATOR)
        noise = np.fromiter(generator.generate(N=N, delta=delta), float)
        symm_interval_noise = self._cast_to_interval(noise, R)
        return symm_interval_noise
    

    def generate_custom_noise(self, N, R, delta=1):
        generator = RandomGeneratorFactory().getGenerator(RandomGeneratorType.CUSTOM_GENERATOR)
        noise = np.fromiter(generator.generate(N=N, delta=delta), float)
        symm_interval_noise = self._cast_to_interval(noise, R)
        return symm_interval_noise




#====================FIRST TASK===========================#
    def trend(self, type : TrendFuncType, a, b, N=1000, delta=1):
        trend_func = TrendFuncFactory().getFunc(type)
        return np.fromiter(trend_func.calculate(a, b, N, delta), float)
    
        
    def handle(self, data):
        pass

