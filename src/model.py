from src.handler import AbstractHandler
from src.mediator import BaseComponent
from enum import Enum
from types import GeneratorType
import numpy as np
import time


class TrendType(Enum):
    LINEAR = 0
    EXPONENTIAL = 1

class Model(AbstractHandler, BaseComponent):

    @classmethod
    def _cast_to_symmetric_interval(cls, data, R):
        data_max = np.max(data)
        data_min = np.min(data)
        data_symmetric_interval = ((data - data_min) / (data_max - data_min) - 0.5) * 2 * R 
        return data_symmetric_interval


    @classmethod
    def _generate_noise(cls, N=1000, delta=1):
        for t in range(0, N, delta):
            yield np.random.uniform(0, 1)

    @classmethod
    def _generate_custom_noise(cls, N=1000, delta=1):
        MAX_INT = 2147483647
        CUSTOM_MULTIPLIER = 12312313
        CUSTOM_OFFSET = 23498259284592
        old_value = time.time()
        for t in range(0, N, delta):
            new_value = (CUSTOM_MULTIPLIER * old_value + CUSTOM_OFFSET) % MAX_INT
            old_value = new_value
            yield new_value
    
    def generate_noise(self, N, R, delta=1):
        noise = np.fromiter(self._generate_noise(N, delta), float)
        symm_interval_noise = self._cast_to_symmetric_interval(noise, R)
        return symm_interval_noise
    

    def generate_custom_noise(self, N, R, delta=1):
        noise = np.fromiter(self._generate_custom_noise(N, delta), float)
        symm_interval_noise = self._cast_to_symmetric_interval(noise, R)
        return symm_interval_noise





    @classmethod
    def _linear_func(cls, a, b, t):
        return -a *t + b
    
    @classmethod
    def _exponential_func(cls, a, b, t):
        return b * np.exp(-a * t)
    
    @classmethod    
    def _generate_linear(cls, a, b, N=1000, delta=1):
        for t in range(0, N, delta):
            yield cls._linear_func(a, b, t)

    @classmethod
    def _generate_exponential(cls, a, b, N=1000, delta=1):
        for t in range(0, N, delta):
            yield cls._exponential_func(a, b, t)

    def trend(self, type : TrendType, a, b, N=1000, delta=1):
        if type == TrendType.LINEAR:
            return np.fromiter(self._generate_linear(a, b, N, delta), float)
        elif type == TrendType.EXPONENTIAL:
            return np.fromiter(self._generate_exponential(a, b, N, delta), float)
    
        
    def handle(self, data):
        pass

