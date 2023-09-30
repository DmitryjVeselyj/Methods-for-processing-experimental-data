from abc import ABC, abstractclassmethod
import numpy as np
from enum import Enum


class TrendFuncType(Enum):
    LINEAR = 0
    EXPONENTIAL = 1

class TrendFunc(ABC):
    @abstractclassmethod
    def calculate(cls, a, b, N=1000, delta=1):
        pass

class LinearFunc(TrendFunc):
    @classmethod
    def _linear_func(cls, a, b, t):
        return -a *t + b

    @classmethod
    def calculate(cls, a, b, N=1000, delta=1):
        for t in range(0, N, delta):
            yield cls._linear_func(a, b, t)

class ExponentialFunc(TrendFunc):
    @classmethod
    def _exponential_func(cls, a, b, t):
        return b * np.exp(-a * t)
    
    @classmethod
    def calculate(cls, a, b, N=1000, delta=1):
        for t in range(0, N, delta):
            yield cls._exponential_func(a, b, t)


class TrendFuncFactory:
    def getFunc(self, TrendType: TrendFuncType):
        match TrendType:
            case TrendType.LINEAR:
                return LinearFunc
            case TrendType.EXPONENTIAL:
                return ExponentialFunc