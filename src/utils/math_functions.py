from abc import ABC, abstractclassmethod
import numpy as np
from enum import Enum


class FuncType(Enum):
    LINEAR = 0,
    EXPONENTIAL = 1,
    # HARM = 2,
    POLY_HARM = 3


class AbsctractFunc(ABC):
    @abstractclassmethod
    def calculate(cls, *args, **kwargs):
        pass


class LinearFunc(AbsctractFunc):
    @classmethod
    def _linear_func(cls, a, b, t):
        return -a * t + b

    @classmethod
    def calculate(cls, a, b, N=1000, delta=1, *args, **kwargs):
        for t in range(0, N, delta):
            yield cls._linear_func(a, b, t)


class ExponentialFunc(AbsctractFunc):
    @classmethod
    def _exponential_func(cls, a, b, t):
        return b * np.exp(-a * t)

    @classmethod
    def calculate(cls, a, b, N=1000, delta=1, *args, **kwargs):
        for t in range(0, N, delta):
            yield cls._exponential_func(a, b, t)



# looks like poly harm 
# class HarmFunc(AbsctractFunc):
#     @classmethod
#     def _harm_func(cls, a, f0, dt, t):
#         return a * np.sin(2 * np.pi * f0 * dt * t)

#     @classmethod
#     def calculate(cls, a, f0, dt, N=1000, delta=1, *args, **kwargs):
#         for t in range(0, N, delta):
#             yield cls._harm_func(a, f0, dt, t)


class PolyHarmFunc(AbsctractFunc):
    @classmethod
    def _harm_func(cls, a, f0, dt, t):
        return a * np.sin(2 * np.pi * f0 * dt * t)

    @classmethod
    def _poly_harm_func(cls, ai : tuple, fi : tuple, dt, t):
        return sum(cls._harm_func(a, f0, dt, t) for a, f0 in zip(ai, fi))


    @classmethod
    def calculate(cls, ai : tuple, fi : tuple, dt, N = 1000, delta = 1, *args, **kwargs):
        for t in range(0, N, delta):
            yield cls._poly_harm_func(ai, fi, dt, t)

class FuncFactory:
    def getFunc(self, func_type: FuncType):
        match func_type:
            case FuncType.LINEAR:
                return LinearFunc
            case FuncType.EXPONENTIAL:
                return ExponentialFunc
            # case FuncType.HARM:
            #     return HarmFunc
            case FuncType.POLY_HARM:
                return PolyHarmFunc
