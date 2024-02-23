from abc import ABC, abstractclassmethod
from collections.abc import Generator
from math import modf
from enum import Enum
import numpy as np
import time


class RandomGeneratorType(Enum):
    UNIFORM_GENERATOR = 0,
    CUSTOM_GENERATOR = 1,
    NORMAL_GENERATOR = 2


class RandomGenerator(ABC):
    @abstractclassmethod
    def generate(cls, a=0, b=1, N=1000, delta=1) -> Generator[float]:
        pass


class UniformGenerator(RandomGenerator):
    @classmethod
    def generate(cls, a=0, b=1, N=1000, delta=1):
        for t in range(0, N, delta):
            yield np.random.uniform(a, b)


class CustomGenerator(RandomGenerator):
    @classmethod
    def generate(cls, a=0, b=1, N=1000, delta=1):
        CUSTOM_MULTIPLIER = 3324.32493
        old_value = time.time()
        for t in range(0, N, delta):
            new_value = CUSTOM_MULTIPLIER * modf(old_value)[0]
            new_value = modf(new_value)[0] * (b - a) + a
            old_value = new_value
            yield new_value


class NormalGenerator(RandomGenerator):
    @classmethod
    def generate(cls, a=0, b=1, N=1000, delta=1):
        for t in range(0, N, delta):
            yield np.random.normal(a, b)


class RandomGeneratorFactory:
    def getGenerator(self, type: RandomGeneratorType) -> RandomGenerator:
        match type:
            case RandomGeneratorType.UNIFORM_GENERATOR:
                return UniformGenerator
            case RandomGeneratorType.CUSTOM_GENERATOR:
                return CustomGenerator
            case RandomGeneratorType.NORMAL_GENERATOR:
                return NormalGenerator
