from typing import Any
import numpy as np
from abc import ABC, abstractmethod

class AbstractWindowFunc(ABC):
    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

class NoWindowFunc(AbstractWindowFunc):
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return 1

class RectangularWindow(AbstractWindowFunc):
    def __init__(self, delim) -> None:
        self._delim = delim

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        assert len(args) == 1
        k = args[0]
        return 1 if k <= self._delim  else 0


def calc_real_part(data, N, n, window):
    return 1 / N * sum(x * np.cos(2 * np.pi * n * k / N) * window(k) for k, x in enumerate(data))

def calc_imag_part(data, N, n, window):
    return 1 / N * sum(x * np.sin(2 * np.pi * n * k / N) * window(k) for k, x in enumerate(data))

def calc_ampliture(real_part, imaginary_part):
    return np.sqrt(real_part ** 2 + imaginary_part ** 2)