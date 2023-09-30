from src.mediator import Mediator
from src.inout import InOuter
from src.model import Model
from src.model import TrendType
from src.analysis import Analyzer
from src.processing import Processor
from matplotlib import pyplot as plt
import numpy as np

def plot_trends(model : Model):
    linear_inc = model.trend(TrendType.LINEAR, -1, 0)
    linear_dec = model.trend(TrendType.LINEAR, 1, 0)
    exp_inc = model.trend(TrendType.EXPONENTIAL, -0.007, 1)
    exp_dec = model.trend(TrendType.EXPONENTIAL, 0.006, 1000)

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(linear_inc)
    ax[0, 1].plot(linear_dec)
    ax[1, 0].plot(exp_inc)
    ax[1, 1].plot(exp_dec)

    plt.show()


def plot_picewice(model: Model):
    linear_inc = model.trend(TrendType.LINEAR, -1, 0)
    linear_dec = model.trend(TrendType.LINEAR, 1, 1000)
    exp_dec = model.trend(TrendType.EXPONENTIAL, 0.006, 1000)

    plt.plot(np.concatenate((exp_dec, linear_inc, linear_dec), axis=None))
    plt.show()


def plot_noise(model : Model):
    noise_system = model.generate_noise(1000, 2)
    noise_custom = model.generate_custom_noise(1000, 2)
    fig, ax = plt.subplots(2)
    ax[0].plot(list(noise_system))
    ax[1].plot(list(noise_custom))
    plt.show()

if __name__ == "__main__":
    model = Model()
    plot_trends(model)
    plot_picewice(model)
    plot_noise(model)
