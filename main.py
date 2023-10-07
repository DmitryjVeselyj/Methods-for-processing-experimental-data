from src.mediator import Mediator
from src.inout import InOuter
from src.model import Model
from src.analysis import Analyzer
from src.processing import Processor
from matplotlib import pyplot as plt
from src.utils.random_generator import RandomGeneratorType, NormalGenerator, CustomGenerator
from src.utils.math_functions import FuncType
import numpy as np

def plot_trends(model : Model):
    linear_inc = model.trend(FuncType.LINEAR, -1, 0)
    linear_dec = model.trend(FuncType.LINEAR, 1, 0)
    exp_inc = model.trend(FuncType.EXPONENTIAL, -0.007, 1)
    exp_dec = model.trend(FuncType.EXPONENTIAL, 0.006, 1000)

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(linear_inc)
    ax[0, 1].plot(linear_dec)
    ax[1, 0].plot(exp_inc)
    ax[1, 1].plot(exp_dec)

    plt.show()


def plot_picewice(model: Model):
    linear_inc = model.trend(FuncType.LINEAR, -1, 0)
    linear_dec = model.trend(FuncType.LINEAR, 1, 1000)
    exp_dec = model.trend(FuncType.EXPONENTIAL, 0.006, 1000)

    plt.plot(np.concatenate((exp_dec, linear_inc, linear_dec), axis=None))
    plt.show()


def plot_noise(model : Model):
    noise_system = model.generate_noise(1000, 2)
    noise_custom = model.generate_custom_noise(1000, 2)
    fig, ax = plt.subplots(2)
    ax[0].plot(noise_system)
    ax[1].plot(noise_custom)
    plt.show()


def plot_spikes(model : Model):
    spikes = model.spikes(100, 10, 500, 300, RandomGeneratorType.UNIFORM_GENERATOR)
    plt.plot(range(0, len(spikes)), spikes)
    plt.show()

def plot_shift(model : Model):
    linear_inc = model.trend(FuncType.LINEAR, -1, 0)
    model.shift(linear_inc, len(linear_inc), 500, 0, 300)
    plt.plot(linear_inc)
    plt.show()

def print_statistics(model : Model, analyzer : Analyzer):
    linear_inc = model.trend(FuncType.LINEAR, -1, 0)
    print("Linear: ")
    print(analyzer.statistics(linear_inc))

    exp_dec = model.trend(FuncType.EXPONENTIAL,  0.006, 1000)
    print("Exponential: ")
    print(analyzer.statistics(exp_dec))

    noise_system = np.fromiter(NormalGenerator.generate(N=100000), float)
    print("Noise Normal")
    print(analyzer.statistics(noise_system))
    print("Stationary: ", analyzer.stationary(noise_system, len(noise_system), 50))

    noise_custom = np.fromiter(CustomGenerator.generate(N=100000), float)
    print("Noise Custom")
    print(analyzer.statistics(noise_custom))
    print("Stationary: ", analyzer.stationary(noise_custom, len(noise_custom), 50))


def plot_harm(model : Model):
    fig, ax = plt.subplots(6)
    for i, f0 in enumerate(range(15, 516, 100)):
        ax[i].plot(model.trend(FuncType.POLY_HARM, ai=(100, ), fi =(f0, ), dt=0.001))

    plt.show()    
    
    ai = (100, 15, 20)
    fi = (33, 5, 170)
    plt.plot(model.trend(FuncType.POLY_HARM, ai, fi, dt=0.001))
    plt.show()

if __name__ == "__main__":
    model = Model()
    analyzer = Analyzer()
    # plot_trends(model)
    # plot_picewice(model)
    # plot_shift(model)
    # plot_spikes(model)  
    # plot_noise(model)
    # print_statistics(model, analyzer)
    plot_harm(model)


