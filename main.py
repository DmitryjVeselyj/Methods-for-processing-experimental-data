from src.mediator import Mediator
from src.inout import InOuter
from src.model import Model
from src.analysis import Analyzer
from src.processing import Processor
from matplotlib import pyplot as plt
from src.utils.random_generator import RandomGeneratorType, NormalGenerator, CustomGenerator, UniformGenerator
from src.utils.math_functions import FuncType
from src.utils.fourier_transform import RectangularWindow

import numpy as np

def plot_getFuncDatas(model : Model):
    linear_inc = model.getFuncData(FuncType.LINEAR, -1, 0)
    linear_dec = model.getFuncData(FuncType.LINEAR, 1, 0)
    exp_inc = model.getFuncData(FuncType.EXPONENTIAL, -0.007, 1)
    exp_dec = model.getFuncData(FuncType.EXPONENTIAL, 0.006, 1000)

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(linear_inc)
    ax[0, 1].plot(linear_dec)
    ax[1, 0].plot(exp_inc)
    ax[1, 1].plot(exp_dec)

    plt.show()


def plot_picewice(model: Model):
    linear_inc = model.getFuncData(FuncType.LINEAR, -1, 0)
    linear_dec = model.getFuncData(FuncType.LINEAR, 1, 1000)
    exp_dec = model.getFuncData(FuncType.EXPONENTIAL, 0.006, 1000)

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
    linear_inc = model.getFuncData(FuncType.LINEAR, -1, 0)
    model.shift(linear_inc, len(linear_inc), 500, 0, 300)
    plt.plot(linear_inc)
    plt.show()

def print_statistics(model : Model, analyzer : Analyzer):
    linear_inc = model.getFuncData(FuncType.LINEAR, -1, 0)
    print("Linear: ")
    print(analyzer.statistics(linear_inc))

    exp_dec = model.getFuncData(FuncType.EXPONENTIAL,  0.006, 1000)
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
        ax[i].plot(model.getFuncData(FuncType.POLY_HARM, ai=(100, ), fi =(f0, ), dt=0.001))

    plt.show()    
    
    ai = (100, 15, 20)
    fi = (33, 5, 170)
    plt.plot(model.getFuncData(FuncType.POLY_HARM, ai, fi, dt=0.001))
    plt.show()


def plot_hist(model : Model, analyzer : Analyzer):
    M = 100
    exp_dec = model.getFuncData(FuncType.EXPONENTIAL, 0.006, 1000)
    analyzer.hist(exp_dec, len(exp_dec), M)

    linear_inc = model.getFuncData(FuncType.LINEAR, -1, 0)
    analyzer.hist(linear_inc, len(linear_inc), M)

    noise_system = np.fromiter(NormalGenerator.generate(N=100000), float)
    analyzer.hist(noise_system, len(noise_system), M)

    noise_custom = np.fromiter(CustomGenerator.generate(N=100000), float)
    analyzer.hist(noise_custom, len(noise_custom), M)

    harm_15 = model.getFuncData(FuncType.POLY_HARM, ai=(100, ), fi =(15, ), dt=0.001)
    analyzer.hist(harm_15, len(harm_15), M)


def _plot_corr_res(values, L):
    plt.axhline(0, linewidth=2)
    plt.vlines(L, 0, values)
    plt.scatter(L, values)
    plt.show()


def plot_acf(model : Model, analyzer : Analyzer):
    noise_system = np.fromiter(NormalGenerator.generate(N=1000), float)
    L = range(40)
    values = [analyzer.acf(noise_system, len(noise_system), l, calc_cov=False) for l in L]
    _plot_corr_res(values, L)
    
    noise_custom = np.fromiter(CustomGenerator.generate(N=1000), float)
    L = range(40)
    values = [analyzer.acf(noise_custom, len(noise_custom), l, calc_cov=False) for l in L]
    _plot_corr_res(values, L)

    harm_15 = model.getFuncData(FuncType.POLY_HARM, ai=(100, ), fi =(15, ), dt=0.001)
    L = range(1000)
    values = [analyzer.acf(harm_15, len(harm_15), l, calc_cov=False) for l in L]
    _plot_corr_res(values, L)

def plot_ccf(model : Model, analyzer : Analyzer):
    noise_system_x = np.fromiter(NormalGenerator.generate(N=1000), float)
    noise_system_y = np.fromiter(NormalGenerator.generate(N=1000), float)
    L = range(40)
    values = [analyzer.ccf(noise_system_x, noise_system_y, l) for l in L]
    _plot_corr_res(values, L)

    noise_custom_x = np.fromiter(CustomGenerator.generate(N=1000), float)
    noise_custom_y = np.fromiter(CustomGenerator.generate(N=1000), float)
    L = range(40)
    values = [analyzer.ccf(noise_custom_x, noise_custom_y, l) for l in L]
    _plot_corr_res(values, L)


    harm_15 = model.getFuncData(FuncType.POLY_HARM, ai=(100, ), fi =(15, ), dt=0.001)
    harm_516 = model.getFuncData(FuncType.POLY_HARM, ai=(100, ), fi =(516, ), dt=0.001)
    L = range(1000)
    values = [analyzer.ccf(harm_15, harm_516, l) for l in L]
    _plot_corr_res(values, L)


def plot_fourier(model : Model, analyzer : Analyzer):
    for i, f0 in enumerate(range(15, 516, 100)):
        harm = model.getFuncData(FuncType.POLY_HARM, ai=(100, ), fi =(f0, ), dt=0.001)
        re, im, amp = analyzer.fourier(harm, len(harm))
        analyzer.spectr_fourier(amp, 0.001)

    ai = (100, 15, 20)
    fi = (33, 5, 170)
    harm = model.getFuncData(FuncType.POLY_HARM, ai=ai, fi =fi, dt=0.001)
    re, im, amp = analyzer.fourier(harm, len(harm))
    analyzer.spectr_fourier(amp, 0.001)

def plot_fourier_window(model : Model, analyzer : Analyzer):
    harm = model.getFuncData(FuncType.POLY_HARM, ai=(100, ), fi =(15, ), dt=0.001, N = 1024)
    for L in (24, 124, 224):
        window = RectangularWindow(len(harm) - L)
        re, im, amp = analyzer.fourier(harm, len(harm), window)
        analyzer.spectr_fourier(amp, 0.001)


    ai = (100, 15, 20)
    fi = (33, 5, 170)
    harm = model.getFuncData(FuncType.POLY_HARM, ai=ai, fi =fi, dt=0.001, N = 1024)
    for L in (24, 124, 224):
        window = RectangularWindow(len(harm) - L)
        re, im, amp = analyzer.fourier(harm, len(harm), window)
        analyzer.spectr_fourier(amp, 0.001)
    


def plot_fourier_file(model : Model, analyzer : Analyzer, fname):
    data = np.fromfile('./data/' + fname, dtype=np.float32)
    re, im, amp = analyzer.fourier(data, len(data))
    analyzer.spectr_fourier(amp, 0.005)

def plot_add_multiple(model : Model, analyzer : Analyzer):
    x1_1 = model.getFuncData(FuncType.LINEAR, a=-0.3, b = 20)
    x1_2 = model.getFuncData(FuncType.POLY_HARM, ai=(5, ), fi=(50, ), dt=0.001)

    plt.plot(model.addArrays(x1_1, x1_2))
    plt.show()
    plt.plot(model.multArrays(x1_1, x1_2))
    plt.show()
  
    x2_1 = model.getFuncData(FuncType.EXPONENTIAL, a=-0.05, b=10,N=100)
    x2_2 = model.generate_noise(N=100, R=10)
    plt.plot(model.addArrays(x2_1, x2_2))
    plt.show()
    plt.plot(model.multArrays(x2_1, x2_2))
    plt.show()


def plot_anti_evth(model : Model, processor : Processor):
    data = model.getFuncData(FuncType.LINEAR, a = 0, b = 3)
    data += np.fromiter(CustomGenerator.generate(N=1000), float)

    fig, ax = plt.subplots(2)
    ax[0].plot(data)
    ax[1].plot(processor.antiShift(data))
    plt.show()


    data_noise_spikes = np.fromiter(UniformGenerator.generate(N=1000), float)
    data_noise_spikes = model.spikes(1000, 10, 0, 4, RandomGeneratorType.UNIFORM_GENERATOR, data_noise_spikes)

    data_harm_spikes = model.getFuncData(FuncType.POLY_HARM, ai=(8, ), fi=(50, ), dt=0.001)
    data_harm_spikes = model.spikes(1000, 10, 20, 20, RandomGeneratorType.UNIFORM_GENERATOR, data_harm_spikes)

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(data_noise_spikes)
    ax[0, 1].plot(processor.antiSpike(data_noise_spikes, len(data_noise_spikes), 1))

    ax[1, 0].plot(data_harm_spikes)
    ax[1, 1].plot(processor.antiSpike(data_harm_spikes, len(data_harm_spikes), 20))

    plt.show()


if __name__ == "__main__":
    model = Model()
    analyzer = Analyzer()
    processor = Processor()
    # plot_getFuncDatas(model)
    # plot_picewice(model)
    # plot_shift(model)
    # plot_spikes(model)  
    # plot_noise(model)
    # print_statistics(model, analyzer)
    # plot_harm(model)
    # plot_hist(model, analyzer)
    # plot_acf(model, analyzer)
    # plot_ccf(model, analyzer)
    # plot_fourier(model, analyzer)
    # plot_fourier_window(model, analyzer)
    # plot_fourier_file(model, analyzer, 'pgp_dt0005.dat')
    # plot_add_multiple(model, analyzer)
    plot_anti_evth(model, processor)

