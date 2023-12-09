from src.mediator import Mediator
from src.inout import InOuter
from src.model import Model
from src.analysis import Analyzer
from src.processing import Processor
from matplotlib import pyplot as plt
from src.utils.random_generator import RandomGeneratorType, NormalGenerator, CustomGenerator, UniformGenerator
from src.utils.math_functions import FuncType
from src.utils.fourier_transform import RectangularWindow
from src.utils.statistics import std, var


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
    data = model.addArrays(data, np.fromiter(CustomGenerator.generate(N=1000), float))

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

def plot_another_anti(model : Model, processor : Processor):
    data_lin = model.getFuncData(FuncType.LINEAR, a=-0.3, b = 20)
    data_harm = model.getFuncData(FuncType.POLY_HARM, ai=(10, ), fi=(5, ), dt=0.001)
    data = model.addArrays(data_lin, data_harm)

    fig, ax = plt.subplots(2)
    ax[0].plot(data)
    ax[1].plot(processor.antiTrendLinear(data, len(data)))
    plt.show()


    data_nonlin = model.getFuncData(FuncType.EXPONENTIAL, 0.006, 1000)
    ai = (100, 15, 20)
    fi = (33, 5, 170)
    data_polyharm = model.getFuncData(FuncType.POLY_HARM, ai=ai, fi =fi, dt=0.001, N = 1000)
    data = model.addArrays(data_nonlin, data_polyharm)
    
    fig, ax = plt.subplots(2)
    ax[0].plot(data)
    ax[1].plot(processor.antiTrendNonLinear(data, len(data), 10))
    plt.show()

    fig, ax = plt.subplots(4)
    for i, M in enumerate([1, 10, 100, 10000]):
        data_noise = np.array([np.array(model.generate_noise(1000, 30)) for _ in range(M)])
        # data_noise = np.array([np.fromiter(NormalGenerator.generate(b=2, N=1000), float) for _ in range(M)])
        anti_data_noise = processor.antiNoise(data_noise, len(data_noise[0]), M)
        ax[i].plot(anti_data_noise)
        ax[i].title.set_text(str(M) + ', ' + str(std(anti_data_noise)))
    plt.show()
    
    # VAR = D( [X1 + X2 + X3 + X4] / N) = 1 / n**2 * n * D(X)
    # STD  = 1/ sqrt(N) * std(D)


    # std_arr = []
    # for M in range(1, 1000, 10):
    #     data_noise = np.array([np.array(model.generate_noise(1000, 30)) for _ in range(M)])
    #     # data_noise = np.array([np.fromiter(NormalGenerator.generate(b=2, N=1000), float) for _ in range(M)])
    #     anti_data_noise = processor.antiNoise(data_noise, len(data_noise[0]), M)
    #     std_arr.append(std(anti_data_noise))

    # plt.plot(np.arange(1, 1000, 10), std_arr)
    # plt.title('std зависимость')
    # plt.show()
    plt.plot([1 / np.sqrt(M) for M in range(1,1000, 10)])
    plt.show()

    d_noise = list(model.generate_noise(1000, 30))
    data_harm = model.getFuncData(FuncType.POLY_HARM, ai=(10, ), fi=(5, ), dt=0.001)
    data = model.addArrays(d_noise, data_harm)

    fig, ax = plt.subplots(2)
    ax[0].plot(data)
    ax[1].plot(processor.antiTrendLinear(data, len(data)))
    plt.show()

def plot_cardio(model : Model, analyzer : Analyzer, processor : Processor):
    harm = model.getFuncData(FuncType.POLY_HARM, ai=(1, ), fi =(7, ), dt=0.005, N = 1000)
    exp_low = model.getFuncData(FuncType.EXPONENTIAL, a=30*0.005, b=1, dt=0.005)
    h = model.multArrays(harm, exp_low)
    h = h / max(h) * 120
    x = np.zeros(1000)
    x[(range(200, 1000, 200))] = np.array([ np.random.random() * (0.2) + 0.9 for _ in range(4)])
    fig, ax = plt.subplots(3)
    ax[0].plot(h)
    ax[1].plot(x)
    ax[2].plot(model.convolModel(x, len(x), h, 200))
    plt.show()

    x = model.spikes(1000, 4, 1, 0.1, RandomGeneratorType.CUSTOM_GENERATOR)
    fig, ax = plt.subplots(3)
    ax[0].plot(h)
    ax[1].plot(x)
    ax[2].plot(model.convolModel(x, len(x), h, 200))
    plt.show()
    
def plot_filter_gp(model : Model, analyzer : Analyzer, processor : Processor):
    fc = 50
    dt = 0.002
    m = 64
    fc1 = 35
    fc2 = 75

    lpf = processor.reflect_lpf(processor.lpf(fc, dt, m))
    hpf = processor.hpf(fc, dt, m)
    bpf = processor.bpf(fc1, fc2, dt, m)
    bsf = processor.bsf(fc1, fc2, dt, m)

    fig, ax = plt.subplots(4)
    ax[0].plot(lpf)
    ax[0].set_title('lpf')

    ax[1].plot(hpf)
    ax[1].set_title('hpf')

    ax[2].plot(bpf)
    ax[2].set_title('bpf')

    ax[3].plot(bsf)
    ax[3].set_title('bsf')

    plt.show()

    lpf_amp = analyzer.transfer(analyzer.fourier(lpf, len(lpf))[2])
    hpf_amp = analyzer.transfer(analyzer.fourier(hpf, len(hpf))[2])
    bpf_amp = analyzer.transfer(analyzer.fourier(bpf, len(bpf))[2])
    bsf_amp = analyzer.transfer(analyzer.fourier(bsf, len(bsf))[2])

    fig, ax = plt.subplots(4)
    ax[0].plot([1/ (len(lpf_amp) * dt) * i for i in range(int(len(lpf_amp)/2))], lpf_amp[:int(len(lpf_amp)/2)])
    ax[0].set_title('lpf')

    ax[1].plot([1/ (len(hpf_amp) * dt) * i for i in range(int(len(hpf_amp)/2))], hpf_amp[:int(len(hpf_amp)/2)])
    ax[1].set_title('hpf')

    ax[2].plot([1/ (len(bpf_amp) * dt) * i for i in range(int(len(bpf_amp)/2))], bpf_amp[:int(len(bpf_amp)/2)])
    ax[2].set_title('bpf')

    ax[3].plot([1/ (len(bsf_amp) * dt) * i for i in range(int(len(bsf_amp)/2))], bsf_amp[:int(len(bsf_amp)/2)])
    ax[3].set_title('bsf')

    plt.show()

def _plot_smt_filter(analyzer, data, data_spectr, filt_freq, conv, conv_filt, flt_name):
    fig, ax = plt.subplots(5)
    dt = 0.002

    ax[0].plot(data)
    # ax[0].set_title('data')

    ax[1].plot(*analyzer.spectre_f(data_spectr, dt))
    # ax[1].set_title('data spectre')

    ax[2].plot(*analyzer.spectre_f(filt_freq, dt))
    # ax[2].set_title('lpf')

    ax[3].plot(conv)
    # ax[3].plot('conv')

    ax[4].plot(*analyzer.spectre_f(conv_filt, dt))
    # ax[4].plot('conv spectre')

    plt.show()

def plot_filter_dat(model : Model, analyzer : Analyzer, processor : Processor, fname):
    fc1 = 5
    fc2 = 40
    dt = 0.005
    m = 64
    data = np.fromfile('./data/' + fname, dtype=np.float32)
    data_spectr = analyzer.fourier(data, len(data))[2]


    lpf = processor.reflect_lpf(processor.lpf(5, dt, m))
    hpf = processor.hpf(50, dt, m)
    bpf = processor.bpf(fc1, fc2, dt, m)
    bsf = processor.bsf(fc1, fc2, dt, m)

    lpf_amp = analyzer.transfer(analyzer.fourier(lpf, len(lpf))[2])
    hpf_amp = analyzer.transfer(analyzer.fourier(hpf, len(hpf))[2])
    bpf_amp = analyzer.transfer(analyzer.fourier(bpf, len(bpf))[2])
    bsf_amp = analyzer.transfer(analyzer.fourier(bsf, len(bsf))[2])

    convol_lpf = model.convolModel(data, len(data), lpf, 2*m+1)
    convol_hpf = model.convolModel(data, len(data), hpf, 2*m+1)
    convol_bpf = model.convolModel(data, len(data), bpf, 2*m+1)
    convol_bsf = model.convolModel(data, len(data), bsf, 2*m+1)


    convol_lpf_amp = analyzer.fourier(convol_lpf, len(convol_lpf))[2]
    convol_hpf_amp = analyzer.fourier(convol_hpf, len(convol_hpf))[2]
    convol_bpf_amp = analyzer.fourier(convol_bpf, len(convol_bpf))[2]
    convol_bsf_amp = analyzer.fourier(convol_bsf, len(convol_bsf))[2]

    _plot_smt_filter(analyzer, data, data_spectr, lpf_amp, convol_lpf, convol_lpf_amp, 'lpf')

    _plot_smt_filter(analyzer, data, data_spectr, hpf_amp, convol_hpf, convol_hpf_amp, 'hpf')

    _plot_smt_filter(analyzer, data, data_spectr, bpf_amp, convol_bpf, convol_bpf_amp, 'bpf')

    _plot_smt_filter(analyzer, data, data_spectr, bsf_amp, convol_bsf, convol_bsf_amp, 'bsf')

def play_wav(inout : InOuter , filename):
    data = inout.read_wav('data/' + filename)
    print(data)

    inout.write_wav('data/' + filename.rstrip('.wav') + '_louder.wav', np.array(data['data'] * 1.5, dtype=np.int16), data['rate'])

    plt.plot(data['data'], c='tab:blue')
    plt.show()

    plt.plot(data['data'] * 1.5, c='tab:blue')
    plt.show()

def play_wav_emphasis(model : Model, analyzer, processor : Processor, inout : InOuter , filename):
    data = inout.read_wav('data/' + filename)
    plt.plot(data['data'], c='tab:blue')
    plt.show()

    data_emp = model.multArrays(data['data'], processor.rw(0.5, 2000,10000, 4, 11500, 15000, data['N']))
    inout.write_wav('data/' + filename.rstrip('.wav') + '_emp.wav', np.array(data_emp, dtype=np.int16), data['rate'])
    plt.plot(data_emp, c='tab:blue')
    plt.show()

if __name__ == "__main__":
    model = Model()
    analyzer = Analyzer()
    processor = Processor()
    inout = InOuter()
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
    # plot_anti_evth(model, processor)
    # plot_another_anti(model, processor)
    # plot_cardio(model, analyzer, processor)
    # plot_filter_gp(model, analyzer, processor)
    # plot_filter_dat(model, analyzer, processor, 'pgp_dt0005.dat')
    # play_wav(inout, 'surf.wav')
    play_wav_emphasis(model, analyzer, processor, inout, 'word.wav')
