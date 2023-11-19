from src.model import Model
from src.analysis import Analyzer
from src.processing import Processor
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm
import scipy.optimize as opt
from src.utils.statistics import mean_absolute_error
import sklearn.metrics as m
from src.utils.statistics import pearson_coef
# np.random.seed(452) # 123 654
model = Model()
analyzer = Analyzer()
processor = Processor()


def _plot_corr_res(values, L):
    plt.grid()
    plt.plot(values)
    plt.show()


def analyze_data(data):
    plt.plot(data)
    plt.title('Amazon')
    plt.show()
   
    stats = analyzer.statistics(data)
    print(stats)

    L = range(len(data))
    values = [analyzer.acf(data, len(data), l, calc_cov=False) for l in L]
    _plot_corr_res(values, L)

    print(analyzer.stationary(data, len(data), 10))


aapl = model.get_stocks_data('aapl', '2y', '1d')
# aapl = np.array(aapl['Close'])
aapl = np.array(processor.antiTrendNonLinear(aapl['Close'], len(aapl['Close']), 55))

tsla = model.get_stocks_data('tsla', '2y', '1d') 
# tsla = np.array(tsla['Close'])
tsla = np.array(processor.antiTrendNonLinear(tsla['Close'], len(tsla['Close']), 55))

amzn = model.get_stocks_data('amzn', '2y', '1d')
amzn = np.array(processor.antiTrendNonLinear(amzn['Close'], len(amzn['Close']), 55))
# amzn = np.array(amzn['Close'])


# analyze_data(aapl)
# analyze_data(tsla)
# analyze_data(amzn)
# analyze_data(amzn)
# print(pearson_coef(aapl, tsla))
# print(pearson_coef(aapl, amzn))
# print(pearson_coef(amzn, tsla))
def get_ito_optimal_parameters(data_tr):
    def fun(args):
        a,b,c,d = args
        data_pred = np.array([model.ito_process(t, a, b, c, d) for t in range(len(data_tr))])
        return mean_absolute_error(data_tr, data_pred)
    

    bounds = ((-10000, 10000), (-100, 1),(0, 2 * np.pi),(0, 100))

    res = opt.differential_evolution(fun, bounds = bounds)
    a, b, c, d = res.x
    data_pred = np.array([model.ito_process(t, a, b, c, d) for t in range(len(data_tr))])
    plt.plot(data_pred, label='pred')
    plt.plot(data_tr, label='true')
    plt.title('Amazon-ito')
    plt.legend()
    plt.show()
    print(res)

get_ito_optimal_parameters(amzn)
