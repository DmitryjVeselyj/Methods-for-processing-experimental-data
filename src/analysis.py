import src.utils.statistics as stat
import numpy as np
from matplotlib import pyplot as plt
from src.utils.fourier_transform import calc_real_part, calc_imag_part, calc_ampliture, NoWindowFunc


class Analyzer:
    def statistics(self, data, N=None, type=None):
        return {"mean": stat.mean(data),
                "min": stat.min(data),
                "max": stat.max(data),
                "var": stat.var(data),
                "std": stat.std(data),
                "skew": stat.skew(data),
                "kurtosis": stat.kurtosis(data),
                "mse": stat.mean_squared_error(data)
                }

    def stationary(self, data, N, M):
        chunks = np.array_split(data, M)
        mean_vals = np.array([stat.mean(chunk) for chunk in chunks])
        std_vals = np.array([stat.std(chunk) for chunk in chunks])

        # переписать через if или ещё как-нибудь, чтобы не считать лишнего
        mean_rel_measure = [abs(mean_vals[i] - mean_vals[j]) for i in range(M) for j in range(i + 1, M)]
        std_rel_measure = [abs(std_vals[i] - std_vals[j]) for i in range(M) for j in range(i + 1, M)]
        range_interval = 0.05 * (stat.max(data) - stat.min(data))
        return all(map(lambda elem: elem < range_interval, mean_rel_measure)) and \
            all(map(lambda elem: elem < range_interval, std_rel_measure))

    def hist(self, data, N, M):
        offset = min(data)

        bins = np.zeros(M)
        xi = np.linspace(min(data), max(data), M)
        iwidth = max([xi[i + 1] - xi[i] for i in range(len(xi) - 1)])

        for d in data:
            bin_index = int((d - offset) / iwidth)
            bins[bin_index] += 1

        # plt.bar(xi, bins, width=iwidth)
        plt.hist(data, bins=M)
        plt.show()
        return xi, bins, iwidth

    def acf(self, data, N, L, calc_cov=True):
        if calc_cov:
            return stat.covariance(data, data, L)
        return stat.autocorrelation(data, L)

    def ccf(self, x, y, L):
        return stat.covariance(x, y, L)

    def fourier(self, data, N, window=NoWindowFunc()):
        re = [0] * N
        im = [0] * N
        amp = [0] * N

        for n in range(N):
            re[n] = calc_real_part(data, N, n, window)
            im[n] = calc_imag_part(data, N, n, window)
            amp[n] = calc_ampliture(re[n], im[n])

        return re, im, amp

    def spectr_fourier(self, amp, dt):
        plt.plot([1 / (len(amp) * dt) * i for i in range(int(len(amp) / 2))], amp[:int(len(amp) / 2)])
        plt.show()

    def spectre_f(self, amp, dt):
        return [1 / (len(amp) * dt) * i for i in range(int(len(amp) / 2))], amp[:int(len(amp) / 2)]

    def transfer(self, amp):
        return [x * len(amp) for x in amp]

# -----------------------------------------------kursovaya----------------------------------------
