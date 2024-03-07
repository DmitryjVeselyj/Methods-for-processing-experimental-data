import src.utils.statistics as stat
import numpy as np
import cv2
from src.utils.image_processing import calc_hist, calc_cdf


class Processor:
    def antiShift(self, data):
        return data - stat.mean(data)

    def antiSpike(self, data, N, R):
        outdata = np.zeros(N)
        for i in range(1, N - 1):
            if abs(data[i]) > R:
                outdata[i] = (data[i - 1] + data[i + 1]) / 2
            else:
                outdata[i] = data[i]
        outdata[0] = data[0]
        outdata[N - 1] = data[N - 1]
        return outdata

    def antiTrendLinear(self, data, N):
        return [data[i + 1] - data[i] for i in range(N - 1)]

    def antiTrendNonLinear(self, data, N, W):
        return [1 / W * sum(data[n:n + W]) for n in range(N - W)]

    def antiNoise(self, data: list, N, M):
        return [1 / M * sum(data[:, i]) for i in range(N)]

    def handle(self, data):
        pass

    def lpf(self, fc, dt, m):
        d = [0.35577019, 0.2436983, 0.07211497, 0.00630165]
        # rectangular part weights
        fact = 2 * fc * dt
        lpw = [fact]
        arg = fact * np.pi
        for i in range(1, m + 1):
            lpw.append(np.sin(arg * i) / (np.pi * i))
        # trapezoid smoothing at the end
        lpw[m] = lpw[m] / 2
        # P310 smoothing window
        sumg = lpw[0]
        for i in range(1, m + 1):
            sum = d[0]
            arg = np.pi * i / m
            for k in range(1, 4):
                sum += 2 * d[k] * np.cos(arg * k)
            lpw[i] = lpw[i] * sum
            sumg += 2 * lpw[i]
        for i in range(m + 1):
            lpw[i] = lpw[i] / sumg
        return lpw

    def reflect_lpf(self, lpw):
        reflection = [lpw[i] for i in range(len(lpw) - 1, 0, -1)]
        reflection.extend(lpw)
        return reflection

    def hpf(self, fc, dt, m):
        lpw = self.reflect_lpf(self.lpf(fc, dt, m))
        hpw = [-lpw[k] for k in range(m)] + [1 - lpw[m]] + [-lpw[k] for k in range(m + 1, 2 * m + 1)]
        return hpw

    def bpf(self, fc1, fc2, dt, m):
        lpw1 = self.reflect_lpf(self.lpf(fc1, dt, m))
        lpw2 = self.reflect_lpf(self.lpf(fc2, dt, m))
        bpw = [lpw2[k] - lpw1[k] for k in range(2 * m + 1)]
        return bpw

    def bsf(self, fc1, fc2, dt, m):
        lpw1 = self.reflect_lpf(self.lpf(fc1, dt, m))
        lpw2 = self.reflect_lpf(self.lpf(fc2, dt, m))
        bsw = [lpw1[k] - lpw2[k] for k in range(m)] + [1 + lpw1[m] - lpw2[m]] + [lpw1[k] - lpw2[k] for k in
                                                                                 range(m + 1, 2 * m + 1)]
        return bsw

    def rw(self, c1, n1, n2, c2, n3, n4, N):
        return [1] * n1 + (n2 - n1) * [c1] + [1] * (n3 - n2) + \
            [c2] * (n4 - n3) + [1] * (N - n4)

    def negative(self, image_data):
        max_pixel_value = np.max(image_data)
        return np.vectorize(lambda x: max_pixel_value - x)(image_data)

    def gamma_correction(self, image_data, const, gamma):
        return np.vectorize(lambda x: const * x ** gamma)(image_data)

    def log_correction(self, image_data, const):
        return np.vectorize(lambda x: const * np.log(x + 1))(image_data)

    def eqval_hist_correction(self, image_data):
        max_pixel_value = np.max(image_data)
        hist = calc_hist(image_data)
        cdf = calc_cdf(hist)
        return np.vectorize(lambda x: round(max_pixel_value * cdf[x]))(image_data)

    def custom_hist_correction(self, image_data, custom_hist):
        max_pixel_value = np.max(image_data)
        native_hist = calc_hist(image_data)
        
        cdf_native = calc_cdf(native_hist)
        cdf_custom_hist = calc_cdf(custom_hist)
        
        sk = (max_pixel_value * cdf_native).round()
        gk = (max_pixel_value * cdf_custom_hist).round()

        map_pixel_rules = {}
        for s in sk:
            map_pixel_rules[s] = (np.abs(gk - s)).argmin()

        equal_hist_image = np.clip(np.vectorize(lambda x: round(max_pixel_value * cdf_native[x]))(image_data), 0, 255)
        for x in range(equal_hist_image.shape[0]):
            for y in range(equal_hist_image.shape[1]):
                map_rule = map_pixel_rules.get(equal_hist_image[x, y], None)
                if map_rule is not None:
                    equal_hist_image[x, y] = map_rule

        return equal_hist_image
