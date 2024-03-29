import numpy as np
from scipy.io import wavfile
from itertools import chain
import soundfile as sf
import cv2
import matplotlib.pyplot as plt


class InOuter:
    def read_wav(self, file_name):
        samplerate, data = wavfile.read(file_name)
        return {'rate': samplerate, 'data': data, 'N': len(data)}

    def write_wav(self, file_name, data, rate):
        sf.write(file_name, data, rate)

    def read_jpg(self, file_name):
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        return np.array(img, dtype=np.int32)

    def show_jpg(self, img, cmap='gray'):
        plt.imshow(img, cmap)

    def write_jpg(self, data, file_name):
        cv2.imwrite(file_name, data)

    def read_xcr(self, file_name, width=1024, height=1024):
        binary_data = np.fromfile(file_name, dtype=np.uint8)
        xcr_data = binary_data[2048:2048 + width * height * 2]
        xcr_data_img = np.array(list(zip(xcr_data[1::2], xcr_data[::2]))).view(np.uint16).reshape(width, height)

        return self.cast_to_interval(xcr_data_img, 255)

    def write_bin(self, array, file_name):
        with open(file_name, 'wb') as fl:
            array.tofile(fl)

    def cast_to_interval(self, data, interval_value):
        return (data - np.min(data)) / (np.max(data) - np.min(data)) * interval_value

    def resize_image(self, img, scale_x, scale_y, method):
        methods = {'nearest': cv2.INTER_NEAREST,
                   'bilinear': cv2.INTER_LINEAR}

        interpolation = methods[method]
        resized_image = cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=interpolation)

        return resized_image
