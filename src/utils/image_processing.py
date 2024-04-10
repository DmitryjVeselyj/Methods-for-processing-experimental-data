import cv2
import numpy as np


def calc_hist(image_data, normalize=True):
    max_pixel_value = np.max(image_data)

    hist_values = np.zeros(max_pixel_value + 1)
    for x in range(image_data.shape[0]):
        for y in range(image_data.shape[1]):
            i = image_data[x, y]
            hist_values[i] += 1

    if normalize:
        return hist_values / (image_data.shape[0] * image_data.shape[1])

    return hist_values


def calc_cdf(histogram):
    cdf = np.zeros(len(histogram))

    for x in range(len(histogram)):
        for y in range(x + 1):
            cdf[x] += histogram[y]

    return cdf


def image_diff(img1, img2, method = 'bilinear'):
    methods = {'nearest': cv2.INTER_NEAREST,
               'bilinear': cv2.INTER_LINEAR}

    interpolation = methods[method]

    if img1.shape[0] > img2.shape[0]:
        img2 = cv2.resize(img2, None, fx=img1.shape[0] / img2.shape[0], fy=1, interpolation=interpolation)
    else:
        img1 = cv2.resize(img1, None, fx=img2.shape[0] / img1.shape[0], fy=1, interpolation=interpolation)

    if img1.shape[1] > img2.shape[1]:
        img2 = cv2.resize(img2, None, fx=1, fy=img1.shape[1] / img2.shape[1], interpolation=interpolation)
    else:
        img1 = cv2.resize(img1, None, fx=1, fy=img2.shape[1] / img1.shape[1], interpolation=interpolation)

    return np.abs(img1 - img2)