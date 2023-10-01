import math

def moment(data, order, central = True):
    mean_val = 0 if not central else 1/len(data) * sum(data)
    return 1 / len(data) * sum((elem - mean_val)**order for elem in data)

def mean(data):
    return moment(data, order=1, central=False)

def min(data):
    min_val = math.inf
    for elem in data:
        if elem < min_val:
            min_val = elem
    return min_val

def max(data):
    max_val = -math.inf
    for elem in data:
        if elem > max_val:
            max_val = elem
    return max_val

def var(data):
    return moment(data, order=2, central=True)

def std(data):
    return math.sqrt(var(data))

def asymmetry(data):
    return moment(data, order=3, central=True)

def skew(data):
    return asymmetry(data) / std(data)**3

def excess(data):
    return moment(data, order=4, central=True)

def kurtosis(data):
    return excess(data) / std(data)**4 - 3

def mean_square(data):
    return moment(data, order=2, central=False)

def mean_squared_error(data):
    return math.sqrt(mean_square(data))