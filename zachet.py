from src.model import Model
from src.analysis import Analyzer
from src.processing import Processor
from src.inout import InOuter
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    model = Model()
    analyzer = Analyzer()
    processor = Processor()
    inout = InOuter()


    filename = 'v31.bin'
    data = np.fromfile('./data/' + filename, dtype=np.float32)
    dt = 0.0005
    print(len(data))
    plt.plot(data)
    plt.show()
    data_spectr = analyzer.fourier(data, len(data))[2]
    plt.plot(*analyzer.spectre_f(data_spectr, dt), c='tab:blue')
    plt.show()
    
