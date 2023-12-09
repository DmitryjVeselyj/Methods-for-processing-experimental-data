from src.handler import AbstractHandler
from src.mediator import BaseComponent

from scipy.io import wavfile
# import librosa
import soundfile as sf


class InOuter(AbstractHandler, BaseComponent):
    def handle(self, data):
        pass


    def read_wav(self, file_name):
        samplerate, data = wavfile.read(file_name)
        return {'rate': samplerate, 'data': data, 'N' : len(data)}

    def write_wav(self, file_name, data, rate):
        sf.write(file_name, data, rate)

