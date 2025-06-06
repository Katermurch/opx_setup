import numpy as np 
import pyvisa
import time 
import pandas as pd

def parse_freq(freq_str):
    try:
        freq_str.replace('\r', '').replace('\n', '')
    except:
        raise ValueError("Invalid frequency string format")
    
    return float(freq_str)
rm = pyvisa.ResourceManager()
class SpectrumAnalyzer:
    gpib_address = "GPIB0::18::INSTR" 
    def __init__(self):
        self.reset()
        self.inst = rm.open_resource(self.gpib_address)
        self.center_freq = None
        self.span_freq = None
        self.start_freq = None
        self.stop_freq = None

    def get_freqs(self):
        # Query center frequency and span
        self.inst.write("CF?")
        cf_str = self.inst.read()
        self.inst.write("SP?")
        sp_str = self.inst.read()
        self.center_freq = parse_freq(cf_str)
        self.span_freq = parse_freq(sp_str)
        self.start_freq = self.center_freq - self.span_freq / 2
        self.stop_freq = self.center_freq + self.span_freq / 2

    def read(self):
        """
        Read the current trace data and return a DataFrame with frequency and amplitude.
        Returns:
            pd.DataFrame: 'Frequency_Hz' and 'Amplitude' columns.
        """
        self.get_freqs()
        self.inst.write("TRA?")
        data = self.inst.read()
        data = data.replace('\r', '').replace('\n', '')
        trace = np.fromstring(data, sep=',')
        N = len(trace)
        freq_axis = np.linspace(self.start_freq, self.stop_freq, N)
        df = pd.DataFrame({'Frequency_Hz': freq_axis, 'Amplitude': trace})
        return df

    