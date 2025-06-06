import json
import numpy as np
import h5py
import experiment_configuration.values as config_values
from scipy.signal.windows import chebwin, taylor, flattop
from scipy.fft import fft, fftfreq
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

json_file = r'C:\Users\quantum1\OneDrive\Documents\Python Scripts\Important Blue Fridge Python Files\New\OPX_setup_06022025\latest_values.json'
with open(json_file, 'r') as f:
    latest_values = json.load(f)

"""OPX controller name"""
cname = 'con1'
qop_ip = '10.225.208.35'
port = 80


"""These are the res settings"""
# Frequencies
lo_res = 6982000000

if_res_1 = latest_values['res_1_freq'] - lo_res
if_res_2 = latest_values['res_2_freq'] - lo_res
if_res_C = latest_values['res_C_freq'] - lo_res

# TWPA
twpa_freq = latest_values['twpa_freq']
twpa_power = latest_values['twpa_power']
twpa_flux_volt = latest_values['twpa_flux_volt']

# Readout pulse
time_readout_res_1 = latest_values['time_readout_res_1']
time_readout_res_2 = latest_values['time_readout_res_2']
time_readout_res_C = latest_values['time_readout_res_C']
amp_readout_res_1 = latest_values['amp_readout_res_1'][0]
amp_readout_res_2 = latest_values['amp_readout_res_2'][0]
amp_readout_res_C = latest_values['amp_readout_res_C'][0]


readout_wf_t = np.arange(16000)

readout_wf_res_1 = np.ones(16000)*amp_readout_res_1 
readout_wf_res_2 = np.ones(16000)*amp_readout_res_2 
readout_fast_wf_res_2  = np.ones(16000)*amp_readout_res_2 
readout_wf_res_C = np.full_like(readout_wf_t, fill_value=amp_readout_res_C, dtype=np.float64)


integration_weights_file_QC = r'C:/shared-root/users/Alexandria/integration_weights.hdf5'
with h5py.File(integration_weights_file_QC, 'r') as hf:
    durations = np.array(hf['durations'][:])
    res_C_iw_I = np.array(hf['res_C_iw_I'][:])
    res_C_iw_Q = np.array(hf['res_C_iw_Q'][:])

integration_weights_file_Q1_Q2 = r'C:/shared-root/users/Alexandria/quantum_efficiency/integration_weights_Q1_Q2.hdf5'
with h5py.File(integration_weights_file_Q1_Q2, 'r') as hf:
    durations = np.array(hf['durations'][:])
    res_2_iw_I = np.array(hf['res_2_iw_I'][:])
    res_2_iw_Q = np.array(hf['res_2_iw_Q'][:])
    res_1_iw_I = np.array(hf['res_1_iw_I'][:])
    res_1_iw_Q = np.array(hf['res_1_iw_Q'][:])



save_load_dir = r'C:/shared-root/users/Alexandria/engine/Data'


"""These are the qubit settings"""
def sine_w(length, f_offset):
    n = np.arange(length)
    N = length - 1
    n_offset = n - N / 2
    return np.sin((np.pi / N) * n) * np.exp((2j * np.pi * f_offset) * n_offset)


def gen_sqisw_12_wf(amp_QC_ffl, freq_QC_ffl, sqisw_12_len, sqisw_12_dc, sqisw_12_taylor_M, sqisw_12_taylor_nbar, sqisw_12_taylor_sll):
    t = np.arange(sqisw_12_len) - 0.5 * (sqisw_12_len - 1)
    taylor_win = taylor(sqisw_12_taylor_M, sqisw_12_taylor_nbar, sqisw_12_taylor_sll)
    taylor_win /= np.sum(taylor_win)
    square_wf = np.full(sqisw_12_len - sqisw_12_taylor_M + 1, 1.0)
    win = np.convolve(square_wf, taylor_win)
    sqisw_12_wf = sqisw_12_dc + amp_QC_ffl * np.cos(2 * np.pi * freq_QC_ffl * t * 1e-9)
    sqisw_12_wf = win * sqisw_12_wf
    return sqisw_12_wf


def myround(x180_len, base=4):
    x180_len = np.round(x180_len)
    return base * round(x180_len/base)

################# Frequencies
lo_qubit = 4241000000
lo_mod = 5347000000

Q1_freq = latest_values['Q1_freq']
Q2_freq = latest_values['Q2_freq']
QC_freq = latest_values['QC_freq']

Q1_ef_freq = latest_values['Q1_ef_freq']
Q2_ef_freq = latest_values['Q2_ef_freq']
QC_ef_freq = latest_values['QC_ef_freq']

if_Q1 = latest_values['Q1_freq'] - lo_qubit
if_Q2 = latest_values['Q2_freq'] - lo_qubit
if_QC = latest_values['QC_freq'] - lo_mod
if_QC_dummy = (Q2_freq - Q1_freq) // 2

if_ACS = latest_values['ACS_freq'] - lo_qubit

if_Q1_ef = latest_values['Q1_ef_freq'] - lo_qubit
if_Q2_ef = latest_values['Q2_ef_freq'] - lo_qubit
if_QC_ef = latest_values['QC_ef_freq'] - lo_mod

# Pi-pulse
Q1_DL_x180_len = latest_values['Q1_DL_x180_len'] #96
Q1_DL_x180_amp = latest_values['Q1_DL_x180_amp'] #0.102


Q1_DL_x180_amp *=1.0

Q1_DL_x90_len = Q1_DL_x180_len // 2
Q1_DL_x90_amp = Q1_DL_x180_amp
Q1_DL_x180_long_amp = latest_values['Q1_DL_x180_long_amp']
Q2_DL_x180_long_amp = latest_values['Q2_DL_x180_long_amp']
QC_x180_long_amp = latest_values['QC_x180_long_amp']



Q2_DL_x180_len = latest_values['Q2_DL_x180_len']
Q2_DL_x180_amp = latest_values['Q2_DL_x180_amp'][0]



Q2_DL_x90_len = Q2_DL_x180_len // 2
Q2_DL_x90_amp = Q2_DL_x180_amp

QC_x180_len = latest_values['QC_x180_len']
QC_x180_amp = latest_values['QC_x180_amp']
QC_x90_len = QC_x180_len // 2
QC_x90_amp = QC_x180_amp

Q1_ef_DL_x180_len = latest_values['Q1_ef_DL_x180_len']
Q1_ef_DL_x180_amp = latest_values['Q1_ef_DL_x180_amp']
Q1_ef_DL_x90_len = Q1_ef_DL_x180_len // 2
Q1_ef_DL_x90_amp = Q1_ef_DL_x180_amp

Q2_ef_DL_x180_len = latest_values['Q2_ef_DL_x180_len']
Q2_ef_DL_x180_amp = latest_values['Q2_ef_DL_x180_amp']
Q2_ef_DL_x90_len = Q2_ef_DL_x180_len // 2
Q2_ef_DL_x90_amp = Q2_ef_DL_x180_amp

QC_ef_x180_len = latest_values['QC_ef_x180_len']
QC_ef_x180_amp = latest_values['QC_ef_x180_amp']
QC_ef_x90_len = QC_ef_x180_len // 2
QC_ef_x90_amp = QC_ef_x180_amp

x180_cos_len = 88
x90_cos_len = 44
x180_cos_wf = sine_w(x180_cos_len, 0)
x90_cos_wf = sine_w(x90_cos_len, 0)

Q1_DL_x180_cos_amp = latest_values['Q1_DL_x180_cos_amp']
Q1_DL_x90_cos_amp = latest_values['Q1_DL_x90_cos_amp']
Q2_DL_x180_cos_amp = latest_values['Q2_DL_x180_cos_amp']
Q2_DL_x90_cos_amp = latest_values['Q2_DL_x90_cos_amp']
QC_x180_cos_amp = latest_values['QC_x180_cos_amp']
QC_x90_cos_amp = latest_values['QC_x90_cos_amp']


x180_ef_cos_len = 64
x90_ef_cos_len = 32
x180_ef_cos_wf = sine_w(x180_ef_cos_len, 0)
x90_ef_cos_wf = sine_w(x90_ef_cos_len, 0)

Q1_ef_DL_x180_cos_amp = latest_values['Q1_ef_DL_x180_cos_amp']
Q1_ef_DL_x90_cos_amp = latest_values['Q1_ef_DL_x90_cos_amp']
Q2_ef_DL_x180_cos_amp = latest_values['Q2_ef_DL_x180_cos_amp']
Q2_ef_DL_x90_cos_amp = latest_values['Q2_ef_DL_x90_cos_amp']
QC_ef_x180_cos_amp = latest_values['QC_ef_x180_cos_amp']
QC_ef_x90_cos_amp = latest_values['QC_ef_x90_cos_amp']

amp_ACS = latest_values['amp_ACS']
QC_FFL_amp = latest_values['QC_FFL_amp']
QC_FFL_freq = latest_values['QC_FFL_freq']
QC_FFL_len = 4 * round(latest_values['QC_FFL_len_mod_4'])
QC_FFL_frame_rot = latest_values['QC_FFL_frame_rot']

#DC offsets
DC_offset1 = latest_values['DC_offset1']
DC_offset2 = latest_values['DC_offset2']
DC_offset3 = latest_values['DC_offset3']
DC_offset4 = latest_values['DC_offset4']
DC_offset5 = latest_values['DC_offset5']
DC_offset6 = latest_values['DC_offset6']
DC_offset7 = latest_values['DC_offset7']
DC_offset8 = latest_values['DC_offset8']
DC_offset9 = latest_values['DC_offset9']
DC_offset10 = latest_values['DC_offset10']
DC_offset9_readout = latest_values['DC_offset9_readout']
DC_offset10_readout = latest_values['DC_offset10_readout']
DC_offset9_qubit = latest_values['DC_offset9_qubit']
DC_offset10_qubit = latest_values['DC_offset10_qubit']

DC_offset_input1 = 0.0
DC_offset_input2 = 0.0


# Mixer correction matrices
Q1_DL_mixer_C_matrix = latest_values['Q1_DL_mixer_C_matrix']
Q2_DL_mixer_C_matrix = latest_values['Q2_DL_mixer_C_matrix']
QC_mixer_C_matrix = latest_values['QC_mixer_C_matrix']
Q1_ef_DL_mixer_C_matrix = latest_values['Q1_ef_DL_mixer_C_matrix']
Q2_ef_DL_mixer_C_matrix = latest_values['Q2_ef_DL_mixer_C_matrix']
QC_ef_mixer_C_matrix = latest_values['QC_ef_mixer_C_matrix']
res_1_mixer_C_matrix = latest_values['res_1_mixer_C_matrix']
res_2_mixer_C_matrix = latest_values['res_2_mixer_C_matrix']
res_C_mixer_C_matrix = latest_values['res_C_mixer_C_matrix']
ACS_mixer_C_matrix = latest_values['ACS_mixer_C_matrix']

"""Below is the config_std that is imported to scripts"""
config = {
    'version': 1,
    'controllers': {
        cname: {
            'type': cname,
            'analog_outputs': {
                1: {'offset': DC_offset1},
                2: {'offset': DC_offset2},
                3: {'offset': DC_offset3},
                4: {'offset': DC_offset4},
                5: {'offset': DC_offset5},
                6: {'offset': DC_offset6},
                7: {'offset': DC_offset7},
                8: {'offset': DC_offset8},
                9: {'offset': DC_offset9},
                10: {'offset': DC_offset10}
            },
            'digital_outputs': {
                1: {},
                2: {},
                3: {}
            },
            'analog_inputs': {
                1: {'gain_db': 20, 'offset': DC_offset_input1},
                2: {'gain_db': 20, 'offset': DC_offset_input2},
            }
        },
    },
    'elements': {
        'res_1': {
            'mixInputs': {
                'I': (cname, 9),
                'Q': (cname, 10),
                'mixer': 'mixer_readout_line',
                'lo_frequency': lo_res
            },
            'digitalInputs': {
                'switch_readout': {
                    'buffer': 0,
                    'delay': 136,
                    'port': (cname, 1)
                },
                'switch_mod': {
                    'buffer': 0,
                    'delay': 136,
                    'port': (cname, 2)
                }
            },
            'thread': 'Q1',
            'intermediate_frequency': if_res_1,
            'operations': {
                'square': 'square_pulse',
                'readout_16us': 'readout_16us_pulse_res_1',
                'readout_fast': 'readout_fast_pulse_res_1',
                'readout_calib': 'readout_calib_pulse_res_1',
                'readout_unit': 'readout_unit_pulse_res_1',
                'readout_unit_5us': 'readout_unit_5us_pulse_res_1',
                'readout_unit_4us': 'readout_unit_4us_pulse_res_1',
                'readout_unit_2us': 'readout_unit_2us_pulse_res_1',
                'readout_unit_1us': 'readout_unit_1us_pulse_res_1'
            },
            'time_of_flight': 1500,
            'smearing': 0,
            'outputs': {
                'out1': (cname, 1),
                'out2': (cname, 2)
            }
        },
        'res_2': {
            'mixInputs': {
                'I': (cname, 9),
                'Q': (cname, 10),
                'mixer': 'mixer_readout_line',
                'lo_frequency': lo_res
            },
            'digitalInputs': {
                'switch_readout': {
                    'buffer': 0,
                    'delay': 136,
                    'port': (cname, 1)
                },
                'switch_mod': {
                    'buffer': 0,
                    'delay': 136,
                    'port': (cname, 2)
                }
            },
            'thread': 'Q2',
            'intermediate_frequency': if_res_2,
            'operations': {
                'square': 'square_pulse',
                'readout_16us': 'readout_16us_pulse_res_2',
                'readout_fast_3200': 'readout_fast_pulse_res_2_3200',
                'readout_fast_6400': 'readout_fast_pulse_res_2_6400',
                'readout_calib': 'readout_calib_pulse_res_2',
                'readout_calib_6400': 'readout_calib_pulse_6400_res_2',
                'readout_unit': 'readout_unit_pulse_res_2',
                'readout_unit_5us': 'readout_unit_5us_pulse_res_2',
                'readout_unit_4us': 'readout_unit_4us_pulse_res_2',
                'readout_unit_2us': 'readout_unit_2us_pulse_res_2',
                'readout_unit_1us': 'readout_unit_1us_pulse_res_2'
            },
            'time_of_flight': 1500,
            'smearing': 0,
            'outputs': {
                'out1': (cname, 1),
                'out2': (cname, 2)
            }
        },
        'res_C': {
            'mixInputs': {
                'I': (cname, 9),
                'Q': (cname, 10),
                'mixer': 'mixer_readout_line',
                'lo_frequency': lo_res
            },
            'digitalInputs': {
                'switch_readout': {
                    'buffer': 0,
                    'delay': 160,
                    'port': (cname, 1)
                }#,
                # 'switch_mod': {
                #     'buffer': 0,
                #     'delay': 160,
                #     'port': (cname, 2)
                # }
            },
            'thread': 'QC',
            'intermediate_frequency': if_res_C,
            'operations': {
                'square': 'square_pulse',
                'readout_16us': 'readout_16us_pulse_res_C',
                'readout_fast': 'readout_fast_pulse_res_C',
                'readout_calib': 'readout_calib_pulse_res_C'
            },
            'time_of_flight': 24,
            'smearing': 0,
            'outputs': {
                'out1': (cname, 1),
                'out2': (cname, 2)
            }
        },
        'Q1': {
            'mixInputs': {
                'I': (cname, 9),
                'Q': (cname, 10),
                'lo_frequency': lo_qubit
            },
            'digitalInputs': {
                'switch_qubit': {
                    'buffer': 0,
                    'delay': 136,
                    'port': (cname, 3)
                },
                'switch_mod': {
                    'buffer': 0,
                    'delay': 136,
                    'port': (cname, 2)
                }
            },
            'intermediate_frequency': if_Q1,
            'thread': 'Q1',
            'operations': {
                'square': 'square_pulse_Q1',
                'x180': 'x180_pulse_Q1',
                'x90': 'x90_pulse_Q1',
                'x180_cos': 'x180_cos_pulse_Q1',
                'x90_cos': 'x90_cos_pulse_Q1'
            }
        },
        'Q2': {
            'mixInputs': {
                'I': (cname, 9),
                'Q': (cname, 10),
                'lo_frequency': lo_qubit
            },
            'digitalInputs': {
                'switch_qubit': {
                    'buffer': 0,
                    'delay': 0,
                    'port': (cname, 3)
                },
                'switch_mod': {
                    'buffer': 0,
                    'delay': 160,
                    'port': (cname, 2)
                }
            },
            'intermediate_frequency': if_Q2,
            'thread': 'Q2',
            'operations': {
                'square': 'square_pulse',
                'x180': 'x180_pulse_Q2',
                'x90': 'x90_pulse_Q2'
            }
        },
        'ACS': {
            'mixInputs': {
                'I': (cname, 9),
                'Q': (cname, 10),
                'mixer': 'mixer_readout_line',
                'lo_frequency': lo_qubit,
            },
            'digitalInputs': {
                'switch_qubit': {
                    'buffer': 0,
                    'delay': 136,
                    'port': (cname, 3)
                },
                'switch_mod': {
                    'buffer': 136,
                    'delay': 136,
                    'port': (cname, 2)
                }
            },
            'intermediate_frequency': if_ACS,
            'thread': 'ACS',
            'operations': {
                'square': 'square_pulse',
                'square_pad': 'square_pad_pulse'
            }
        },
        'QC': {
            'mixInputs': {
                'I': (cname, 9),
                'Q': (cname, 10),
                'mixer': 'mixer_readout_line',
                'lo_frequency': lo_mod,
            },
            'intermediate_frequency': if_QC,
            'thread': 'QC',
            'operations': {
                'square': 'square_pulse',
                'x180': 'x180_pulse_QC',
                'x90': 'x90_pulse_QC',
                'x180_cos': 'x180_cos_pulse_QC',
                'x90_cos': 'x90_cos_pulse_QC'
            }
        },
        'QC_ef': {
            'mixInputs': {
                'I': (cname, 9),
                'Q': (cname, 10),
                'mixer': 'mixer_readout_line',
                'lo_frequency': lo_mod,
            },
            'intermediate_frequency': if_QC_ef,
            'thread': 'QC',
            'operations': {
                'square': 'square_pulse',
                'x180': 'x180_pulse_QC_ef',
                'x90': 'x90_pulse_QC_ef',
                'x180_cos': 'x180_cos_pulse_QC_ef',
                'x90_cos': 'x90_cos_pulse_QC_ef'
            }
        },
        'Q1_ef': {
            'mixInputs': {
                'I': (cname, 3),
                'Q': (cname, 4)
            },
            'digitalInputs': {
                'switch': {
                    'buffer': 0,
                    'delay': 0,
                    'port': (cname, 3)
                }
            },
            'intermediate_frequency': if_Q1_ef,
            'thread': 'Q1',
            'operations': {
                'square': 'square_pulse',
                'x180': 'x180_pulse_Q1_DL',
                'x90': 'x90_pulse_Q1_DL',
                'x180_cos': 'x180_cos_pulse_Q2_DL',
                'x90_cos': 'x90_cos_pulse_Q2_DL'
            }
        },
        'Q2_ef': {
            'mixInputs': {
                'I': (cname, 3),
                'Q': (cname, 4)
            },
            'digitalInputs': {
                'switch': {
                    'buffer': 0,
                    'delay': 0,
                    'port': (cname, 3)
                }
            },
            'intermediate_frequency': if_Q2_ef,
            'thread': 'Q2',
            'operations': {
                'square': 'square_pulse'
            }
        },
        'Q1_DL': {
            'mixInputs': {
                'I': (cname, 3),
                'Q': (cname, 4),
                'mixer': 'mixer_Q1_DL',  
                'lo_frequency': lo_qubit
            },
            'intermediate_frequency': if_Q1,
            'thread': 'Q1',
            'operations': {
                'square': 'square_pulse',
                'square_pad': 'square_pad_pulse',
                'x180': 'x180_pulse_Q1_DL',
                'x90': 'x90_pulse_Q1_DL',
                'x180_cos': 'x180_cos_pulse_Q1_DL',
                'x90_cos': 'x90_cos_pulse_Q1_DL'
            }
        },
        'Q1_ef_DL': {
            'mixInputs': {
                'I': (cname, 3),
                'Q': (cname, 4),
                'mixer': 'mixer_Q1_ef_DL',
                'lo_frequency': lo_qubit
            },
            'intermediate_frequency': if_Q1_ef,
            'thread': 'Q1',
            'operations': {
                'square': 'square_pulse',
                'x180': 'x180_pulse_Q1_ef_DL',
                'x90': 'x90_pulse_Q1_ef_DL',
                'x180_cos': 'x180_cos_pulse_Q1_ef_DL',
                'x90_cos': 'x90_cos_pulse_Q1_ef_DL'
            }
        },
        'Q2_DL': {
            'mixInputs': {
                'I': (cname, 5),
                'Q': (cname, 6),
                'mixer': 'mixer_Q2_DL',
                'lo_frequency': lo_qubit
            },
            'intermediate_frequency': if_Q2,
            'thread': 'Q2',
            'operations': {
                'square': 'square_pulse',
                'square_pad': 'square_pad_pulse',
                'x180': 'x180_pulse_Q2_DL',
                'x90': 'x90_pulse_Q2_DL',
                'x180_cos': 'x180_cos_pulse_Q2_DL',
                'x90_cos': 'x90_cos_pulse_Q2_DL'
            }
        },
        'Q2_ef_DL': {
            'mixInputs': {
                'I': (cname, 5),
                'Q': (cname, 6),
                'mixer': 'mixer_Q2_ef_DL',
                'lo_frequency': lo_qubit
            },
            'intermediate_frequency': if_Q2_ef,
            'thread': 'Q2',
            'operations': {
                'square': 'square_pulse',
                'x180': 'x180_pulse_Q2_ef_DL',
                'x90': 'x90_pulse_Q2_ef_DL',
                'x180_cos': 'x180_cos_pulse_Q2_ef_DL',
                'x90_cos': 'x90_cos_pulse_Q2_ef_DL'
            }
        },
        'Q2_DL_HF': {
            'mixInputs': {
                'I': (cname, 7),
                'Q': (cname, 8)
            }
        },
        'pd_1C': {
            'singleInput': {
                'port': (cname, 1)
            },
            'intermediate_frequency': 404000000,
            'operations': {
                'square': 'square_pulse_pd_1C',
                'sqisw': 'sqisw_pulse_pd_1C'
            }
        },
        'ext_flux': {
            'singleInput': {
                'port': (cname, 7)
            }
        },
        'pd_12': {
            'singleInput': {
                'port': (cname, 1)
            },
            'intermediate_frequency': QC_FFL_freq,
            'thread': 'Q1',
            'operations': {
                'square': 'square_pulse_pd_12'
            }
        },
        'QC_FFL_IQ': {
            'mixInputs': {
                'I': (cname, 1),
                'Q': (cname, 7),
                'mixer': 'mixer_QC_FFL_dummy',
                'lo_frequency': 0
            },
            'intermediate_frequency': (Q2_freq - Q1_freq) // 2,
            'operations': {
                'square': 'square_pulse_QC_FFL',
                'square_DC': 'square_pulse_DC_QC_FFL_IQ', 
            }
        },
        'QC_FFL': {
            'singleInput': {
                'port': (cname, 1),
            },
            'intermediate_frequency': (Q2_freq - Q1_freq) // 2,
            'operations': {
                'square_DC': 'square_pulse_DC_QC_FFL', 
            }
        },
        'Q2_FFL': {
            'singleInput': {
                'port': (cname, 2),
            },
            'intermediate_frequency': 0.0,
            'operations': {
                'square_DC': 'square_pulse_DC_Q2_FFL', 
            }
        },
        'dc_offset_Q2': {
            'singleInput': {
                'port': (cname, 2)
            }
        },
        'dc_offset_QC': {
            'singleInput': {
                'port': (cname, 1)
            }
        },
    },
    'pulses': {
        'readout_16us_pulse_res_1': {
            'operation': 'measurement',
            'length': 16000,
            'waveforms': {
                'I': 'readout_wf_res_1',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
            'integration_weights': {
                'calib_3200ns': 'calib_3200ns_weights',
                'calib_16us': 'calib_16us_weights',
                'I_cos': 'res_1_iw_I_cosine',
                'I_sin': 'res_1_iw_I_sine',
                'J_cos': 'res_1_iw_J_cosine',
                'J_sin': 'res_1_iw_J_sine',
                'K_cos': 'res_1_iw_K_cosine',
                'K_sin': 'res_1_iw_K_sine',
                'L_cos': 'res_1_iw_L_cosine',
                'L_sin': 'res_1_iw_L_sine',
            }
        },
        'readout_calib_pulse_res_1': {
            'operation': 'measurement',
            'length': 16000,
            'waveforms': {
                'I': 'readout_wf_res_1',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
            'integration_weights': {
                'calib_1000ns': '1us_weights_cosine',
                'calib_3200ns': 'calib_3200ns_weights',
                'calib_4000ns': '4us_weights_cosine',
                'calib_10us': 'calib_10us_weights',
                'calib_16us': 'calib_16us_weights'
            }
        },
        'readout_unit_pulse_res_1': {
            'operation': 'measurement',
            'length': 16000,
            'waveforms': {
                'I': 'readout_wf_res_1',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
            'integration_weights': {
                'cos': '16us_weights_cosine',
                'sin': '16us_weights_sine',
                'minus_sin': '16us_weights_minus_sine'
            }
        },
        'readout_unit_5us_pulse_res_1': {
            'operation': 'measurement',
            'length': 5000,
            'waveforms': {
                'I': 'readout_wf_res_1_5us',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
            'integration_weights': {
                'cos': '5us_weights_cosine',
                'sin': '5us_weights_sine',
                'minus_sin': '5us_weights_minus_sine'
            }
        },
        'readout_unit_4us_pulse_res_1': {
            'operation': 'measurement',
            'length': 4000,
            'waveforms': {
                'I': 'readout_wf_res_1_4us',
                'Q': 'zero_wf'
            },
             'digital_marker': 'ON',
            'integration_weights': {
                'cos': '4us_weights_cosine',
                'sin': '4us_weights_sine',
                'minus_sin': '4us_weights_minus_sine'
            }
        },
        'readout_unit_2us_pulse_res_1': {
            'operation': 'measurement',
            'length': 2000,
            'waveforms': {
                'I': 'readout_wf_res_1_2us',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
            'integration_weights': {
                'cos': '2us_weights_cosine',
                'sin': '2us_weights_sine',
                'minus_sin': '2us_weights_minus_sine'
            }
        },
        'readout_unit_1us_pulse_res_1': {
            'operation': 'measurement',
            'length': 1000,
            'waveforms': {
                'I': 'readout_wf_res_1_1us',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON_1us',
            'integration_weights': {
                'cos': '1us_weights_cosine',
                'sin': '1us_weights_sine',
                'minus_sin': '1us_weights_minus_sine'
            }
        },
        'readout_fast_pulse_res_1': {
            'operation': 'measurement',
            'length': 6400,
            'waveforms': {
                'I': 'readout_fast_wf_res_1',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
            'integration_weights': {
                'cos': 'res_1_iw_fast_cosine',
                'sin': 'res_1_iw_fast_sine',
                'minus_sin': 'res_1_iw_fast_minus_sine',
            }
        },
        'readout_16us_pulse_res_2': {
            'operation': 'measurement',
            'length': 16000,
            'waveforms': {
                'I': 'readout_wf_res_2',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
            'integration_weights': {
                'calib_3200ns': 'calib_3200ns_weights',
                'calib_16us': 'calib_16us_weights',
                'I_cos': 'res_2_iw_I_cosine',
                'I_sin': 'res_2_iw_I_sine',
                'J_cos': 'res_2_iw_J_cosine',
                'J_sin': 'res_2_iw_J_sine',
                'K_cos': 'res_2_iw_K_cosine',
                'K_sin': 'res_2_iw_K_sine',
                'L_cos': 'res_2_iw_L_cosine',
                'L_sin': 'res_2_iw_L_sine',
            }
        },
        'readout_calib_pulse_res_2': {
            'operation': 'measurement',
            'length': 16000,
            'waveforms': {
                'I': 'readout_wf_res_2',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
            'integration_weights': {
                'calib_1000ns': '1us_weights_cosine',
                'calib_3200ns': 'calib_3200ns_weights',
                'calib_4000ns': '4us_weights_cosine',
                'calib_10us': 'calib_10us_weights',
                'calib_16us': 'calib_16us_weights'
            }
        },
        'readout_unit_pulse_res_2': {
            'operation': 'measurement',
            'length': 16000,
            'waveforms': {
                'I': 'readout_wf_res_2',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
            'integration_weights': {
                'cos': '16us_weights_cosine',
                'sin': '16us_weights_sine',
                'minus_sin': '16us_weights_minus_sine'
            }
        },
        'readout_unit_5us_pulse_res_2': {
            'operation': 'measurement',
            'length': 5000,
            'waveforms': {
                'I': 'readout_wf_res_2_5us',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
            'integration_weights': {
                'cos': '5us_weights_cosine',
                'sin': '5us_weights_sine',
                'minus_sin': '5us_weights_minus_sine'
            }
        },
        'readout_unit_4us_pulse_res_2': {
            'operation': 'measurement',
            'length': 4000,
            'waveforms': {
                'I': 'readout_wf_res_2_4us',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
            'integration_weights': {
                'cos': '4us_weights_cosine',
                'sin': '4us_weights_sine',
                'minus_sin': '4us_weights_minus_sine'
            }
        },
        'readout_unit_2us_pulse_res_2': {
            'operation': 'measurement',
            'length': 2000,
            'waveforms': {
                'I': 'readout_wf_res_2_2us',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
            'integration_weights': {
                'cos': '2us_weights_cosine',
                'sin': '2us_weights_sine',
                'minus_sin': '2us_weights_minus_sine'
            }
        },
        'readout_unit_1us_pulse_res_2': {
            'operation': 'measurement',
            'length': 1000,
            'waveforms': {
                'I': 'readout_wf_res_2_1us',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON_1us',
            'integration_weights': {
                'cos': '1us_weights_cosine',
                'sin': '1us_weights_sine',
                'minus_sin': '1us_weights_minus_sine'
            }
        },
        'readout_calib_pulse_6400_res_2': {
            'operation': 'measurement',
            'length': 6400,
            'waveforms': {
                'I': 'readout_fast_wf_res_2',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
            'integration_weights': {
                'calib_6400ns': 'calib_6400ns_weights'
            }
        },
        'readout_fast_pulse_res_2_6400': {
            'operation': 'measurement',
            'length': 6400,
            'waveforms': {
                'I': 'readout_fast_wf_res_2',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
            'integration_weights': {
                'cos': 'res_2_iw_fast_cosine',
                'sin': 'res_2_iw_fast_sine',
                'minus_sin': 'res_2_iw_fast_minus_sine',
            }
        
        },
        'readout_fast_pulse_res_2_3200': {
            'operation': 'measurement',
            'length': 3200,
            'waveforms': {
                'I': 'readout_fast_wf_res_2_3200',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
            'integration_weights': {
                'cos': 'res_2_iw_fast_cosine',
                'sin': 'res_2_iw_fast_sine',
                'minus_sin': 'res_2_iw_fast_minus_sine',
            }
        
        },
        'readout_16us_pulse_res_C': {
            'operation': 'measurement',
            'length': 16000,
            'waveforms': {
                'I': 'readout_wf_res_C',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
            'integration_weights': {
                'calib_3200ns': 'calib_3200ns_weights',
                'calib_16us': 'calib_16us_weights',
                'I_cos': 'res_C_iw_I_cosine',
                'I_sin': 'res_C_iw_I_sine',
                'J_cos': 'res_C_iw_J_cosine',
                'J_sin': 'res_C_iw_J_sine',
                'K_cos': 'res_C_iw_K_cosine',
                'K_sin': 'res_C_iw_K_sine',
                'L_cos': 'res_C_iw_L_cosine',
                'L_sin': 'res_C_iw_L_sine',
            }
        },
        'readout_calib_pulse_res_C': {
            'operation': 'measurement',
            'length': 16000,
            'waveforms': {
                'I': 'readout_wf_res_C',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
            'integration_weights': {
                'calib_3200ns': 'calib_3200ns_weights',
                'calib_16us': 'calib_16us_weights'
            }
        },
        'readout_fast_pulse_res_C': {
            'operation': 'measurement',
            'length': 6400,
            'waveforms': {
                'I': 'readout_fast_wf_res_C',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
            'integration_weights': {
                'cos': 'res_C_iw_fast_cosine',
                'sin': 'res_C_iw_fast_sine',
            }
        },
        'square_pulse': {
            'operation': 'control',
            'length': 16000,
            'waveforms': {
                'I': 'square_wf',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON'
        },
        'square_pad_pulse': {
            'operation': 'control',
            'length': 16016,
            'waveforms': {
                'I': 'square_pad_wf',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON'
        },
        'square_pulse_Q1': {
            'operation': 'control',
            'length': 16000,
            'waveforms': {
                'I': 'square_wf',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON'
        },
        'x180_pulse_Q1': {
            'operation': 'control',
            'length': Q1_DL_x180_len,
            'waveforms': {
                'I': 'x180_wf_Q1',
                'Q': 'zero_wf'
            }
        },
        'x90_pulse_Q1': {
            'operation': 'control',
            'length': Q1_DL_x90_len,
            'waveforms': {
                'I': 'x90_wf_Q1',
                'Q': 'zero_wf'
            }
        },
        'x180_pulse_Q1_DL': {
            'operation': 'control',
            'length': Q1_DL_x180_len,
            'waveforms': {
                'I': 'x180_wf_Q1',
                'Q': 'zero_wf'
            }
        },
        'x90_pulse_Q1_DL': {
            'operation': 'control',
            'length': Q1_DL_x90_len,
            'waveforms': {
                'I': 'x90_wf_Q1',
                'Q': 'zero_wf'
            }
        },
        'x180_cos_pulse_Q1': {
            'operation': 'control',
            'length': x180_cos_len,
            'waveforms': {
                'I': 'x180_cos_wf_Q1',
                'Q': 'zero_wf'
            }
        },
        'x90_cos_pulse_Q1': {
            'operation': 'control',
            'length': x90_cos_len,
            'waveforms': {
                'I': 'x90_cos_wf_Q1',
                'Q': 'zero_wf'
            }
        },
        'x180_pulse_Q2': {
            'operation': 'control',
            'length': 80,
            'waveforms': {
                'I': 'x180_wf_Q2',
                'Q': 'zero_wf'
            }
        },
        'x90_pulse_Q2': {
            'operation': 'control',
            'length': 40,
            'waveforms': {
                'I': 'x90_wf_Q2',
                'Q': 'zero_wf'
            }
        },
        'x180_pulse_QC': {
            'operation': 'control',
            'length': QC_x180_len,
            'waveforms': {
                'I': 'x180_wf_QC',
                'Q': 'zero_wf'
            }
        },
        'x90_pulse_QC': {
            'operation': 'control',
            'length': QC_x90_len,
            'waveforms': {
                'I': 'x90_wf_QC',
                'Q': 'zero_wf'
            }
        },
        'x180_cos_pulse_QC': {
            'operation': 'control',
            'length': x180_cos_len,
            'waveforms': {
                'I': 'x180_cos_wf_QC',
                'Q': 'zero_wf'
            }
        },
        'x90_cos_pulse_QC': {
            'operation': 'control',
            'length': x90_cos_len,
            'waveforms': {
                'I': 'x90_cos_wf_QC',
                'Q': 'zero_wf'
            }
        },
        'x180_pulse_Q2_DL': {
            'operation': 'control',
            'length': Q2_DL_x180_len,
            'waveforms': {
                'I': 'x180_wf_Q2_DL',
                'Q': 'zero_wf'
            }
        },
        'x90_pulse_Q2_DL': {
            'operation': 'control',
            'length': Q2_DL_x90_len,
            'waveforms': {
                'I': 'x90_wf_Q2_DL',
                'Q': 'zero_wf'
            }
        },
        'x180_cos_pulse_Q1_DL': {
            'operation': 'control',
            'length': x180_cos_len,
            'waveforms': {
                'I': 'x180_cos_wf_Q1_DL',
                'Q': 'zero_wf'
            }
        },
        'x90_cos_pulse_Q1_DL': {
            'operation': 'control',
            'length': x90_cos_len,
            'waveforms': {
                'I': 'x90_cos_wf_Q1_DL',
                'Q': 'zero_wf'
            }
        },
        'x180_cos_pulse_Q2_DL': {
            'operation': 'control',
            'length': x180_cos_len,
            'waveforms': {
                'I': 'x180_cos_wf_Q2_DL_I',
                'Q': 'x180_cos_wf_Q2_DL_Q'
            }
        },
        'x90_cos_pulse_Q2_DL': {
            'operation': 'control',
            'length': x90_cos_len,
            'waveforms': {
                'I': 'x90_cos_wf_Q2_DL_I',
                'Q': 'x90_cos_wf_Q2_DL_Q'
            }
        },
        'x180_pulse_Q1_ef_DL': {
            'operation': 'control',
            'length': Q1_ef_DL_x180_len,
            'waveforms': {
                'I': 'x180_wf_Q1_ef_DL',
                'Q': 'zero_wf'
            }
        },
        'x90_pulse_Q1_ef_DL': {
            'operation': 'control',
            'length': Q1_ef_DL_x90_len,
            'waveforms': {
                'I': 'x90_wf_Q1_ef_DL',
                'Q': 'zero_wf'
            }
        },
        'x180_cos_pulse_Q1_ef_DL': {
            'operation': 'control',
            'length': x180_ef_cos_len,
            'waveforms': {
                'I': 'x180_cos_wf_Q1_ef_DL',
                'Q': 'zero_wf'
            }
        },
        'x90_cos_pulse_Q1_ef_DL': {
            'operation': 'control',
            'length': x90_ef_cos_len,
            'waveforms': {
                'I': 'x90_cos_wf_Q1_ef_DL',
                'Q': 'zero_wf'
            }
        },
        'x180_pulse_Q2_ef_DL': {
            'operation': 'control',
            'length': Q2_ef_DL_x180_len,
            'waveforms': {
                'I': 'x180_wf_Q2_ef_DL',
                'Q': 'zero_wf'
            }
        },
        'x90_pulse_Q2_ef_DL': {
            'operation': 'control',
            'length': Q2_ef_DL_x90_len,
            'waveforms': {
                'I': 'x90_wf_Q2_ef_DL',
                'Q': 'zero_wf'
            }
        },
        'x180_cos_pulse_Q2_ef_DL': {
            'operation': 'control',
            'length': x180_ef_cos_len,
            'waveforms': {
                'I': 'x180_cos_wf_Q2_ef_DL',
                'Q': 'zero_wf'
            }
        },
        'x90_cos_pulse_Q2_ef_DL': {
            'operation': 'control',
            'length': x90_ef_cos_len,
            'waveforms': {
                'I': 'x90_cos_wf_Q2_ef_DL',
                'Q': 'zero_wf'
            }
        },
        'x180_pulse_QC_ef': {
            'operation': 'control',
            'length': QC_ef_x180_len,
            'waveforms': {
                'I': 'x180_wf_QC_ef',
                'Q': 'zero_wf'
            }
        },
        'x90_pulse_QC_ef': {
            'operation': 'control',
            'length': QC_ef_x90_len,
            'waveforms': {
                'I': 'x90_wf_QC_ef',
                'Q': 'zero_wf'
            }
        },
        'x180_cos_pulse_QC_ef': {
            'operation': 'control',
            'length': x180_ef_cos_len,
            'waveforms': {
                'I': 'x180_cos_wf_QC_ef',
                'Q': 'zero_wf'
            }
        },
        'x90_cos_pulse_QC_ef': {
            'operation': 'control',
            'length': x90_ef_cos_len,
            'waveforms': {
                'I': 'x90_cos_wf_QC_ef',
                'Q': 'zero_wf'
            }
        },
        'square_pulse_pd_1C': {
            'operation': 'control',
            'length': 500,
            'waveforms': {
                'single': 'square_wf_pd_1C'
            }
        },
        'sqisw_pulse_pd_1C': {
            'operation': 'control',
            'length': 60,
            'waveforms': {
                'single': 'sqisw_wf_pd_1C'
            }
        },
        'square_pulse_pd_12': {
            'operation': 'control',
            'length': 500,
            'waveforms': {
                'single': 'square_wf_pd_12'
            }
        },
        'square_pulse_QC_FFL': {
            'operation': 'control',
            'length': QC_FFL_len,
            'waveforms': {
                'I': 'square_wf_QC_FFL_I',
                'Q': 'square_wf_QC_FFL_Q'
            }
        },
        'square_pulse_DC_QC_FFL_IQ': {
            'operation': 'control',
            'length': 1000,
            'waveforms': {
                'I': 'square_wf_DC_QC_FFL',
                'Q': 'zero_wf',
            }
        },
        'square_pulse_DC_QC_FFL': {
            'operation': 'control',
            'length': 16000,
            'waveforms': {
                'single': 'square_wf_DC_QC_FFL',
            }
        },
        'square_pulse_DC_Q2_FFL': {
            'operation': 'control',
            'length': 16000,
            'waveforms': {
                'single': 'square_wf_DC_Q2_FFL',
            }
        },
        '125MHz_pulse_Q1_ef': {
            'operation': 'control',
            'length': 16000,
            'digital_marker': '125MHz_wf_Q1_ef'
        },
        '167MHz_pulse_Q1_ef': {
            'operation': 'control',
            'length': 16000,
            'digital_marker': '167MHz_wf_Q1_ef'
        },
        '187.5MHz_pulse_Q1_ef': {
            'operation': 'control',
            'length': 16000,
            'digital_marker': '187.5MHz_wf_Q1_ef'
        },
    },
    'waveforms': {
        'zero_wf': {
            'type': 'constant',
            'sample': 0.0
        },
        'readout_fast_wf_res_1': {
            'type': 'arbitrary',
            'samples': readout_wf_res_1[:6400]
        },
        'readout_fast_wf_res_2': {
            'type': 'arbitrary',
            'samples': readout_fast_wf_res_2[:6400]
        },
        'readout_fast_wf_res_2_3200': {
            'type': 'arbitrary',
            'samples': readout_fast_wf_res_2[:3200]
        },
        'readout_fast_wf_res_C': {
            'type': 'arbitrary',
            'samples': readout_wf_res_C[:6400]
        },
        'readout_wf_res_1': {
            'type': 'arbitrary',
            'samples': readout_wf_res_1
        },
        'readout_wf_res_2': {
            'type': 'arbitrary',
            'samples': readout_wf_res_2
        },
        'readout_wf_res_1_5us': {
            'type': 'arbitrary',
            'samples': readout_wf_res_1[:5000]
        },
        'readout_wf_res_1_4us': {
            'type': 'arbitrary',
            'samples': readout_wf_res_1[:4000]
        },
        'readout_wf_res_1_2us': {
            'type': 'arbitrary',
            'samples': readout_wf_res_1[:2000]
        },
        'readout_wf_res_1_1us': {
            'type': 'arbitrary',
            'samples': readout_wf_res_1[:1000]
        },
        'readout_wf_res_2_5us': {
            'type': 'arbitrary',
            'samples': readout_wf_res_2[:5000]
        },
        'readout_wf_res_2_4us': {
            'type': 'arbitrary',
            'samples': readout_wf_res_2[:4000]
        },
        'readout_wf_res_2_2us': {
            'type': 'arbitrary',
            'samples': readout_wf_res_2[:2000]
        },
        'readout_wf_res_2_1us': {
            'type': 'arbitrary',
            'samples': readout_wf_res_2[:1000]
        },
        'readout_wf_res_C': {
            'type': 'arbitrary',
            'samples': readout_wf_res_C
        },
        'square_wf': {
            'type': 'constant',
            'sample': 0.4
        },
        'square_pad_wf': {
            'type': 'arbitrary',
            'samples': [0.0] * 16 + [0.4] * 16000
        },
        'x180_wf_Q1': {
            'type': 'constant',
            'sample': Q1_DL_x180_amp
        },
        'x90_wf_Q1': {
            'type': 'constant',
            'sample': Q1_DL_x90_amp
        },
        'x180_cos_wf_Q1': {
            'type': 'arbitrary',
            'samples': Q1_DL_x180_cos_amp * np.real(x180_cos_wf)
        },
        'x90_cos_wf_Q1': {
            'type': 'arbitrary',
            'samples': Q1_DL_x90_cos_amp * np.real(x90_cos_wf)
        },
        'x180_wf_Q2': {
            'type': 'constant',
            'sample': 0.08208
        },
        'x90_wf_Q2': {
            'type': 'constant',
            'sample': 0.08208
        },
        'x180_wf_QC': {
            'type': 'constant',
            'sample': QC_x180_amp
        },
        'x90_wf_QC': {
            'type': 'constant',
            'sample': QC_x90_amp
        },
        'x180_cos_wf_QC': {
            'type': 'arbitrary',
            'samples': QC_x180_cos_amp * np.real(x180_cos_wf)
        },
        'x90_cos_wf_QC': {
            'type': 'arbitrary',
            'samples': QC_x90_cos_amp * np.real(x90_cos_wf)
        },
        'x180_wf_Q1_DL': {
            'type': 'constant',
            'sample': Q1_DL_x180_amp
        },
        'x90_wf_Q1_DL': {
            'type': 'constant',
            'sample': Q1_DL_x90_amp
        },
        'x180_cos_wf_Q1_DL': {
            'type': 'arbitrary',
            'samples': Q1_DL_x180_cos_amp * np.real(x180_cos_wf)
        },
        'x90_cos_wf_Q1_DL': {
            'type': 'arbitrary',
            'samples': Q1_DL_x90_cos_amp * np.real(x90_cos_wf)
        },
        'x180_cos_wf_Q1_DL_I': {
            'type': 'arbitrary',
            'samples': Q1_DL_x180_cos_amp * np.real(x180_cos_wf * np.exp((2j * np.pi) * 0.0 * 1e-3 * (np.arange(x180_cos_len) - 0.5 * (x180_cos_len - 1))))
        },
        'x180_cos_wf_Q1_DL_Q': {
            'type': 'arbitrary',
            'samples': Q1_DL_x180_cos_amp * np.imag(x180_cos_wf * np.exp((2j * np.pi) * 0.0 * 1e-3 * (np.arange(x180_cos_len) - 0.5 * (x180_cos_len - 1))))
        },
        'x90_cos_wf_Q1_DL_I': {
            'type': 'arbitrary',
            'samples': Q1_DL_x90_cos_amp * np.real(x90_cos_wf * np.exp((2j * np.pi) * 0.0 * 1e-3 * (np.arange(x90_cos_len) - 0.5 * (x90_cos_len - 1))))
        },
        'x90_cos_wf_Q1_DL_Q': {
            'type': 'arbitrary',
            'samples': Q1_DL_x90_cos_amp * np.imag(x90_cos_wf * np.exp((2j * np.pi) * 0.0 * 1e-3 * (np.arange(x90_cos_len) - 0.5 * (x90_cos_len - 1))))
        },
        'x180_wf_Q2_DL': {
            'type': 'constant',
            'sample': Q2_DL_x180_amp
        },
        'x90_wf_Q2_DL': {
            'type': 'constant',
            'sample': Q2_DL_x90_amp
        },
        'x180_wf_Q2_DL_I': {
            'type': 'arbitrary',
            'samples': Q2_DL_x180_amp * np.real(np.exp((2j * np.pi) * 1.3135294516728901 * 1e-3 * (np.arange(Q2_DL_x180_len) - 0.5 * (Q2_DL_x180_len - 1))))
        },
        'x180_wf_Q2_DL_Q': {
            'type': 'arbitrary',
            'samples': Q2_DL_x180_amp * np.imag(np.exp((2j * np.pi) * 1.3135294516728901 * 1e-3 * (np.arange(Q2_DL_x180_len) - 0.5 * (Q2_DL_x180_len - 1))))
        },
        'x90_wf_Q2_DL_I': {
            'type': 'arbitrary',
            'samples': Q2_DL_x90_amp * np.real(np.exp((2j * np.pi) * 1.0030452391713387 * 1e-3 * (np.arange(Q2_DL_x90_len) - 0.5 * (Q2_DL_x90_len - 1))))
        },
        'x90_wf_Q2_DL_Q': {
            'type': 'arbitrary',
            'samples': Q2_DL_x90_amp * np.imag(np.exp((2j * np.pi) * 1.0030452391713387 * 1e-3 * (np.arange(Q2_DL_x90_len) - 0.5 * (Q2_DL_x90_len - 1))))
        },
        'x180_cos_wf_Q2_DL_I': {
            'type': 'arbitrary',
            'samples': Q2_DL_x180_cos_amp * np.real(x180_cos_wf * np.exp((2j * np.pi) * 0.6039780486608042 * 1e-3 * (np.arange(x180_cos_len) - 0.5 * (x180_cos_len - 1))))
        },
        'x180_cos_wf_Q2_DL_Q': {
            'type': 'arbitrary',
            'samples': Q2_DL_x180_cos_amp * np.imag(x180_cos_wf * np.exp((2j * np.pi) * 0.6039780486608042 * 1e-3 * (np.arange(x180_cos_len) - 0.5 * (x180_cos_len - 1))))
        },
        'x90_cos_wf_Q2_DL_I': {
            'type': 'arbitrary',
            'samples': Q2_DL_x90_cos_amp * np.real(x90_cos_wf * np.exp((2j * np.pi) * 0.41517457448854705 * 1e-3 * (np.arange(x90_cos_len) - 0.5 * (x90_cos_len - 1))))
        },
        'x90_cos_wf_Q2_DL_Q': {
            'type': 'arbitrary',
            'samples': Q2_DL_x90_cos_amp * np.imag(x90_cos_wf * np.exp((2j * np.pi) * 0.41517457448854705 * 1e-3 * (np.arange(x90_cos_len) - 0.5 * (x90_cos_len - 1))))
        },
        'x180_cos_wf_Q2_DL': {
            'type': 'arbitrary',
            'samples': Q2_DL_x180_cos_amp * np.real(x180_cos_wf)
        },
        'x90_cos_wf_Q2_DL': {
            'type': 'arbitrary',
            'samples': Q2_DL_x90_cos_amp * np.real(x90_cos_wf)
        },
        'x180_wf_Q1_ef_DL': {
            'type': 'constant',
            'sample': Q1_ef_DL_x180_amp
        },
        'x90_wf_Q1_ef_DL': {
            'type': 'constant',
            'sample': Q1_ef_DL_x90_amp
        },
        'x180_cos_wf_Q1_ef_DL': {
            'type': 'arbitrary',
            'samples': Q1_ef_DL_x180_cos_amp * np.real(x180_ef_cos_wf)
        },
        'x90_cos_wf_Q1_ef_DL': {
            'type': 'arbitrary',
            'samples': Q1_ef_DL_x90_cos_amp * np.real(x90_ef_cos_wf)
        },
        'x180_wf_Q2_ef_DL': {
            'type': 'constant',
            'sample': Q2_ef_DL_x180_amp
        },
        'x90_wf_Q2_ef_DL': {
            'type': 'constant',
            'sample': Q2_ef_DL_x90_amp
        },
        'x180_cos_wf_Q2_ef_DL': {
            'type': 'arbitrary',
            'samples': Q2_ef_DL_x180_cos_amp * np.real(x180_ef_cos_wf)
        },
        'x90_cos_wf_Q2_ef_DL': {
            'type': 'arbitrary',
            'samples': Q2_ef_DL_x90_cos_amp * np.real(x90_ef_cos_wf)
        },
        'x180_wf_QC_ef': {
            'type': 'constant',
            'sample': QC_ef_x180_amp
        },
        'x90_wf_QC_ef': {
            'type': 'constant',
            'sample': QC_ef_x90_amp
        },
        'x180_cos_wf_QC_ef': {
            'type': 'arbitrary',
            'samples': QC_ef_x180_cos_amp * np.real(x180_ef_cos_wf)
        },
        'x90_cos_wf_QC_ef': {
            'type': 'arbitrary',
            'samples': QC_ef_x90_cos_amp * np.real(x90_ef_cos_wf)
        },
        'square_wf_pd_1C': {
            'type': 'constant',
            'sample': 0.5
        },
        'sqisw_wf_pd_1C': {
            'type': 'constant',
            'sample': 0.048
        },
        'square_wf_pd_12': {
            'type': 'constant',
            'sample': 0.5
        },
        'square_wf_QC_FFL': {
            'type': 'constant',
            'sample': 0.5
        },
        'square_wf_DC_QC_FFL': {
            'type': 'constant',
            'sample': QC_FFL_amp
        },
        'square_wf_DC_Q2_FFL': {
            'type': 'constant',
            'sample': 0.5
        },
        'square_wf_QC_FFL_I': {
            'type': 'arbitrary',
            'samples': QC_FFL_amp * np.real(np.exp((2j * np.pi * (QC_FFL_freq - (Q2_freq - Q1_freq) // 2) * 1e-9) * (np.arange(QC_FFL_len) - 0.5 * (QC_FFL_len - 1))))
            # 'samples': QC_FFL_amp * np.real(np.exp((2j * np.pi * 139731 * 1e-9) * (np.arange(QC_FFL_len) - 0.5 * (QC_FFL_len - 1))))
        },
        'square_wf_QC_FFL_Q': {
            'type': 'arbitrary',
            'samples': QC_FFL_amp * np.imag(np.exp((2j * np.pi * (QC_FFL_freq - (Q2_freq - Q1_freq) // 2) * 1e-9) * (np.arange(QC_FFL_len) - 0.5 * (QC_FFL_len - 1))))
            # 'samples': QC_FFL_amp * np.imag(np.exp((2j * np.pi * 139731 * 1e-9) * (np.arange(QC_FFL_len) - 0.5 * (QC_FFL_len - 1))))
        }
    },
    'digital_waveforms': {
        'ON': {
            'samples': [(1, 0)]
        },
        'ON_1us': {
            'samples': [(1,0)]
        },
        '125MHz_wf_Q1_ef': {
            'samples': [(1, 4), (0, 4)] * 2000
        },
        '167MHz_wf_Q1_ef': {
            'samples': [(1, 3), (0, 3)] * 2666 + [(1, 3), (0, 1)]
        },
        '187.5MHz_wf_Q1_ef': {
            'samples': [(1, 3), (0, 2), (1, 3), (0, 3), (1, 2), (0, 3)] * 1000
        },
        # 'x180_wf_Q1_ef': {
        #     'samples': [(1, 3), (0, 2), (1, 3), (0, 3), (1, 2), (0, 3)] * (Q1_ef_DL_x180_len // 16)
        # },
    },
    'integration_weights': {
        'res_1_iw_I_cosine': {
            'cosine': [(val, duration) for val, duration in zip(res_1_iw_I[0], durations)],
            'sine': [(-val, duration) for val, duration in zip(res_1_iw_Q[0], durations)]
        },
        'res_1_iw_I_sine': {
            'cosine': [(val, duration) for val, duration in zip(res_1_iw_Q[0], durations)],
            'sine': [(val, duration) for val, duration in zip(res_1_iw_I[0], durations)]
        },
        'res_1_iw_I_minus_sine': {
            'cosine': [(-val, duration) for val, duration in zip(res_1_iw_Q[0], durations)],
            'sine': [(-val, duration) for val, duration in zip(res_1_iw_I[0], durations)]
        },
        'res_1_iw_J_cosine': {
            'cosine': [(val, duration) for val, duration in zip(res_1_iw_I[1], durations)],
            'sine': [(-val, duration) for val, duration in zip(res_1_iw_Q[1], durations)]
        },
        'res_1_iw_J_sine': {
            'cosine': [(val, duration) for val, duration in zip(res_1_iw_Q[1], durations)],
            'sine': [(val, duration) for val, duration in zip(res_1_iw_I[1], durations)]
        },
        'res_1_iw_K_cosine': {
            'cosine': [(val, duration) for val, duration in zip(res_1_iw_I[2], durations)],
            'sine': [(-val, duration) for val, duration in zip(res_1_iw_Q[2], durations)]
        },
        'res_1_iw_K_sine': {
            'cosine': [(val, duration) for val, duration in zip(res_1_iw_Q[2], durations)],
            'sine': [(val, duration) for val, duration in zip(res_1_iw_I[2], durations)]
        },
        'res_1_iw_L_cosine': {
            'cosine': [(val, duration) for val, duration in zip(res_1_iw_I[3], durations)],
            'sine': [(-val, duration) for val, duration in zip(res_1_iw_Q[3], durations)]
        },
        'res_1_iw_L_sine': {
            'cosine': [(val, duration) for val, duration in zip(res_1_iw_Q[3], durations)],
            'sine': [(val, duration) for val, duration in zip(res_1_iw_I[3], durations)]
        },
        'res_2_iw_I_cosine': {
            'cosine': [(val, duration) for val, duration in zip(res_2_iw_I[0], durations)],
            'sine': [(-val, duration) for val, duration in zip(res_2_iw_Q[0], durations)]
        },
        'res_2_iw_I_sine': {
            'cosine': [(val, duration) for val, duration in zip(res_2_iw_Q[0], durations)],
            'sine': [(val, duration) for val, duration in zip(res_2_iw_I[0], durations)]
        },
        'res_2_iw_I_minus_sine': {
            'cosine': [(-val, duration) for val, duration in zip(res_2_iw_Q[0], durations)],
            'sine': [(-val, duration) for val, duration in zip(res_2_iw_I[0], durations)]
        },
        'res_2_iw_J_cosine': {
            'cosine': [(val, duration) for val, duration in zip(res_2_iw_I[1], durations)],
            'sine': [(-val, duration) for val, duration in zip(res_2_iw_Q[1], durations)]
        },
        'res_2_iw_J_sine': {
            'cosine': [(val, duration) for val, duration in zip(res_2_iw_Q[1], durations)],
            'sine': [(val, duration) for val, duration in zip(res_2_iw_I[1], durations)]
        },
        'res_2_iw_K_cosine': {
            'cosine': [(val, duration) for val, duration in zip(res_2_iw_I[2], durations)],
            'sine': [(-val, duration) for val, duration in zip(res_2_iw_Q[2], durations)]
        },
        'res_2_iw_K_sine': {
            'cosine': [(val, duration) for val, duration in zip(res_2_iw_Q[2], durations)],
            'sine': [(val, duration) for val, duration in zip(res_2_iw_I[2], durations)]
        },
        'res_2_iw_L_cosine': {
            'cosine': [(val, duration) for val, duration in zip(res_2_iw_I[3], durations)],
            'sine': [(-val, duration) for val, duration in zip(res_2_iw_Q[3], durations)]
        },
        'res_2_iw_L_sine': {
            'cosine': [(val, duration) for val, duration in zip(res_2_iw_Q[3], durations)],
            'sine': [(val, duration) for val, duration in zip(res_2_iw_I[3], durations)]
        },
        'res_2_iw_L_minus_sine': {
            'cosine': [(-val, duration) for val, duration in zip(res_2_iw_Q[3], durations)],
            'sine': [(-val, duration) for val, duration in zip(res_2_iw_I[3], durations)]
        },
        'res_C_iw_I_cosine': {
            'cosine': [(val, duration) for val, duration in zip(res_C_iw_I[0], durations)],
            'sine': [(-val, duration) for val, duration in zip(res_C_iw_Q[0], durations)]
        },
        'res_C_iw_I_sine': {
            'cosine': [(val, duration) for val, duration in zip(res_C_iw_Q[0], durations)],
            'sine': [(val, duration) for val, duration in zip(res_C_iw_I[0], durations)]
        },
        'res_C_iw_J_cosine': {
            'cosine': [(val, duration) for val, duration in zip(res_C_iw_I[1], durations)],
            'sine': [(-val, duration) for val, duration in zip(res_C_iw_Q[1], durations)]
        },
        'res_C_iw_J_sine': {
            'cosine': [(val, duration) for val, duration in zip(res_C_iw_Q[1], durations)],
            'sine': [(val, duration) for val, duration in zip(res_C_iw_I[1], durations)]
        },
        'res_C_iw_K_cosine': {
            'cosine': [(val, duration) for val, duration in zip(res_C_iw_I[2], durations)],
            'sine': [(-val, duration) for val, duration in zip(res_C_iw_Q[2], durations)]
        },
        'res_C_iw_K_sine': {
            'cosine': [(val, duration) for val, duration in zip(res_C_iw_Q[2], durations)],
            'sine': [(val, duration) for val, duration in zip(res_C_iw_I[2], durations)]
        },
        'res_C_iw_L_cosine': {
            'cosine': [(val, duration) for val, duration in zip(res_C_iw_I[3], durations)],
            'sine': [(-val, duration) for val, duration in zip(res_C_iw_Q[3], durations)]
        },
        'res_C_iw_L_sine': {
            'cosine': [(val, duration) for val, duration in zip(res_C_iw_Q[3], durations)],
            'sine': [(val, duration) for val, duration in zip(res_C_iw_I[3], durations)]
        },
        'calib_3200ns_weights': {
            'cosine': [(1.0, 3200)],
            'sine': [(0.0, 3200)]
        },
        'calib_6400ns_weights': {
            'cosine': [(1.0, 6400)],
            'sine': [(0.0, 6400)]
        },
        'calib_10us_weights': {
            'cosine': [(1.0, 10000)],
            'sine': [(0.0, 10000)]
        },
        'calib_16us_weights': {
            'cosine': [(1.0, 16000)],
            'sine': [(0.0, 16000)]
        },
        '16us_weights_cosine': {
            'cosine': [(1.0, 16000)],
            'sine': [(0.0, 16000)]
        },
        '16us_weights_sine': {
            'cosine': [(0.0, 16000)],
            'sine': [(1.0, 16000)]
        },
         '16us_weights_minus_sine': {
            'cosine': [(0.0, 16000)],
            'sine': [(-1.0, 16000)]
        },
        '5us_weights_cosine': {
            'cosine': [(1.0, 5000)],
            'sine': [(0.0, 5000)]
        },
        '5us_weights_sine': {
            'cosine': [(0.0,5000)],
            'sine': [(1.0, 5000)]
        },
         '5us_weights_minus_sine': {
            'cosine': [(0.0, 5000)],
            'sine': [(-1.0, 5000)]
        },
        '4us_weights_cosine': {
            'cosine': [(1.0, 4000)],
            'sine': [(0.0, 4000)]
        },
        '4us_weights_sine': {
            'cosine': [(0.0,4000)],
            'sine': [(1.0, 4000)]
        },
         '4us_weights_minus_sine': {
            'cosine': [(0.0, 4000)],
            'sine': [(-1.0, 4000)]
        },
         '2us_weights_cosine': {
            'cosine': [(1.0, 2000)],
            'sine': [(0.0, 2000)]
        },
         '2us_weights_sine': {
            'cosine': [(0.0,2000)],
            'sine': [(1.0, 2000)]
        },
         '2us_weights_minus_sine': {
            'cosine': [(0.0, 2000)],
            'sine': [(-1.0, 2000)]
        },
         '1us_weights_cosine': {
            'cosine': [(1.0, 1000)],
            'sine': [(0.0, 1000)]
        },
         '1us_weights_sine': {
            'cosine': [(0.0,1000)],
            'sine': [(1.0, 1000)]
        },
         '1us_weights_minus_sine': {
            'cosine': [(0.0, 1000)],
            'sine': [(-1.0, 1000)]
        },

        'res_1_iw_fast_cosine': {
            'cosine': [(val, duration) for val, duration in zip(res_1_iw_I[0][:18], durations[:18])],
            'sine': [(-val, duration) for val, duration in zip(res_1_iw_Q[0][:18], durations[:18])]
        },
        'res_1_iw_fast_sine': {
            'cosine': [(val, duration) for val, duration in zip(res_1_iw_Q[0][:18], durations[:18])],
            'sine': [(val, duration) for val, duration in zip(res_1_iw_I[0][:18], durations[:18])]
        },
        'res_1_iw_fast_minus_sine': {
            'cosine': [(-val, duration) for val, duration in zip(res_1_iw_Q[0][:18], durations[:18])],
            'sine': [(-val, duration) for val, duration in zip(res_1_iw_I[0][:18], durations[:18])]
        },
        'res_2_iw_fast_cosine': {
            'cosine': [(val, duration) for val, duration in zip(res_2_iw_I[0][:18], durations[:18])],
            'sine': [(-val, duration) for val, duration in zip(res_2_iw_Q[0][:18], durations[:18])]
        },
        'res_2_iw_fast_sine': {
            'cosine': [(val, duration) for val, duration in zip(res_2_iw_Q[0][:18], durations[:18])],
            'sine': [(val, duration) for val, duration in zip(res_2_iw_I[0][:18], durations[:18])]
        },
        'res_2_iw_fast_minus_sine': {
            'cosine': [(-val, duration) for val, duration in zip(res_2_iw_Q[0][:18], durations[:18])],
            'sine': [(-val, duration) for val, duration in zip(res_2_iw_I[0][:18], durations[:18])]
        },
        'res_C_iw_fast_cosine': {
            'cosine': [(val, duration) for val, duration in zip(res_C_iw_I[0][:18], durations[:18])],
            'sine': [(-val, duration) for val, duration in zip(res_C_iw_Q[0][:18], durations[:18])]
        },
        'res_C_iw_fast_sine': {
            'cosine': [(val, duration) for val, duration in zip(res_C_iw_Q[0][:18], durations[:18])],
            'sine': [(val, duration) for val, duration in zip(res_C_iw_I[0][:18], durations[:18])]
        }
    },
   "mixers": {
        "mixer_Q1_DL": [
            {
                "correction": Q1_DL_mixer_C_matrix,
                "intermediate_frequency": if_Q1,
                "lo_frequency": lo_qubit
            }
        ],
        "mixer_Q2_DL": [
            {
                "correction": Q2_DL_mixer_C_matrix,
                "intermediate_frequency": if_Q2,
                "lo_frequency": lo_qubit
            }
        ],
        "mixer_Q1_ef_DL": [
            {
                "correction": Q1_ef_DL_mixer_C_matrix,
                "intermediate_frequency": if_Q1_ef,
                "lo_frequency": lo_qubit
            }
        ],
        "mixer_Q2_ef_DL": [
            {
                "correction": Q2_ef_DL_mixer_C_matrix,
                "intermediate_frequency": if_Q2_ef,
                "lo_frequency": lo_qubit
            }
        ],
        "mixer_readout_line": [
            {
                "correction": QC_mixer_C_matrix,
                "intermediate_frequency": if_QC,
                "lo_frequency": lo_mod
            },
            {
                "correction": QC_ef_mixer_C_matrix,
                "intermediate_frequency": if_QC_ef,
                "lo_frequency": lo_mod
            },
            {
                "correction": res_1_mixer_C_matrix,
                "intermediate_frequency": if_res_1,
                "lo_frequency": lo_res
            },
            {
                "correction": res_2_mixer_C_matrix,
                "intermediate_frequency": if_res_2,
                "lo_frequency": lo_res
            },
            {
                "correction": res_C_mixer_C_matrix,
                "intermediate_frequency": if_res_C,
                "lo_frequency": lo_res
            },
            {
                "correction": ACS_mixer_C_matrix,
                "intermediate_frequency": if_ACS,
                "lo_frequency": lo_qubit
            }
        ],
        "mixer_drive_line1": [
            {
                "correction": [
                    1,
                    0,
                    0,
                    1
                ],
                "intermediate_frequency": if_Q1,
                "lo_frequency": lo_qubit
            },
            {
                "correction": [
                    1,
                    0,
                    0,
                    1
                ],
                "intermediate_frequency": if_Q2,
                "lo_frequency": lo_qubit
            }
        ],
        "mixer_drive_line2": [
            {
                "correction": [
                    1,
                    0,
                    0,
                    1
                ],
                "intermediate_frequency": if_Q2,
                "lo_frequency": lo_qubit
            }
        ]
    }
}