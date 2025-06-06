readout_dict = {
    "RO_LO": 6.6247,
    "RO_LO_34":6.4804,
    "RO_LO_45":6.3804,
    "RO_LO_pwr": 16,
    "ro_dur": 5000,
}
swap_gate = {
    "swap_freq": 0,
    "swap_amp": 0,
    "swap_time": 0,
}
q1_dict = {
    "qubit_id": "q1",
    "ro_freq": 6.72734,
    "ge_ssm": -0.117,
    "ef_ssm": -0.259,#-0.2568,#-0.2581,#
    "ge_time": 77,
    "ef_time": 46,
    "ef_half_time": 25,
    "ef_half_amp": 1.3,
    "ge_amp": 1,#.5
    "ef_amp":1.5,
    "IQ_angle":60,
    "ro_amp": .25,#0.2#.35,c:\Users\quantum1\OneDrive\Documents\Python Scripts\Important Blue Fridge Python Files\New\nonlinear_QM\standard_sequences\parametric_coupling.py
    "qubit_thr": [-10000, -600],
    "mixer_offset_ge":2.5,
    "mixer_offset_ef":6,
}
q2_dict = {
    "qubit_id": "q2",
    "ro_freq": 6.6554,
    "ge_ssm": -0.155,
    "ef_ssm": -0.2962,
    "ge_time": 62,
    "ge_amp": 1,
    "ef_amp": 1,
    "IQ_angle": 25,
    "ro_amp": 0.6,
    "qubit_thr": [-10000, 1900],
    "mixer_offset_ge":2.5,
}
q3_dict = {
    "qubit_id": "q3",
    "ro_freq": 6.58374,
    "ge_ssm": -0.106,
    "ge_time": 75,
    "ge_amp": 1,
    "ef_amp": 1,
    "IQ_angle": 25,
    "ro_amp": .7,#5,
    "qubit_thr": [-10000, 1900],
    "mixer_offset_ge":2.5,
}
q4_dict = {
    "qubit_id": "q4",
    "ro_freq": 6.51117,
    "ge_ssm": -0.252,#rolo=4.4
    "ge_time": 68,
    "ge_amp": 1,
    "ef_amp": 1,
    "IQ_angle": 25,
    "ro_amp": .7,#5,
    "qubit_thr": [-10000, 1900],
    "mixer_offset_ge":2.5,
}
q5_dict = {
    "qubit_id": "q5",
    "ro_freq": 6.44418,
    "ge_ssm": -0.087,
    "ge_time": 59,
    "ge_amp": 1,
    "ef_amp": 1,
    "IQ_angle": 25,
    "ro_amp": .5,#5,
    "qubit_thr": [-10000, 1900],
    "mixer_offset_ge":2.5,
}

general_vals_dict = {
    "mixer_offset": 0,
    "mixer_offset_ef": 20,
    "wx_amps": [1,1,1.7,1.015],  # maximum 1.9
    "coupler_off_value": 0.7,
    "wx_offs": [-0.036, 0, 0, -0.098],
    "qubit_bnc": 4.6,
    "qubit_bnc_34": 4.4,
    "qubit_bnc_45": 4.25,
    "TWPA_freq":5.1,
    "TWPA_amp": -4.4,

}
bnc_address = {
    "target_bnc_black": "GPIB0::19::INSTR",
    "big_agilent": "GPIB0::11::INSTR",
    "agilent_function_generator": "GPIB0::30::INSTR",
    "target_bnc_6": "USB0::0x03EB::0xAFFF::411-433500000-0753::INSTR",
    "wx_address": "10.225.208.204",
    "TWPA_address":"USB0::0x03EB::0xAFFF::471-43A6D0000-1458::INSTR"

}
