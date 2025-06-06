# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 09:53:38 2020.

@author: J. Monroe
"""
import numpy as np
import pyvisa

rm = pyvisa.ResourceManager()


def gigatronics_bnc_output(
    freq_GHz, power_dBm, bnc_addr="USB0::0x03EB::0xAFFF::141-216340000-0292::INSTR"
):
    freq_string = str(freq_GHz)
    pow_string = str(power_dBm)
    gigadict = {
        "1": "D1",
        "2": "D2",
        "3": "D3",
        "4": "D4",
        "5": "D5",
        "6": "D6",
        "7": "D7",
        "8": "D8",
        "9": "D9",
        "0": "D0",
        ".": "DA",
        "FREQ": "02",
        "GHZ": "DC",
        "dBm": "DC",
        "power": "40",
        "-": "DB",
    }

    bnc_handle = rm.open_resource(bnc_addr)

    # Writing frequency to gigatronics
    giga_write = "KK " + gigadict["FREQ"] + " "
    for i in range(len(freq_string)):
        giga_write += gigadict[freq_string[i]] + " "
    giga_write += gigadict["GHZ"]
    print(" writing to gigatronics", giga_write)

    bnc_handle.write_raw(giga_write)

    # Writing power (dBm) to gigatronics
    giga_write_pow = "KK " + gigadict["power"] + " "
    for i in range(len(pow_string)):
        giga_write_pow += gigadict[pow_string[i]] + " "
    giga_write_pow += gigadict["dBm"]
    print(" writing to gigatronics", giga_write_pow)

    bnc_handle.write_raw(giga_write_pow)

    bnc_handle.close()


def set_bnc_output(
    freq_GHz, power_dBm, bnc_addr="USB0::0x03EB::0xAFFF::141-216340000-0292::INSTR"
):
    try:
        bnc_handle = rm.open_resource(bnc_addr)
        bnc_handle.write(f"sour:freq {freq_GHz*1E9}")
        bnc_handle.write(f"sour:pow {power_dBm}")
    finally:
        bnc_handle.close()


##END set_bnc_output
def output_sinusoidal_signal(
    ssm_freq, amp, phase, duration, bnc_addr="GPIB0::3::INSTR"
):
    try:
        bnc_handle = rm.open_resource(bnc_addr)
        bnc_handle.write("AWGControl:FG1:FUNC SIN")
        bnc_handle.write("AWGControl:FG:FREQ {ssm_freq}GHz")
        bnc_handle.write("AWGControl:FG:PHAS {phase}DEG")
        bnc_handle.write("Source1:VOLT:LEV:IMM:AMP {amp}mV")
        # bnc_handle.write(f"sour:freq {ssm_freq*1E9}")
        # bnc_handle.write(f"sour:pow {amp}")
    finally:
        bnc_handle.close()


def set_noise_voltage(
    noise, bnc_addr="USB0::0x03EB::0xAFFF::141-216340000-0292::INSTR"
):
    try:
        bnc_handle = rm.open_resource(bnc_addr)
        bnc_handle.write(f"appl:nois DEF,{noise} dBm,0.00")
    finally:
        bnc_handle.close()


def get_bnc_freq_GHz(bnc_addr="USB0::0x03EB::0xAFFF::141-216340000-0292::INSTR"):

    try:
        bnc_handle = rm.open_resource(bnc_addr)
        freq = bnc_handle.query(f"sour:freq?")
        return np.float(freq) / 1e9
    finally:
        bnc_handle.close()


##END get_bnc_freq


def configure_for_sweep(
    start_freq_GHz,
    stop_freq_GHz,
    power_dBm,
    dwell_sec=0.05,
    num_freq_points=201,
    bnc_addr="USB0::0x03EB::0xAFFF::421-4385A0002-0619::INSTR",
):
    try:
        bnc_handle = rm.open_resource(bnc_addr)

        print("Configuring BNC for sweep")
        # 0. Rest instrument
        bnc_handle.write("*CLS")
        print("0, ", bnc_handle.query(":SYST:ERR?"))

        # 1. configure RF out
        bnc_handle.write(f"sour:freq {start_freq_GHz*1E9}")
        bnc_handle.write(f"sour:powe {power_dBm}")
        print("1, ", bnc_handle.query(":SYST:ERR?"))

        # 2. configure RF sweep
        bnc_handle.write(":SWE:DEL 0")
        bnc_handle.write(":SWE:DIR UP;  :SWE:SPAC LIN;  :SWE:COUN 1.0")
        bnc_handle.write(f":SWE:DWEL {dwell_sec};  :SWE:POIN {num_freq_points}")
        print("2, ", bnc_handle.query(":SYST:ERR?"))

        # 3. configure to trigger internally
        bnc_handle.write(":TRIG:TYPE NORM;  :TRIG:SLOP POS;  :TRIG:SOUR IMM;")
        bnc_handle.write(
            ":TRIG:DEL  0.0;   :TRIG:ECO 1.0"
        )  # last command: "run (E)very Nth count"
        bnc_handle.write(":TRIG:OUTP:MODE POIN;   :TRIG:OUTP:POL NORM;")
        print("3, ", bnc_handle.query(":SYST:ERR?"))

        # 4. setup trigger to VNA (Low Frequency Output)
        bnc_handle.write(":LFO:STAT ON;  :LFO:SOUR TRIG;  :LFO:SHAP SQU;")
        bnc_handle.write(":LFO:FREQ 0.0;  :LFO:AMPL 1.0")
        print("4, ", bnc_handle.query(":SYST:ERR?"))

        # 5. setup frequency sweep
        bnc_handle.write("INIT:CONT OFF;  :FREQ:MODE FIX;  :FREQ:MODE SWE")
        bnc_handle.write(
            f":FREQ:STAR {start_freq_GHz*1E9};  :FREQ:STOP {stop_freq_GHz*1E9}"
        )
        print("5, ", bnc_handle.query(":SYST:ERR?"))

        # 6. prepare to run
        bnc_handle.write(":OUTP ON")
        bnc_handle.write("INIT:CONT OFF")  # do not repeat sweep
        bnc_handle.write(":INIT")  # arm trigger
        print("6, ", bnc_handle.query(":SYST:ERR?"))

    finally:
        bnc_handle.close()


def turn_on_and_off_bnc_output(
    bnc_addr="USB0::0x03EB::0xAFFF::421-4385A0002-0619::INSTR", RF_out="on"
):
    bnc_handle = rm.open_resource(bnc_addr)
    try:
        if RF_out == "on":
            bnc_handle.write(":OUTPut:STATe ON")
            # print("RF output turned off.")
        if RF_out == "off":
            bnc_handle.write(":OUTPut:STATe OFF")
            # print("RF output turned off.")
    finally:
        bnc_handle.close()


# Is this the right address?
def set_DC_output(
    bnc_addr="USB0::0x03EB::0xAFFF::421-4385A0002-0619::INSTR", voltage=0.022
):
    bnc_handle = rm.open_resource(bnc_addr)
    try:
        bnc_handle.write(f"APPL:DC DEF, DEF, {voltage}")
    finally:
        bnc_handle.close()


##END configure_for_Sweep
#
#    bnc_handle.write("*CLS;")
#    print("1, ", bnc_handle.query(":SYST:ERR?") )
#    bnc_handle.write(":FREQ 4.000000 GHZ;:POW -15.000000;")
#    print("2, ", bnc_handle.query(":SYST:ERR?") )
#    bnc_handle.write(":SWE:DEL  0.000000;:SWE:DWEL 0.050000;:SWE:DIR UP;:SWE:SPAC LIN;:SWE:POIN 251.000000;:SWE:COUN 1.000000")
#    print("3, ", bnc_handle.query(":SYST:ERR?") )
#    bnc_handle.write(":TRIG:TYPE NORM;:TRIG:SLOP POS;:TRIG:SOUR IMM;:TRIG:DEL 0.000000;:TRIG:ECO 1.000000")
#    print("4, ", bnc_handle.query(":SYST:ERR?") )
#    bnc_handle.write(":TRIG:OUTP:MODE POIN;:TRIG:OUTP:POL NORM;")
#    print("5, ", bnc_handle.query(":SYST:ERR?") )
#    bnc_handle.write(":LFO:STAT ON;:LFO:SOUR TRIG;:LFO:SHAP SQU;:LFO:FREQ 0.000000;:LFO:AMPL 1.000000")
#    print("6, ", bnc_handle.query(":SYST:ERR?") )
#    bnc_handle.write(":OUTP ON;")
#    print("7, ", bnc_handle.query(":SYST:ERR?") )
#    bnc_handle.write("INIT:CONT OFF;:FREQ:MODE FIX;:FREQ:STAR 4.000000 GHZ;:FREQ:STOP 4.500000 GHZ;:FREQ:MODE SWE")
#    print("8, ", bnc_handle.query(":SYST:ERR?") )
#    bnc_handle.write("INIT:CONT OFF;:INIT")
#    print("9, ", bnc_handle.query(":SYST:ERR?") )
