# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 12:26:45 2020.

@author: P. M. Harrington
"""
import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from . import tewx
from . import expt_parameters as expt_parameters


def get_wx_address():
    instr_addr = expt_parameters.get_instrument_address("wx")
    return instr_addr


def wx_initialize():

    instr_addr = get_wx_address()

    # Initializing the instrument
    inst = tewx.TEWXAwg(instr_addr, paranoia_level=1)
    inst.send_cmd("*CLS")  # Clear errors
    # inst.send_cmd('*RST') # Reset the device
    inst.send_cmd(":FREQ:RAST 1000000001.000000", paranoia_level=1)
    inst.send_cmd(":FREQ:RAST 1000000000.000000", paranoia_level=1)

    #
    syst_err = inst.send_query(":SYST:ERR?")
    print(syst_err)
    inst.close()


def wx_set_and_amplitude_and_offset(
    amp=[1.5, 1.5, 1.5, 1.5], offset=[0, 0, 0, 0]
):  # offset=[-.052,+.053,0,0]
    #                                    offset=[-0.062, 0.016 , 0.09, -0.086]): #offset=[-.052,+.053,0,0]
    # offset=[0., 0., 0.0105, -0.083]
    # offset=[-.052,+.053, 0.057, -0.061]
    #    [0., 0., 0.0105, -0.083]
    # offset=[0., 0., 0.057, -0.061]
    """
    DESCRIPTION: sets the amplitudes and offsets of each WX channel
    """

    instr_addr = get_wx_address()

    # Initializing the instrument
    inst = tewx.TEWXAwg(instr_addr, paranoia_level=1)
    #    inst = tewx.TEWXAwg("128.252.134.15", paranoia_level=1)
    # Setting up amplitudes and offsets
    #    amp = [amp12, amp12, 1.5, 1.5]
    #
    for ch_index in range(4):
        inst.send_cmd(":INST:SEL {0}".format(ch_index + 1))
        inst.send_cmd(":VOLT {}".format(amp[ch_index]))
        inst.send_cmd(":VOLT:OFFS {}".format(offset[ch_index]))

    # query system error
    syst_err = inst.send_query(":SYST:ERR?")
    print(syst_err)
    inst.close()

    #
    # set_marker_level(which_channel='ch1ch2', which_marker=1, marker_voltage_level = 1.2)
    # set_marker_level(which_channel='ch3ch4', which_marker=2, marker_voltage_level = 1.2)
    set_marker_level(which_channel="ch1ch2", which_marker=1, marker_voltage_level=0.5)
    set_marker_level(which_channel="ch3ch4", which_marker=2, marker_voltage_level=0.5)


def set_run_mode_continuous():

    instr_addr = get_wx_address()

    # Initializing the instrument
    inst = tewx.TEWXAwg(instr_addr, paranoia_level=1)

    #
    inst.send_cmd("INIT:CONT:STAT ON")

    # query system error
    syst_err = inst.send_query(":SYST:ERR?")
    print(syst_err)
    inst.close()


def set_marker_level(which_channel="ch1ch2", which_marker=2, marker_voltage_level=1.2):
    #
    instr_addr = get_wx_address()
    print(instr_addr)
    # Initializing the instrument
    inst = tewx.TEWXAwg(instr_addr, paranoia_level=1)

    if which_channel == "ch1ch2":
        inst.send_cmd(":INST:SEL {0}".format(0 + 1))
    elif which_channel == "ch3ch4":
        inst.send_cmd(":INST:SEL {0}".format(2 + 1))

    inst.send_cmd("MARK:SEL {}".format(which_marker))
    inst.send_cmd("MARK:VOLT:LEV:HIGH {}".format(marker_voltage_level))

    # query system error
    syst_err = inst.send_query(":SYST:ERR?")
    print(syst_err)
    inst.close()


if __name__ == "__main__":
    pass
