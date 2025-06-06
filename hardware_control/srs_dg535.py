# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 12:22:49 2020.

@author: P. M. Harrington, 25 January 2020
"""

import numpy as np
import pyvisa
import matplotlib.pyplot as plt

# import time


def set_trigger_level(low_or_high_string=""):

    #
    inst = pyvisa.ResourceManager().open_resource("GPIB0::15::INSTR")

    if low_or_high_string == "low":
        inst.write("tz2,0;om2,3;op2,1;oa2,-0.1;oo2,-0.1;")
    elif low_or_high_string == "high":
        inst.write("tz2,0;om2,3;op2,1;oa2,1.1;oo2,-0.1;")

    inst.close()


if __name__ == "__main__":
    pass
