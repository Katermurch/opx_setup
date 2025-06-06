# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 11:23:50 2020

@author: P. M. Harrington
"""

import numpy as np
import pyvisa
import time

# address = "GPIB0::24::INSTR"
rm = pyvisa.ResourceManager()
# inst = pyvisa.ResourceManager().open_resource(address)

#inst.write(":SYST:BEEP:STAT OFF")

def set_current(address, set_value_mA, step_size_mA=0.0001):
    #try:
    inst = pyvisa.ResourceManager().open_resource(address)
    value_stop = 1e-3*set_value_mA
    value_start = measure_current(address) #np.round(measure_current(), 3+2)
    num_steps = int(np.ceil(abs((value_stop-value_start)/(1e-3*step_size_mA))))
    value_sweep = np.linspace(value_start, value_stop, num_steps)
    
    #
    for v in value_sweep:
        str_cmd = "SOUR:FUNC CURR;:SOUR:CURR "+"{}".format(v)+";:VOLT:PROT 21;"
        inst.write(str_cmd)
        time.sleep(0.1) #KWM ADDED THIS SMALL DELAY
        
    mA_start = 1e3*value_start
    mA_end = 1e3*measure_current(address)
    return mA_start, mA_end
    #finally:
   #inst.close() 
        
def measure_current(address):
    #try:
    inst = pyvisa.ResourceManager().open_resource(address)
    inst.write(':SENS:FUNC "CURR"')
    inst.timeout = 5000  # kWM Set timeout to 5 seconds (adjust as needed) 
    vals = inst.query_ascii_values(":READ?")

    value = vals[1]
    print("Keithley current: {:.5f} mA".format(1e3*value))
    return value
    #finally:
    #inst.close()

def set_voltage(address,set_value_V, step_size_mV=100):
    #try:
    inst = pyvisa.ResourceManager().open_resource(address)
    value_stop = 1e-3*set_value_V
    value_start = measure_voltage() #np.round(measure_current(), 3+2)
    num_steps = int(np.ceil(abs((value_stop-value_start)/(1e-3*step_size_mV))))
    value_sweep = np.linspace(value_start, value_stop, num_steps)
    
    for v in value_sweep:
        str_cmd = "SOUR:FUNC VOLT;:SOUR:VOLT "+"{}".format(v)+";:CURR:PROT 1000;"
        inst.write(str_cmd)
        
    mV_start = 1e3*value_start
    mV_end = 1e3*measure_voltage(address)
    return mV_start, mV_end
    #finally: 
    #inst.close()
        
def measure_voltage(address):
    #try:
    inst = pyvisa.ResourceManager().open_resource(address)
    inst.write(':SENS:FUNC "VOLT"')
    vals = inst.query_ascii_values(":READ?")

    value = vals[0]
    print("Keithley voltage: {:.5f} mV".format(1e3*value))
    return value
    #finally:
    #inst.close()
    
# if __name__=="__main__":
    
    
    
#     pass  
#     inst.close()  #kwm mod 