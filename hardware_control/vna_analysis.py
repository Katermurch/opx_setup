# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 10:47:41 2020

@author: J. Monroe
@date: Feb 12, 2020
"""
import numpy as np
import time

import pyvisa
rm = pyvisa.ResourceManager()

class Smith_data():
    
    def __init__(self, vna_smith_data, f_min=None, f_max=None):
        self.real = np.array( vna_smith_data[0::2] )
        num_points = self.real.size
        self.imag = np.array( vna_smith_data[1::2] )
        self.lin_mag = np.sqrt(self.real**2 + self.imag**2)
        self.log_mag = 20*np.log10( self.lin_mag)
        # arctan2 chooses the quadrant based on the sign of each step
        self.wrapped_phase = np.arctan2(self.imag, self.real)
        
        ## unwrap the phase
        unwrapped = np.unwrap(self.wrapped_phase)
        # a quick linear fit to subtract off electrical delay. 
        # calculating physical ED requires frequency information. I'll instead
        #   keep this function robust.    
        xs = np.arange(num_points)
        # solve y = mx +b as  y = A.f where A is Nx2 with x as first  column and 
        # ones as second column f is 2x1 of coefficients m and b.
        # fit by minimizing (data-A.f)^2
        A_mat = np.array([xs, np.ones(xs.shape)]).T
        m, b = np.linalg.lstsq(A_mat, unwrapped, rcond=None)[0]
        
        self.phase_rad = unwrapped - (m*xs + b)
        self.phase_deg = self.phase_rad/np.pi*180
              
        if f_min and f_max:
            self.freqs = np.linspace(f_min, f_max, num_points)
        else:
            self.freqs = None    
        
    ##END __init__
##END Smith_data
        

def get_smith_data(restart_average=False):
    '''
    DESCRIPTION: Talk to VNA instrument to extract smith data
    INPUT: active visa session handle for VNA
    OUTPUT: Smith_data object
    '''
    # note that "fdata" returns the same as sdata, but the even-index values are 
    #   whatever is displayed on screen and odd-index values are 0
    
    vna_handle = rm.open_resource('GPIB0::16::INSTR')
    try:
        if restart_average:
            # start averaging by toggling the "Avg Trigger" setting. Upon next trigger
            # (assume internal), the average will restart
            vna_handle.write("sens:aver 0"); 
            vna_handle.write("sens:AVER 1")
            ##TODO: WAIT FOR COLLECtion to end?
            num_avgs = float( vna_handle.query("sens1:aver:coun?") )
            sweep_time = float(vna_handle.query("sense1:swe:time?") )
            delay_time = sweep_time*num_avgs
            print("VNA waiting {:.2f} seconds".format(delay_time)) 
            time.sleep(delay_time)
            
        ## a quick hack to allow data to come off
        vna_handle.write(":calc1:par:def \"FOOO1\",S21")   #modified by Chandra for VNA in 104; Change: S12 -> S21
        vna_handle.write(":calc1:par:sel \"FOOO1\"")
        vna_handle.write(":FORM:DATA ascii")    
        
        ## offload data
        string_data = vna_handle.query('calc1:data? sdata')
        numerical_data = np.array(string_data.split(','), dtype='float')
    
        ## get frequency range
        f_start = float( vna_handle.query('SENS1:FREQ:start?') )
        f_stop = float( vna_handle.query('SENS1:FREQ:stop?') )
        
        ## parse data
        smith_data = Smith_data(numerical_data, f_min=f_start, f_max=f_stop)
        return smith_data
    finally:
        vna_handle.close()
##END get_smith_data

  
def prepare_for_cw_spec(vna_freq_GHz, vna_power_dBm, dwell_sec=0.05, num_freq=201,\
                        vna_gpib='GPIB0::16::INSTR'):
    bw_hz = 1/dwell_sec
    try:
        vna_handle = rm.open_resource(vna_gpib)
        
#        # save VNA settings to be reloaded after sweep
#        vna_handle.write(':MMEM:STOR "recent_state.sta"')
    
        # setup source to single frequency
        vna_handle.write(f":SOUR:POW {vna_power_dBm}")
        vna_handle.write(f":SENSE:FREQ:STAR {vna_freq_GHz*1E9}")
        vna_handle.write(f":SENSE:FREQ:STOP {vna_freq_GHz*1E9}")
        
        # configure sweep        
        vna_handle.write(f":SENSE:BAND {bw_hz}")
        vna_handle.write(":SENS:SWE:TYPE LIN;  GEN ANAL;  TIME:AUTO ON;")
        vna_handle.write(f":SENS:SWE:POIN {num_freq}")
        vna_handle.write(":SENS:AVER ON;  AVER:COUN 1")
        
        # configure trigger
        vna_handle.write(":TRIG:SOUR EXT")
        # ideally, we would also set: trigger scope to "channel";
        #   trigger mode to point
        #vna_handle.write(":TRIG:POIN ON") # when trigger received, take single point
        #   and external source to I/O trig IN
        # reference: http://na.support.keysight.com/vna/help/latest/S1_Settings/Trigger.htm#state_point
        #   but programming docs for :TRIG:POIN indicate that using external source suffices
    finally:
        vna_handle.close()
        # this command would restore settings, but we first need to take data
        #   perhaps we should make another function called restore_settings() or something...
        #vna_handle.write(':MMEM:LOAD "recent_state.sta"')
##END prepare_for_spec
    