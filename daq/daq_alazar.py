# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 12:43:51 2020.

@author: P. M. Harrington, 25 January 2020
"""

from __future__ import division
import ctypes
import numpy as np
import os, sys
import time
import hardware_control.atsapi as ats

# import dg535_control
from tqdm import tqdm

# gen_path = r"C:\Users\Crow108\Documents\Python Scripts\sequence_generator"
# if gen_path not in sys.path:
#    sys.path.append(gen_path)
##import generator
# ctrl_path = r"C:\Users\Crow108\Documents\Python\controller"
# if ctrl_path not in sys.path: sys.path.append(ctrl_path)
# import instruments


print("define class...")


class Nop:
    def __init__(self):
        self.name = None
        pass


# def start_from(step_index=0):
#    driver = instruments.load('proteus8ch')
#    driver.start_from(step_index)

print("get params...")


def get_alazar_parameters(daq_params=None, verbose=True):
    alazar_params = Nop()

    #
    alazar_params.post_trigger_samples = 8192
    alazar_params.samples_per_sec = 1e9
    alazar_params.buffer_count = 64  # 64 for ATS9870, found in SDK manual

    #
    alazar_params.num_total_records = (
        daq_params.num_patterns * daq_params.num_records_per_pattern
    )
    alazar_params.records_per_buffer = min(1024, alazar_params.num_total_records)
    alazar_params.samples_per_buffer = (
        alazar_params.post_trigger_samples * alazar_params.records_per_buffer
    )
    alazar_params.buffers_per_acquisition = int(
        np.ceil(alazar_params.num_total_records / alazar_params.records_per_buffer)
    )

    if verbose:
        print("Patterns: {}".format(daq_params.num_patterns))
        print("Records per pattern: {}".format(daq_params.num_records_per_pattern))
        print(
            "Buffers per acquistion: {}".format(alazar_params.buffers_per_acquisition)
        )
        print("DAQ samples per record: {}".format(alazar_params.post_trigger_samples))

    return alazar_params


def rotate_iq(angle_deg=0.0, rec=[None, None]):
    ## this function rotates the I/Q outcomes to aid in thresholding
    #   the matrix rec is usually large (eg 1024 x 8192), so this step
    #   can bog down the acquisition time significantly.

    if angle_deg == 0:
        return rec

    rec_rotated = [None, None]

    ch_cmplx = rec[0] + 1j * rec[1]
    ch_cmplx_rot = (
        abs(ch_cmplx)
        * np.exp(1j * np.angle(ch_cmplx))
        * np.exp(1j * np.pi * angle_deg / 180)
    )
    rec_rotated[0] = np.real(ch_cmplx_rot)
    rec_rotated[1] = np.imag(ch_cmplx_rot)

    return rec_rotated


# Configures a board for acquisition
def configure_board(alazar_params, board):
    # Select clock parameters as required to generate this
    # sample rate
    #
    # For example: if samples_per_sec is 100e6 (100 MS/s), then you can
    # either:
    #  - select clock source INTERNAL_CLOCK and sample rate
    #    SAMPLE_RATE_100MSPS
    #  - or select clock source FAST_EXTERNAL_CLOCK, sample rate
    #    SAMPLE_RATE_USER_DEF, and connect a 100MHz signal to the
    #    EXT CLK BNC connector
    samples_per_sec = alazar_params.samples_per_sec  # 1000000000.0
    board.setCaptureClock(
        ats.INTERNAL_CLOCK, ats.SAMPLE_RATE_1000MSPS, ats.CLOCK_EDGE_RISING, 0
    )
    # Select channel A input parameters as required.
    board.inputControlEx(
        ats.CHANNEL_A,
        ats.AC_COUPLING,  # DVK changed it from DC_Coupling for Heterodyne
        ats.INPUT_RANGE_PM_400_MV,  # DVK change it from 2 V 10/14/2022 _ JTM changed from 100 mV, 20/07/14
        ats.IMPEDANCE_50_OHM,
    )

    # Select channel A bandwidth limit as required.
    board.setBWLimit(ats.CHANNEL_A, 0)

    # Select channel B input parameters as required.
    board.inputControlEx(
        ats.CHANNEL_B,
        ats.AC_COUPLING,
        ats.INPUT_RANGE_PM_400_MV,  # JTM changed from 100 mV, 20/07/14
        ats.IMPEDANCE_50_OHM,
    )
    # Select channel B bandwidth limit as required.
    board.setBWLimit(ats.CHANNEL_B, 0)

    # Select trigger inputs and levels as required.
    board.setTriggerOperation(
        ats.TRIG_ENGINE_OP_J,
        ats.TRIG_ENGINE_J,  # engine1
        ats.TRIG_EXTERNAL,  # source1
        ats.TRIGGER_SLOPE_POSITIVE,  # slope1
        129,  # level1  # was 135
        ats.TRIG_ENGINE_K,
        ats.TRIG_DISABLE,
        ats.TRIGGER_SLOPE_POSITIVE,
        128,
    )
    #    # Select external trigger parameters as required.
    #    board.setExternalTrigger(ats.DC_COUPLING,
    #                             ats.ETR_1V)
    # Set trigger delay as required.
    triggerDelay_sec = 0
    triggerDelay_samples = int(triggerDelay_sec * samples_per_sec + 0.5)
    board.setTriggerDelay(triggerDelay_samples)

    # Set trigger timeout as required.
    #
    # NOTE: The board will wait for a for this amount of time for a
    # trigger event.  If a trigger event does not arrive, then the
    # board will automatically trigger. Set the trigger timeout value
    # to 0 to force the board to wait forever for a trigger event.
    #
    # IMPORTANT: The trigger timeout value should be set to zero after
    # appropriate trigger parameters have been determined, otherwise
    # the board may trigger if the timeout interval expires before a
    # hardware trigger event arrives.
    triggerTimeout_sec = 0
    triggerTimeout_clocks = int(triggerTimeout_sec / 10e-6 + 0.5)
    board.setTriggerTimeOut(triggerTimeout_clocks)

    # Configure AUX I/O connector as required
    board.configureAuxIO(ats.AUX_OUT_TRIGGER, 0)


def acquire_data(daq_params, alazar_params, board, verbose=True):
    rec_avg_all = []
    rec_readout = []

    # No pre-trigger samples in NPT mode
    # print("\ndaq_alazar Troubleshoot stuck at this step 1")
    preTriggerSamples = 0

    # Select the number of samples per record.
    # print("\ndaq_alazar Troubleshoot stuck at this step 2")
    post_trigger_samples = alazar_params.post_trigger_samples

    # Select the number0 of records per DMA buffer.
    # print("\ndaq_alazar Troubleshoot stuck at this step 3")
    records_per_buffer = alazar_params.records_per_buffer  # 2**10 # up to 2**14

    # Select the number of buffers per acquisition.
    # print("\ndaq_alazar Troubleshoot stuck at this step 4")
    buffers_per_acquisition = alazar_params.buffers_per_acquisition

    # print("\ndaq_alazar Troubleshoot stuck at this step 5")
    records_per_acquisition = records_per_buffer * buffers_per_acquisition

    # Select the active channels.
    # print("\ndaq_alazar Troubleshoot stuck at this step 6")
    channels = ats.CHANNEL_A | ats.CHANNEL_B
    channelCount = 0
    # print("A")
    # print("\ndaq_alazar Troubleshoot stuck at this step 7")
    for c in ats.channels:
        channelCount += c & channels == c

    # Should data be saved to file?
    saveData = False
    dataFile = None
    # print("\ndaq_alazar Troubleshoot stuck at this step 8")
    if saveData:
        dataFile = open(os.path.join(os.path.dirname(__file__), "data.bin"), "wb")
    # print("\ndaq_alazar Troubleshoot stuck at this step 9")
    # Compute the number of bytes per record and per buffer
    memorySize_samples, bitsPerSample = board.getChannelInfo()
    bytesPerSample = (bitsPerSample.value + 7) // 8
    samplesPerRecord = preTriggerSamples + post_trigger_samples
    bytesPerRecord = bytesPerSample * samplesPerRecord
    bytesPerBuffer = bytesPerRecord * records_per_buffer * channelCount

    # Select number of DMA buffers to allocate
    buffer_count = alazar_params.buffer_count

    # Allocate DMA buffers
    # print("\ndaq_alazar Troubleshoot stuck at this step 10")
    sample_type = ctypes.c_uint8
    # print("B")
    if bytesPerSample > 1:
        sample_type = ctypes.c_uint16

    buffers = []
    for i in range(buffer_count):
        buffers.append(ats.DMABuffer(board.handle, sample_type, bytesPerBuffer))
    # print("\ndaq_alazar Troubleshoot stuck at this step 11")
    # Set the record size
    board.setRecordSize(preTriggerSamples, post_trigger_samples)
    # print("C")
    # Configure the board to make an NPT AutoDMA acquisition
    board.beforeAsyncRead(
        channels,
        -preTriggerSamples,
        samplesPerRecord,
        records_per_buffer,
        records_per_acquisition,
        ats.ADMA_EXTERNAL_STARTCAPTURE | ats.ADMA_NPT,
    )

    # print("\ndaq_alazar Troubleshoot stuck at this step 12")
    index_avg_start = daq_params.readout_start
    index_avg_end = daq_params.readout_start + daq_params.readout_duration - 1

    #    index_ch = [None]*2
    #    index_ch[0] = np.arange(0,post_trigger_samples*records_per_buffer) # channel A
    #    index_ch[1] = post_trigger_samples*records_per_buffer + np.arange(0,post_trigger_samples*records_per_buffer) # channel B
    # print("\ndaq_alazar Troubleshoot stuck at this step 13")
    # print("D")
    chA_indices = np.arange(0, post_trigger_samples * records_per_buffer)
    chB_indices = post_trigger_samples * records_per_buffer + np.arange(
        0, post_trigger_samples * records_per_buffer
    )  # channel B
    indices_for_ch = [chA_indices, chB_indices]

    rec_all_raw = [None] * 2
    rec_avg_all = [None] * 2
    rec_readout = [[]] * 2
    # print("\ndaq_alazar Troubleshoot stuck at this step 14")
    # Post DMA buffers to board
    # print("E")
    # print(buffers)
    for buffer in buffers:
        board.postAsyncBuffer(buffer.addr, buffer.size_bytes)

    start = time.clock()  # Keep track of when acquisition started

    ## for fast loading
    #    # begin SRS DG535 and Proteus board triggers
    #    generator.proteus_init()
    #    generator.proteus_start_seq()
    # print("\ndaq_alazar Troubleshoot stuck at this step 15")
    # for original, slow loading
    start_from(0)
    #    reset_proteus_clock(slotId=3)
    #    reset_proteus_clock(slotId=5)
    # print("F")
    try:
        # print("\n1")
        board.startCapture()  # Start the acquisition
        # print("\n2")
        #        dg535_control.set_state(1) ## shifting this line from L227 solved sequence timing and averaging window shift issues.

        if verbose:
            print(
                "Capturing %d buffers. Press <enter> to abort" % buffers_per_acquisition
            )

        buffersCompleted = 0
        bytesTransferred = 0
        # print("\n3\n")
        # print(buffers_per_acquisition)

        while buffersCompleted < buffers_per_acquisition and not ats.enter_pressed():
            # Wait for the buffer at the head of the list of available
            # buffers to be filled by the board.
            # print("\n")
            # print(buffersCompleted)
            # print(len(buffers))
            # print("\na")
            buffer = buffers[buffersCompleted % len(buffers)]
            # print("aa")
            # print("\n")
            # print(buffers[0].addr)
            # print("\n")
            # print(buffers[1].addr)
            # print("\n")
            # print(buffers[2].addr)
            # print("\nb")
            board.waitAsyncBufferComplete(buffer.addr, timeout_ms=5000)
            # print("bb")#timeout_ms = 5000 earlier
            # print("\nc")
            buffersCompleted += 1
            bytesTransferred += buffer.size_bytes
            # print("cc")

            #
            # print("\nd")
            for idx, idx_ch in enumerate(indices_for_ch):
                rec_all_raw[idx] = np.reshape(
                    buffer.buffer[idx_ch], (records_per_buffer, post_trigger_samples)
                )
            #
            # print("\ne")
            # print("dd")
            rec_all = rotate_iq(daq_params.iq_angle_deg, rec_all_raw)
            #
            # print("\nf")
            # print("ee")
            for idx in [0, 1]:
                rec_avg_all[idx] = np.mean(
                    rec_all[idx], axis=0
                )  # is this just the avg of the last loop?
                rec_readout[idx] = np.concatenate(
                    (
                        rec_readout[idx],
                        np.mean(rec_all[idx][:, index_avg_start:index_avg_end], axis=1),
                    )
                )

            # NOTE:
            # print("ff")
            #
            # While you are processing this buffer, the board is already
            # filling the next available buffer(s).
            #
            # You MUST finish processing this buffer and post it back to the
            # board before the board fills all of its available DMA buffers
            # and on-board memory.
            #
            # Samples are arranged in the buffer as follows:
            # S0A, S0B, ..., S1A, S1B, ...
            # with SXY the sample number X of channel Y.
            #
            # Sample code are stored as 8-bit values.
            #
            # Sample codes are unsigned by default. As a result:
            # - 0x00 represents a negative full scale input signal.
            # - 0x80 represents a ~0V signal.
            # - 0xFF represents a positive full scale input signal.
            # Optionaly save data to file
            # print("\ng")
            if dataFile:
                buffer.buffer.tofile(dataFile)

            # Add the buffer to the end of the list of available buffers.
            # print("\nh")
            board.postAsyncBuffer(buffer.addr, buffer.size_bytes)
            # print("G")
    finally:
        board.abortAsyncRead()
    #        generator.proteus_close()
    # stop SRS DG535 triggers
    #        dg535_control.set_state(0)
    # print("\ndaq_alazar Troubleshoot stuck at this step 16")
    # Compute the total transfer time, and display performance information.
    # print("H")
    if verbose:
        transferTime_sec = time.clock() - start
        print("Capture completed in %f sec" % transferTime_sec)
        buffersPerSec = 0
        bytesPerSec = 0
        recordsPerSec = 0
        if transferTime_sec > 0:
            buffersPerSec = buffersCompleted / transferTime_sec
            bytesPerSec = bytesTransferred / transferTime_sec
            recordsPerSec = records_per_buffer * buffersCompleted / transferTime_sec
        print(
            "Captured %d buffers (%f buffers per sec)"
            % (buffersCompleted, buffersPerSec)
        )
        print(
            "Captured %d records (%f records per sec)"
            % (records_per_buffer * buffersCompleted, recordsPerSec)
        )
        print(
            "Transferred %d bytes (%f bytes per sec)" % (bytesTransferred, bytesPerSec)
        )
    # print("I")
    return (rec_avg_all, rec_readout)


##END acquire_data


def acquire_data_het(daq_params, alazar_params, board, ssm_if=0.02, verbose=True):
    rec_avg_all = []
    rec_readout = []

    # No pre-trigger samples in NPT mode
    # print("\ndaq_alazar Troubleshoot stuck at this step 1")
    preTriggerSamples = 0

    # Select the number of samples per record.
    # print("\ndaq_alazar Troubleshoot stuck at this step 2")
    post_trigger_samples = alazar_params.post_trigger_samples

    # Select the number0 of records per DMA buffer.
    # print("\ndaq_alazar Troubleshoot stuck at this step 3")
    records_per_buffer = alazar_params.records_per_buffer  # 2**10 # up to 2**14

    # Select the number of buffers per acquisition.
    # print("\ndaq_alazar Troubleshoot stuck at this step 4")
    buffers_per_acquisition = alazar_params.buffers_per_acquisition

    # print("\ndaq_alazar Troubleshoot stuck at this step 5")
    records_per_acquisition = records_per_buffer * buffers_per_acquisition

    # Select the active channels.
    # print("\ndaq_alazar Troubleshoot stuck at this step 6")
    channels = ats.CHANNEL_A | ats.CHANNEL_B
    channelCount = 0
    # print("A")
    # print("\ndaq_alazar Troubleshoot stuck at this step 7")
    for c in ats.channels:
        channelCount += c & channels == c

    # Should data be saved to file?
    saveData = False
    dataFile = None
    # print("\ndaq_alazar Troubleshoot stuck at this step 8")
    if saveData:
        dataFile = open(os.path.join(os.path.dirname(__file__), "data.bin"), "wb")
    # print("\ndaq_alazar Troubleshoot stuck at this step 9")
    # Compute the number of bytes per record and per buffer
    memorySize_samples, bitsPerSample = board.getChannelInfo()
    bytesPerSample = (bitsPerSample.value + 7) // 8
    samplesPerRecord = preTriggerSamples + post_trigger_samples
    bytesPerRecord = bytesPerSample * samplesPerRecord
    bytesPerBuffer = bytesPerRecord * records_per_buffer * channelCount

    # Select number of DMA buffers to allocate
    buffer_count = alazar_params.buffer_count

    # Allocate DMA buffers
    # print("\ndaq_alazar Troubleshoot stuck at this step 10")
    sample_type = ctypes.c_uint8
    # print("B")
    if bytesPerSample > 1:
        sample_type = ctypes.c_uint16

    buffers = []
    for i in range(buffer_count):
        buffers.append(ats.DMABuffer(board.handle, sample_type, bytesPerBuffer))
    # print("\ndaq_alazar Troubleshoot stuck at this step 11")
    # Set the record size
    board.setRecordSize(preTriggerSamples, post_trigger_samples)
    # print("C")
    # Configure the board to make an NPT AutoDMA acquisition
    board.beforeAsyncRead(
        channels,
        -preTriggerSamples,
        samplesPerRecord,
        records_per_buffer,
        records_per_acquisition,
        ats.ADMA_EXTERNAL_STARTCAPTURE | ats.ADMA_NPT,
    )

    # print("\ndaq_alazar Troubleshoot stuck at this step 12")
    index_avg_start = daq_params.readout_start
    index_avg_end = daq_params.readout_start + daq_params.readout_duration - 1

    #    index_ch = [None]*2
    #    index_ch[0] = np.arange(0,post_trigger_samples*records_per_buffer) # channel A
    #    index_ch[1] = post_trigger_samples*records_per_buffer + np.arange(0,post_trigger_samples*records_per_buffer) # channel B
    # print("\ndaq_alazar Troubleshoot stuck at this step 13")
    # print("D")
    chA_indices = np.arange(0, post_trigger_samples * records_per_buffer)
    chB_indices = post_trigger_samples * records_per_buffer + np.arange(
        0, post_trigger_samples * records_per_buffer
    )  # channel B
    indices_for_ch = [chA_indices, chB_indices]

    rec_all_raw = [None] * 2
    rec_avg_all = [None] * 2
    rec_readout = [[]] * 2
    # print("\ndaq_alazar Troubleshoot stuck at this step 14")
    # Post DMA buffers to board
    # print("E")
    # print(buffers)
    for buffer in buffers:
        board.postAsyncBuffer(buffer.addr, buffer.size_bytes)

    start = time.clock()  # Keep track of when acquisition started

    ## for fast loading
    #    # begin SRS DG535 and Proteus board triggers
    #    generator.proteus_init()
    #    generator.proteus_start_seq()
    # print("\ndaq_alazar Troubleshoot stuck at this step 15")
    # for original, slow loading
    #    start_from(0)
    #    reset_proteus_clock(slotId=3)
    #    reset_proteus_clock(slotId=5)
    # print("F")
    try:
        # print("\n1")
        board.startCapture()  # Start the acquisition
        # print("\n2")
        #        dg535_control.set_state(1) ## shifting this line from L227 solved sequence timing and averaging window shift issues.

        if verbose:
            print(
                "Capturing %d buffers. Press <enter> to abort" % buffers_per_acquisition
            )

        buffersCompleted = 0
        bytesTransferred = 0
        # print("\n3\n")
        # print(buffers_per_acquisition)

        while buffersCompleted < buffers_per_acquisition and not ats.enter_pressed():
            # Wait for the buffer at the head of the list of available
            # buffers to be filled by the board.
            # print("\n")
            # print(buffersCompleted)
            # print(len(buffers))
            # print("\na")
            buffer = buffers[buffersCompleted % len(buffers)]
            # print("aa")
            # print("\n")
            # print(buffers[0].addr)
            # print("\n")
            # print(buffers[1].addr)
            # print("\n")
            # print(buffers[2].addr)
            # print("\nb")
            board.waitAsyncBufferComplete(buffer.addr, timeout_ms=5000)
            # print("bb")#timeout_ms = 5000 earlier
            # print("\nc")
            buffersCompleted += 1
            bytesTransferred += buffer.size_bytes
            # print("cc")

            #
            # print("\nd")
            for idx, idx_ch in enumerate(indices_for_ch):
                rec_all_raw[idx] = np.reshape(
                    buffer.buffer[idx_ch], (records_per_buffer, post_trigger_samples)
                )
            #
            # print("\ne")
            # print("dd")
            rec_all = rotate_iq(daq_params.iq_angle_deg, rec_all_raw)
            rec_all = np.array(rec_all)
            omega = np.arange(index_avg_start, index_avg_end) * (2 * np.pi * ssm_if)
            v_cos = np.cos(omega)  # / np.sqrt(alazar_params.post_trigger_samples)
            v_sin = np.sin(omega)  # / np.sqrt(alazar_params.post_trigger_samples)
            v = np.array([[v_cos, -v_sin], [v_sin, v_cos]])
            rec_all_het = np.einsum(
                "ijk, lik->lj", rec_all[:, :, index_avg_start:index_avg_end], v
            )

            #
            # print("\nf")
            # print("ee")
            for idx in [0, 1]:
                # rec_avg_all[idx] = np.mean(rec_all[idx], axis=0) # is this just the avg of the last loop?
                #                rec_readout[idx] = np.concatenate((rec_readout[idx], np.mean(rec_all[idx][:,index_avg_start:index_avg_end], axis=1)))
                rec_readout[idx] = np.concatenate((rec_readout[idx], rec_all_het[idx]))

            # NOTE:
            # print("ff")
            #
            # While you are processing this buffer, the board is already
            # filling the next available buffer(s).
            #
            # You MUST finish processing this buffer and post it back to the
            # board before the board fills all of its available DMA buffers
            # and on-board memory.
            #
            # Samples are arranged in the buffer as follows:
            # S0A, S0B, ..., S1A, S1B, ...
            # with SXY the sample number X of channel Y.
            #
            # Sample code are stored as 8-bit values.
            #
            # Sample codes are unsigned by default. As a result:
            # - 0x00 represents a negative full scale input signal.
            # - 0x80 represents a ~0V signal.
            # - 0xFF represents a positive full scale input signal.
            # Optionaly save data to file
            # print("\ng")
            if dataFile:
                buffer.buffer.tofile(dataFile)

            # Add the buffer to the end of the list of available buffers.
            # print("\nh")
            board.postAsyncBuffer(buffer.addr, buffer.size_bytes)
            # print("G")
    finally:
        board.abortAsyncRead()
    #        generator.proteus_close()
    # stop SRS DG535 triggers
    #       dg535_control.set_state(0)
    # print("\ndaq_alazar Troubleshoot stuck at this step 16")
    # Compute the total transfer time, and display performance information.
    # print("H")
    if verbose:
        transferTime_sec = time.clock() - start
        print("Capture completed in %f sec" % transferTime_sec)
        buffersPerSec = 0
        bytesPerSec = 0
        recordsPerSec = 0
        if transferTime_sec > 0:
            buffersPerSec = buffersCompleted / transferTime_sec
            bytesPerSec = bytesTransferred / transferTime_sec
            recordsPerSec = records_per_buffer * buffersCompleted / transferTime_sec
        print(
            "Captured %d buffers (%f buffers per sec)"
            % (buffersCompleted, buffersPerSec)
        )
        print(
            "Captured %d records (%f records per sec)"
            % (records_per_buffer * buffersCompleted, recordsPerSec)
        )
        print(
            "Transferred %d bytes (%f bytes per sec)" % (bytesTransferred, bytesPerSec)
        )
    # print("I")
    return (rec_avg_all, rec_readout, rec_all, rec_all_het)


# Simultaneous RO DAQ
def acquire_data_het_2q(
    daq_params,
    alazar_params,
    board,
    ssm_if_1=-0.04,
    ssm_if_2=0.10692,
    deg_1=0,
    deg_2=0,
    verbose=True,
):
    rec_avg_all = []
    rec_readout = []

    # No pre-trigger samples in NPT mode
    # print("\ndaq_alazar Troubleshoot stuck at this step 1")
    preTriggerSamples = 0

    # Select the number of samples per record.
    # print("\ndaq_alazar Troubleshoot stuck at this step 2")
    post_trigger_samples = alazar_params.post_trigger_samples

    # Select the number0 of records per DMA buffer.
    # print("\ndaq_alazar Troubleshoot stuck at this step 3")
    records_per_buffer = alazar_params.records_per_buffer  # 2**10 # up to 2**14

    # Select the number of buffers per acquisition.
    # print("\ndaq_alazar Troubleshoot stuck at this step 4")
    buffers_per_acquisition = alazar_params.buffers_per_acquisition

    # print("\ndaq_alazar Troubleshoot stuck at this step 5")
    records_per_acquisition = records_per_buffer * buffers_per_acquisition

    # Select the active channels.
    # print("\ndaq_alazar Troubleshoot stuck at this step 6")
    channels = ats.CHANNEL_A | ats.CHANNEL_B
    channelCount = 0
    # print("A")
    # print("\ndaq_alazar Troubleshoot stuck at this step 7")
    for c in ats.channels:
        channelCount += c & channels == c

    # Should data be saved to file?
    saveData = False
    dataFile = None
    # print("\ndaq_alazar Troubleshoot stuck at this step 8")
    if saveData:
        dataFile = open(os.path.join(os.path.dirname(__file__), "data.bin"), "wb")
    # print("\ndaq_alazar Troubleshoot stuck at this step 9")
    # Compute the number of bytes per record and per buffer
    memorySize_samples, bitsPerSample = board.getChannelInfo()
    bytesPerSample = (bitsPerSample.value + 7) // 8
    samplesPerRecord = preTriggerSamples + post_trigger_samples
    bytesPerRecord = bytesPerSample * samplesPerRecord
    bytesPerBuffer = bytesPerRecord * records_per_buffer * channelCount

    # Select number of DMA buffers to allocate
    buffer_count = alazar_params.buffer_count

    # Allocate DMA buffers
    # print("\ndaq_alazar Troubleshoot stuck at this step 10")
    sample_type = ctypes.c_uint8
    # print("B")
    if bytesPerSample > 1:
        sample_type = ctypes.c_uint16

    buffers = []
    for i in range(buffer_count):
        buffers.append(ats.DMABuffer(board.handle, sample_type, bytesPerBuffer))
    # print("\ndaq_alazar Troubleshoot stuck at this step 11")
    # Set the record size
    board.setRecordSize(preTriggerSamples, post_trigger_samples)
    # print("C")
    # Configure the board to make an NPT AutoDMA acquisition
    board.beforeAsyncRead(
        channels,
        -preTriggerSamples,
        samplesPerRecord,
        records_per_buffer,
        records_per_acquisition,
        ats.ADMA_EXTERNAL_STARTCAPTURE | ats.ADMA_NPT,
    )

    # print("\ndaq_alazar Troubleshoot stuck at this step 12")
    index_avg_start = daq_params.readout_start
    index_avg_end = daq_params.readout_start + daq_params.readout_duration - 1

    #    index_ch = [None]*2
    #    index_ch[0] = np.arange(0,post_trigger_samples*records_per_buffer) # channel A
    #    index_ch[1] = post_trigger_samples*records_per_buffer + np.arange(0,post_trigger_samples*records_per_buffer) # channel B
    # print("\ndaq_alazar Troubleshoot stuck at this step 13")
    # print("D")
    chA_indices = np.arange(0, post_trigger_samples * records_per_buffer)
    chB_indices = post_trigger_samples * records_per_buffer + np.arange(
        0, post_trigger_samples * records_per_buffer
    )  # channel B
    indices_for_ch = [chA_indices, chB_indices]

    rec_all_raw = [None] * 2
    rec_avg_all = [None] * 2
    rec_readout_1 = [[]] * 2
    rec_readout_2 = [[]] * 2
    # print("\ndaq_alazar Troubleshoot stuck at this step 14")
    # Post DMA buffers to board
    # print("E")
    # print(buffers)
    for buffer in buffers:
        board.postAsyncBuffer(buffer.addr, buffer.size_bytes)

    start = time.clock()  # Keep track of when acquisition started

    ## for fast loading
    #    # begin SRS DG535 and Proteus board triggers
    #    generator.proteus_init()
    #    generator.proteus_start_seq()
    # print("\ndaq_alazar Troubleshoot stuck at this step 15")
    # for original, slow loading
    start_from(0)
    #    reset_proteus_clock(slotId=3)
    #    reset_proteus_clock(slotId=5)
    # print("F")
    try:
        # print("\n1")
        board.startCapture()  # Start the acquisition
        # print("\n2")
        #        dg535_control.set_state(1) ## shifting this line from L227 solved sequence timing and averaging window shift issues.

        if verbose:
            print(
                "Capturing %d buffers. Press <enter> to abort" % buffers_per_acquisition
            )

        buffersCompleted = 0
        bytesTransferred = 0
        # print("\n3\n")
        # print(buffers_per_acquisition)
        with tqdm(
            total=buffers_per_acquisition,
            desc="Averaging",
            bar_format="{l_bar}{bar} [time left: {remaining}]",
        ) as pbar:

            while (
                buffersCompleted < buffers_per_acquisition and not ats.enter_pressed()
            ):
                # Wait for the buffer at the head of the list of available
                # buffers to be filled by the board.
                # print("\n")
                # print(buffersCompleted)
                # print(len(buffers))
                # print("\na")
                buffer = buffers[buffersCompleted % len(buffers)]
                # print("aa")
                # print("\n")
                # print(buffers[0].addr)
                # print("\n")
                # print(buffers[1].addr)
                # print("\n")
                # print(buffers[2].addr)
                # print("\nb")
                board.waitAsyncBufferComplete(buffer.addr, timeout_ms=5000)
                # print("bb")#timeout_ms = 5000 earlier
                # print("\nc")
                buffersCompleted += 1
                bytesTransferred += buffer.size_bytes
                # print("cc")

                #
                # print("\nd")
                for idx, idx_ch in enumerate(indices_for_ch):
                    rec_all_raw[idx] = np.reshape(
                        buffer.buffer[idx_ch],
                        (records_per_buffer, post_trigger_samples),
                    )
                #
                # print("\ne")
                # print("dd")
                # rec_all = rotate_iq(daq_params.iq_angle_deg, rec_all_raw)
                rec_all = np.array(rec_all_raw)
                omega_1 = np.arange(index_avg_start, index_avg_end) * (
                    2 * np.pi * ssm_if_1
                )
                v_cos_1 = np.cos(
                    omega_1 + ((np.pi * deg_1) / 180)
                )  # / np.sqrt(alazar_params.post_trigger_samples)
                v_sin_1 = np.sin(
                    omega_1 + ((np.pi * deg_1) / 180)
                )  # / np.sqrt(alazar_params.post_trigger_samples)
                v_1 = np.array([[v_cos_1, -v_sin_1], [v_sin_1, v_cos_1]])
                rec_all_het_1 = np.einsum(
                    "ijk, lik->lj", rec_all[:, :, index_avg_start:index_avg_end], v_1
                )

                omega_2 = np.arange(index_avg_start, index_avg_end) * (
                    (2 * np.pi * ssm_if_2)
                )
                v_cos_2 = np.cos(
                    omega_2 + ((np.pi * deg_2) / 180)
                )  # / np.sqrt(alazar_params.post_trigger_samples)
                v_sin_2 = np.sin(
                    omega_2 + ((np.pi * deg_2) / 180)
                )  # / np.sqrt(alazar_params.post_trigger_samples)
                v_2 = np.array([[v_cos_2, -v_sin_2], [v_sin_2, v_cos_2]])
                rec_all_het_2 = np.einsum(
                    "ijk, lik->lj", rec_all[:, :, index_avg_start:index_avg_end], v_2
                )

                #
                # print("\nf")
                # print("ee")
                for idx in [0, 1]:
                    # rec_avg_all[idx] = np.mean(rec_all[idx], axis=0) # is this just the avg of the last loop?
                    # rec_readout[idx] = np.concatenate((rec_readout[idx], np.mean(rec_all[idx][:,index_avg_start:index_avg_end], axis=1)))
                    rec_readout_1[idx] = np.concatenate(
                        (rec_readout_1[idx], rec_all_het_1[idx])
                    )
                    rec_readout_2[idx] = np.concatenate(
                        (rec_readout_2[idx], rec_all_het_2[idx])
                    )

                # NOTE:
                # print("ff")
                #
                # While you are processing this buffer, the board is already
                # filling the next available buffer(s).
                #
                # You MUST finish processing this buffer and post it back to the
                # board before the board fills all of its available DMA buffers
                # and on-board memory.
                #
                # Samples are arranged in the buffer as follows:
                # S0A, S0B, ..., S1A, S1B, ...
                # with SXY the sample number X of channel Y.
                #
                # Sample code are stored as 8-bit values.
                #
                # Sample codes are unsigned by default. As a result:
                # - 0x00 represents a negative full scale input signal.
                # - 0x80 represents a ~0V signal.
                # - 0xFF represents a positive full scale input signal.
                # Optionaly save data to file
                # print("\ng")
                if dataFile:
                    buffer.buffer.tofile(dataFile)

                # Add the buffer to the end of the list of available buffers.
                # print("\nh")
                board.postAsyncBuffer(buffer.addr, buffer.size_bytes)
                # print("G")
                pbar.update(1)
    finally:
        board.abortAsyncRead()
    #        generator.proteus_close()
    # stop SRS DG535 triggers
    #        dg535_control.set_state(0)
    # print("\ndaq_alazar Troubleshoot stuck at this step 16")
    # Compute the total transfer time, and display performance information.
    # print("H")
    if verbose:
        transferTime_sec = time.clock() - start
        print("Capture completed in %f sec" % transferTime_sec)
        buffersPerSec = 0
        bytesPerSec = 0
        recordsPerSec = 0
        if transferTime_sec > 0:
            buffersPerSec = buffersCompleted / transferTime_sec
            bytesPerSec = bytesTransferred / transferTime_sec
            recordsPerSec = records_per_buffer * buffersCompleted / transferTime_sec
        print(
            "Captured %d buffers (%f buffers per sec)"
            % (buffersCompleted, buffersPerSec)
        )
        print(
            "Captured %d records (%f records per sec)"
            % (records_per_buffer * buffersCompleted, recordsPerSec)
        )
        print(
            "Transferred %d bytes (%f bytes per sec)" % (bytesTransferred, bytesPerSec)
        )
    # print("I")
    return (
        rec_avg_all,
        rec_readout_1,
        rec_readout_2,
        rec_all,
        rec_all_het_1,
        rec_all_het_2,
    )


# def acquire_data2(daq_params, alazar_params, board, verbose=True):
#    rec_avg_all = []
#    rec_readout = []
#    temp = []
#
#    # No pre-trigger samples in NPT mode
#    preTriggerSamples = 0
#
#    # Select the number of samples per record.
#    post_trigger_samples = alazar_params.post_trigger_samples
#
#    # Select the number0 of records per DMA buffer.
#    records_per_buffer = alazar_params.records_per_buffer #2**10 # up to 2**14
#
#    # Select the number of buffers per acquisition.
#    buffers_per_acquisition = alazar_params.buffers_per_acquisition
#
#    records_per_acquisition = records_per_buffer * buffers_per_acquisition
#
#    # Select the active channels.
#    channels = ats.CHANNEL_A | ats.CHANNEL_B
#    channelCount = 0
#    for c in ats.channels:
#        channelCount += (c & channels == c)
#
#    # Should data be saved to file?
#    saveData = False
#    dataFile = None
#    if saveData:
#        dataFile = open(os.path.join(os.path.dirname(__file__),
#                                     "data.bin"), 'wb')
#
#    # Compute the number of bytes per record and per buffer
#    memorySize_samples, bitsPerSample = board.getChannelInfo()
#    bytesPerSample = (bitsPerSample.value + 7) // 8
#    samplesPerRecord = preTriggerSamples + post_trigger_samples
#    bytesPerRecord = bytesPerSample * samplesPerRecord
#    bytesPerBuffer = bytesPerRecord * records_per_buffer * channelCount
#
#    # Select number of DMA buffers to allocate
#    buffer_count = alazar_params.buffer_count
#
#    # Allocate DMA buffers
#
#    sample_type = ctypes.c_uint8
#    if bytesPerSample > 1:
#        sample_type = ctypes.c_uint16
#
#    buffers = []
#    for i in range(buffer_count):
#        buffers.append(ats.DMABuffer(board.handle, sample_type, bytesPerBuffer))
#
#    # Set the record size
#    board.setRecordSize(preTriggerSamples, post_trigger_samples)
#
#    # Configure the board to make an NPT AutoDMA acquisition
#    board.beforeAsyncRead(channels,
#                          -preTriggerSamples,
#                          samplesPerRecord,
#                          records_per_buffer,
#                          records_per_acquisition,
#                          ats.ADMA_EXTERNAL_STARTCAPTURE | ats.ADMA_NPT)
#
#    index_avg_start = daq_params.readout_start
#    index_avg_end = daq_params.readout_start + daq_params.readout_duration - 1
#
#    index_ch = [None]*2
#    index_ch[0] = np.arange(0,post_trigger_samples*records_per_buffer) # channel A
#    index_ch[1] = post_trigger_samples*records_per_buffer + np.arange(0,post_trigger_samples*records_per_buffer) # channel B
#
#    rec_all_raw = [None]*2
#    rec_avg_all = [None]*2
#    rec_readout = [[]]*2
#
#    # Post DMA buffers to board
#    for buffer in buffers:
#        board.postAsyncBuffer(buffer.addr, buffer.size_bytes)
#
#    start = time.clock() # Keep track of when acquisition started
#
#    # start SRS DG535 triggers
#    dg535_control.set_state(1)
#
#    try:
#        board.startCapture() # Start the acquisition
#        if verbose:
#            print("Capturing %d buffers. Press <enter> to abort" %
#                  buffers_per_acquisition)
#        buffersCompleted = 0
#        bytesTransferred = 0
#        while (buffersCompleted < buffers_per_acquisition and not
#               ats.enter_pressed()):
#            # Wait for the buffer at the head of the list of available
#            # buffers to be filled by the board.
#            buffer = buffers[buffersCompleted % len(buffers)]
#            board.waitAsyncBufferComplete(buffer.addr, timeout_ms=5000)
#            buffersCompleted += 1
#            bytesTransferred += buffer.size_bytes
#
#            #
#            for idx, idx_ch in enumerate(index_ch):
#                rec_all_raw[idx] = np.reshape(buffer.buffer[idx_ch], (records_per_buffer, post_trigger_samples))
#
#            #
#            rec_all = rotate_iq(daq_params.iq_angle_deg, rec_all_raw)
#            temp.append(rec_all)
#            #
#            for idx in [0, 1]:
#                rec_avg_all[idx] = np.mean(rec_all[idx], axis=0) # is this just the avg of the last loop?
#                rec_readout[idx] = np.concatenate((rec_readout[idx], np.mean(rec_all[idx][:,index_avg_start:index_avg_end], axis=1)))
#
#            # NOTE:
#            #
#            # While you are processing this buffer, the board is already
#            # filling the next available buffer(s).
#            #
#            # You MUST finish processing this buffer and post it back to the
#            # board before the board fills all of its available DMA buffers
#            # and on-board memory.
#            #
#            # Samples are arranged in the buffer as follows:
#            # S0A, S0B, ..., S1A, S1B, ...
#            # with SXY the sample number X of channel Y.
#            #
#            # Sample code are stored as 8-bit values.
#            #
#            # Sample codes are unsigned by default. As a result:
#            # - 0x00 represents a negative full scale input signal.
#            # - 0x80 represents a ~0V signal.
#            # - 0xFF represents a positive full scale input signal.
#            # Optionaly save data to file
#            if dataFile:
#                buffer.buffer.tofile(dataFile)
#
#            # Add the buffer to the end of the list of available buffers.
#            board.postAsyncBuffer(buffer.addr, buffer.size_bytes)
#    finally:
#        board.abortAsyncRead()
#
#    # stop SRS DG535 triggers
#    dg535_control.set_state(0)
#
#    # Compute the total transfer time, and display performance information.
#    if verbose:
#        transferTime_sec = time.clock() - start
#        print("Capture completed in %f sec" % transferTime_sec)
#        buffersPerSec = 0
#        bytesPerSec = 0
#        recordsPerSec = 0
#        if transferTime_sec > 0:
#            buffersPerSec = buffersCompleted / transferTime_sec
#            bytesPerSec = bytesTransferred / transferTime_sec
#            recordsPerSec = records_per_buffer * buffersCompleted / transferTime_sec
#        print("Captured %d buffers (%f buffers per sec)" %
#              (buffersCompleted, buffersPerSec))
#        print("Captured %d records (%f records per sec)" %
#              (records_per_buffer * buffersCompleted, recordsPerSec))
#        print("Transferred %d bytes (%f bytes per sec)" %
#              (bytesTransferred, bytesPerSec))
#
#    return (rec_avg_all, rec_readout, temp)
###END acquire_data2


# def restart_proteus():
#    for ch_index in range(4):
#        proteusapi.SendScpi(":INST:CHAN {}".format(ch_index+1), inst_index=0)
#        proteusapi.SendScpi(":TASK:SEL 1", inst_index=0)
#        proteusapi.SendScpi(":TASK:DEF:NEXT1 2", inst_index=0)

##END restart_proteus
print("COMPLETE!")
if __name__ == "__main__":
    pass
