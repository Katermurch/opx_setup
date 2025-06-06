# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 11:50:58 2020.

@author: P. M. Harrington
"""

import numpy as np
import matplotlib.pyplot as plt
import analysis
import hardware_control.expt_parameters as expt_parameters

# from daq_alazar_homo import *
import daq.daq_alazar_homo as daq_alazar_homo  # serra
import hardware_control.atsapi as ats  # serra

# from daq_processing import *
import daq.daq_processing as daq_processing
import hardware_control.wx_programs as wx_programs
from classes.Nop_class import Nop
import math
from dataclasses import dataclass


def get_daq_parameters(
    num_patterns=None, num_records_per_pattern=None, ro_dur=7000, IQangle=90
):
    daq_params = expt_parameters.get_daq_parameters(ro_dur, IQangle)

    # number of patterns
    if num_patterns is None:
        daq_params.num_patterns = 51
    else:
        daq_params.num_patterns = num_patterns

    # number of repetitions for each pattern
    if num_records_per_pattern is None:
        daq_params.num_records_per_pattern = 50
    else:
        daq_params.num_records_per_pattern = num_records_per_pattern

    return daq_params


def run_daq(num_patterns=None, num_records_per_pattern=None, ro_dur=7000, IQangle=90):
    # get parameters: number of patterns, etc.
    daq_params = get_daq_parameters(
        num_patterns=num_patterns,
        num_records_per_pattern=num_records_per_pattern,
        ro_dur=ro_dur,
        IQangle=IQangle,
    )
    alazar_params = daq_alazar_homo.get_alazar_parameters(daq_params=daq_params)

    print("\nSetup Alazar configuration")
    board = ats.Board(systemId=1, boardId=1)
    daq_alazar_homo.configure_board(alazar_params, board)

    # setup wx to start at first pattern
    print("Initialize WX")
    wx_programs.wx_initialize()

    #
    print("Acquire data\n")
    (rec_avg_all, rec_readout, rec_all) = daq_alazar_homo.acquire_data(
        daq_params, alazar_params, board
    )
    times = np.linspace(0, ro_dur, len(rec_readout[0]))
    #    rec_readout = rec_readout*np.exp(-0.001*times)

    # reshape the records
    rec_readout_vs_pats = daq_processing.record_vs_patterns(daq_params, rec_readout)

    # average all repetitions for each pattern
    rec_avg_vs_pats_ch_a, rec_avg_vs_pats_ch_b = daq_processing.record_avg_vs_patterns(
        daq_params, rec_readout_vs_pats
    )

    # threshold the readout signal for every record (channel a)
    n_readout = daq_processing.threshold_record_averages(
        daq_params, signal_in=rec_readout[0]
    )
    n_vs_pats, p_vs_pats = daq_processing.readout_vs_patterns(daq_params, n_readout)

    #
    bins, counts = daq_processing.make_iq_plot(rec_readout)

    #
    daq_processing.make_readout_vs_patterns_plot(p_vs_pats)

    #    return daq_params, rec_readout_vs_pats, p_vs_pats, rec_avg_vs_pats_ch_a, rec_avg_vs_pats_ch_b
    return (
        daq_params,
        rec_readout_vs_pats,
        p_vs_pats,
        rec_avg_vs_pats_ch_a,
        rec_avg_vs_pats_ch_b,
        bins[0],
        counts[0],
        rec_readout,
        rec_avg_all,
        rec_all,
    )


def run_daq_het(
    ssm_if=0.02,
    num_patterns=None,
    num_records_per_pattern=None,
    ro_dur=None,
    verbose=True,
):
    # get parameters: number of patterns, etc.
    daq_params = get_daq_parameters(
        num_patterns=num_patterns,
        num_records_per_pattern=num_records_per_pattern,
        ro_dur=ro_dur,
    )

    alazar_params = daq_alazar_homo.get_alazar_parameters(
        daq_params=daq_params
    )  # , verbose=False)

    # print("\nTroubleshoot stuck at this step 1")
    board = ats.Board(systemId=1, boardId=1)
    # print("\nTroubleshoot stuck at this step 2")
    daq_alazar_homo.configure_board(alazar_params, board)
    # print("\nTroubleshoot stuck at this step 3")
    wx_programs.wx_initialize()
    (rec_avg_all, rec_readout, rec_all, rec_all_het) = daq_alazar_homo.acquire_data_het(
        daq_params, alazar_params, board, ssm_if
    )  # , verbose=True)
    # print(alazar_params.buffer_count)
    # print("\nTroubleshoot stuck at this step 4")
    # reshape the records
    rec_readout_vs_pats = daq_processing.record_vs_patterns(daq_params, rec_readout)
    # print("\nTroubleshoot stuck at this step 5")
    # average all repetitions for each pattIern
    rec_avg_vs_pats_ch_a, rec_avg_vs_pats_ch_b = daq_processing.record_avg_vs_patterns(
        daq_params, rec_readout_vs_pats
    )
    # print("\nTroubleshoot stuck at this step 6")
    rec_avg_vs_pats = [rec_avg_vs_pats_ch_a, rec_avg_vs_pats_ch_b]

    if verbose:
        pass
        bins, counts = daq_processing.make_iq_plot(rec_readout)

    # return rec_avg_all, rec_readout, rec_avg_vs_pats
    return rec_avg_all, rec_readout, rec_avg_vs_pats, rec_all_het, bins, counts


def run_daq_het_2q(
    qubit1, qubit2, num_patterns=None, num_records_per_pattern=None, verbose=True
):
    # Define a local dataclass to structure the DAQ results

    qubit_1_thr = qubit1.qubit_thr
    qubit_2_thr = qubit2.qubit_thr
    ssm_if_1 = qubit1.ROIF
    ssm_if_2 = qubit2.ROIF
    deg_1 = qubit1.IQ_angle
    deg_2 = qubit2.IQ_angle
    ro_dur = qubit1.ro_dur

    @dataclass
    class DAQResult:
        n_vs_pats_1: any
        n_vs_pats_2: any
        rec_avg_all: any
        rec_all: any
        rec_readout_1: any
        rec_readout_2: any
        rec_avg_vs_pats_1: any
        rec_avg_vs_pats_2: any
        rec_all_het_1: any
        rec_all_het_2: any
        bins_1: any
        bins_2: any
        counts_1: any
        counts_2: any
        prob_vs_pats_1: any
        prob_vs_pats_2: any
        n_readout_1: any
        n_readout_2: any
        rec_readout_vs_pats_1: any
        rec_readout_vs_pats_2: any

    # 1. Get DAQ parameters
    daq_params = get_daq_parameters(
        num_patterns=num_patterns,
        num_records_per_pattern=num_records_per_pattern,
        ro_dur=ro_dur,
    )

    # 2. Get Alazar (DAQ board) parameters and configure board
    alazar_params = daq_alazar_homo.get_alazar_parameters(daq_params=daq_params)
    board = ats.Board(systemId=1, boardId=1)
    daq_alazar_homo.configure_board(alazar_params, board)

    # 3. Initialize any required GUI or supplementary modules
    wx_programs.wx_initialize()

    # 4. Acquire data (heterodyne, 2-qubit)
    (
        rec_avg_all,
        rec_readout_1,
        rec_readout_2,
        rec_all,
        rec_all_het_1,
        rec_all_het_2,
    ) = daq_alazar_homo.acquire_data_het_2q(
        daq_params, alazar_params, board, ssm_if_1, ssm_if_2, deg_1, deg_2, verbose=True
    )

    # 5. Process the acquired records: reshape and average vs. patterns
    rec_readout_vs_pats_1 = daq_processing.record_vs_patterns(daq_params, rec_readout_1)
    rec_readout_vs_pats_2 = daq_processing.record_vs_patterns(daq_params, rec_readout_2)

    rec_avg_vs_pats_1_ch_a, rec_avg_vs_pats_1_ch_b = (
        daq_processing.record_avg_vs_patterns(daq_params, rec_readout_vs_pats_1)
    )
    rec_avg_vs_pats_2_ch_a, rec_avg_vs_pats_2_ch_b = (
        daq_processing.record_avg_vs_patterns(daq_params, rec_readout_vs_pats_2)
    )
    rec_avg_vs_pats_1 = [rec_avg_vs_pats_1_ch_a, rec_avg_vs_pats_1_ch_b]
    rec_avg_vs_pats_2 = [rec_avg_vs_pats_2_ch_a, rec_avg_vs_pats_2_ch_b]

    # 6. Optionally create IQ plots for visualization
    if verbose:
        bins_1, counts_1 = daq_processing.make_iq_plot(rec_readout_1, "qubit 1")
        bins_2, counts_2 = daq_processing.make_iq_plot(rec_readout_2, "qubit 2")
    else:
        bins_1 = bins_2 = counts_1 = counts_2 = None

    # 7. Process thresholds and compute readout vs. patterns for qubit 1
    daq_params.threshold = qubit_1_thr
    n_readout_1 = daq_processing.threshold_record_averages(
        daq_params, signal_in=rec_readout_1[0]
    )
    n_vs_pats_1, prob_vs_pats_1 = daq_processing.readout_vs_patterns(
        daq_params, n_readout_1
    )

    # 8. Process thresholds and compute readout vs. patterns for qubit 2
    daq_params.threshold = qubit_2_thr
    n_readout_2 = daq_processing.threshold_record_averages(
        daq_params, signal_in=rec_readout_2[0]
    )
    n_vs_pats_2, prob_vs_pats_2 = daq_processing.readout_vs_patterns(
        daq_params, n_readout_2
    )

    # 9. Pack and return all the results using the DAQResult dataclass
    return DAQResult(
        n_vs_pats_1=n_vs_pats_1,
        n_vs_pats_2=n_vs_pats_2,
        rec_avg_all=rec_avg_all,
        rec_all=rec_all,
        rec_readout_1=rec_readout_1,
        rec_readout_2=rec_readout_2,
        rec_avg_vs_pats_1=rec_avg_vs_pats_1,
        rec_avg_vs_pats_2=rec_avg_vs_pats_2,
        rec_all_het_1=rec_all_het_1,
        rec_all_het_2=rec_all_het_2,
        bins_1=bins_1,
        bins_2=bins_2,
        counts_1=counts_1,
        counts_2=counts_2,
        prob_vs_pats_1=prob_vs_pats_1,
        prob_vs_pats_2=prob_vs_pats_2,
        n_readout_1=n_readout_1,
        n_readout_2=n_readout_2,
        rec_readout_vs_pats_1=rec_readout_vs_pats_1,
        rec_readout_vs_pats_2=rec_readout_vs_pats_2,
    )


def run_daq_auto_threshold_modify_ec(
    prev_threshold=[-160.5, -157],
    num_patterns=None,
    num_records_per_pattern=None,
    authr=0,
    fg=3,
    ro_dur=8000,
    IQangle=90,
):
    # get parameters: number of patterns, etc.

    daq_params = get_daq_parameters(
        num_patterns=num_patterns,
        num_records_per_pattern=num_records_per_pattern,
        ro_dur=ro_dur,
        IQangle=IQangle,
    )
    alazar_params = get_alazar_parameters(daq_params=daq_params)

    print("\nSetup Alazar configuration")
    board = ats.Board(systemId=1, boardId=1)
    configure_board(alazar_params, board)

    # setup wx to start at first pattern
    print("Initialize WX")
    wx_programs.wx_initialize()

    #
    print("Acquire data\n")
    (rec_avg_all, rec_readout, rec_all) = acquire_data(daq_params, alazar_params, board)

    # reshape the records
    rec_readout_vs_pats = record_vs_patterns(daq_params, rec_readout)

    #
    bins, counts = make_iq_plot(rec_readout)

    # average all repetitions for each pattern
    rec_avg_vs_pats_ch_a, rec_avg_vs_pats_ch_b = record_avg_vs_patterns(
        daq_params, rec_readout_vs_pats
    )

    guess_thr = prev_threshold
    if authr == 0:
        # threshold the readout signal for every record (channel a)
        try:
            if fg == 3:
                daq_params.threshold = analysis.fit_three_gaussian(
                    bins[0], counts[0]
                )  ###
                print(daq_params.threshold)
            elif fg == 2:
                daq_params.threshold = analysis.fit_two_gaussian(
                    bins[0], counts[0], guess_thr
                )
                print(daq_params.threshold)
        except:
            daq_params.threshold = prev_threshold
            print(daq_params.threshold)

    else:
        daq_params.threshold = prev_threshold
    #    daq_params.threshold=prev_threshold
    n_readout = threshold_record_averages(daq_params, signal_in=rec_readout[0])
    n_vs_pats, p_vs_pats = readout_vs_patterns(daq_params, n_readout)

    #
    make_readout_vs_patterns_plot(p_vs_pats)

    return daq_params, rec_readout_vs_pats, p_vs_pats, bins, counts, n_readout


def run_daq_cluster_threshold(
    model,
    Ax=[0, 1, 2],
    By=[0, 1, 2],
    num_patterns=None,
    num_records_per_pattern=None,
    authr=0,
    fg=3,
    ro_dur=8000,
    IQangle=90,
):
    # get parameters: number of patterns, etc.

    daq_params = get_daq_parameters(
        num_patterns=num_patterns,
        num_records_per_pattern=num_records_per_pattern,
        ro_dur=ro_dur,
        IQangle=IQangle,
    )
    alazar_params = get_alazar_parameters(daq_params=daq_params)

    print("\nSetup Alazar configuration")
    board = ats.Board(systemId=1, boardId=1)
    configure_board(alazar_params, board)

    # setup wx to start at first pattern
    print("Initialize WX")
    wx_programs.wx_initialize()

    #
    print("Acquire data\n")
    (rec_avg_all, rec_readout, rec_all) = acquire_data(daq_params, alazar_params, board)

    # reshape the records
    rec_readout_vs_pats = record_vs_patterns(daq_params, rec_readout)

    #
    bins, counts = make_iq_plot(rec_readout)

    # average all repetitions for each pattern
    rec_avg_vs_pats_ch_a, rec_avg_vs_pats_ch_b = record_avg_vs_patterns(
        daq_params, rec_readout_vs_pats
    )

    n_readout = cluster_threshold_record_averages(
        model, Ax, By, daq_params, signal_in=rec_readout
    )
    n_vs_pats, p_vs_pats = readout_vs_patterns(daq_params, n_readout)

    make_readout_vs_patterns_plot(p_vs_pats)

    return daq_params, rec_readout_vs_pats, p_vs_pats, bins, counts, n_readout


def run_daq_auto_threshold(num_patterns=None, num_records_per_pattern=None, authr=0):
    # get parameters: number of patterns, etc.

    daq_params = get_daq_parameters(
        num_patterns=num_patterns, num_records_per_pattern=num_records_per_pattern
    )
    alazar_params = get_alazar_parameters(daq_params=daq_params)

    print("\nSetup Alazar configuration")
    board = ats.Board(systemId=1, boardId=1)
    configure_board(alazar_params, board)

    # setup wx to start at first pattern
    print("Initialize WX")
    wx_programs.wx_initialize()

    #
    print("Acquire data\n")
    (rec_avg_all, rec_readout) = acquire_data(daq_params, alazar_params, board)

    # reshape the records
    rec_readout_vs_pats = record_vs_patterns(daq_params, rec_readout)

    #
    bins, counts = make_iq_plot(rec_readout)

    # average all repetitions for each pattern
    rec_avg_vs_pats_ch_a, rec_avg_vs_pats_ch_b = record_avg_vs_patterns(
        daq_params, rec_readout_vs_pats
    )

    if authr == 0:
        # threshold the readout signal for every record (channel a)
        daq_params.threshold = analysis.fit_three_gaussian(bins[0], counts[0])
    print(daq_params.threshold)
    n_readout = threshold_record_averages(daq_params, signal_in=rec_readout[0])
    n_vs_pats, p_vs_pats = readout_vs_patterns(daq_params, n_readout)

    #
    make_readout_vs_patterns_plot(p_vs_pats)

    return daq_params, rec_readout_vs_pats, p_vs_pats, bins, counts


def run_daq_rawdata(num_patterns=None, num_records_per_pattern=None):
    # get parameters: number of patterns, etc.
    daq_params = get_daq_parameters(
        num_patterns=num_patterns, num_records_per_pattern=num_records_per_pattern
    )
    alazar_params = get_alazar_parameters(daq_params=daq_params)

    print("\nSetup Alazar configuration")
    board = ats.Board(systemId=1, boardId=1)
    configure_board(alazar_params, board)

    # setup wx to start at first pattern
    print("Initialize WX")
    wx_programs.wx_initialize()

    #
    print("Acquire data\n")
    (rec_avg_all, rec_readout, rec_all_raw) = acquire_data_raw(
        daq_params, alazar_params, board
    )

    rec_all_raw_ave = np.zeros(
        (
            np.shape(rec_all_raw)[1],
            np.shape(rec_all_raw)[0] * np.shape(rec_all_raw)[2],
            np.shape(rec_all_raw)[3],
        )
    )
    for k in np.arange(np.shape(rec_all_raw)[0]):
        rec_all_raw_ave[
            :, k * np.shape(rec_all_raw)[2] : (k + 1) * np.shape(rec_all_raw)[2], :
        ] = rec_all_raw[k]

    #    start_time = time.time()
    rec_all_raw_ave = rec_all_raw_ave[:, 0 : num_patterns * num_records_per_pattern, :]
    #    print("--- %s seconds ---" % (time.time() - start_time))

    return daq_params, rec_all_raw_ave


def run_iq_vs_patterns(num_patterns=None, num_records_per_pattern=None, ro_dur=7000):
    # get parameters: number of patterns, etc.
    daq_params = get_daq_parameters(
        num_patterns=num_patterns,
        num_records_per_pattern=num_records_per_pattern,
        ro_dur=ro_dur,
    )
    alazar_params = get_alazar_parameters(daq_params=daq_params)

    print("\nSetup Alazar configuration")
    board = ats.Board(systemId=1, boardId=1)
    configure_board(alazar_params, board)

    # setup wx to start at first pattern
    print("Initialize WX")
    wx_programs.wx_initialize()

    #
    print("Acquire data\n")
    (rec_avg_all, rec_readout) = acquire_data(daq_params, alazar_params, board)

    # reshape the records
    rec_readout_vs_pats = record_vs_patterns(daq_params, rec_readout)

    # make IQ plot for each pattern
    bins_cntr, counts = make_n_state_iq_plot(rec_readout_vs_pats)
    #    fit_readout_histogram(rec_readout[0], bins_cntr[0], counts[0], num_gaussians=3)

    # average all repetitions for each pattern
    # rec_avg_vs_pats_ch_a, rec_avg_vs_pats_ch_b = record_avg_vs_patterns(daq_params, rec_readout_vs_pats)

    # threshold the readout signal for every record (channel a)
    n_readout = threshold_record_averages(daq_params, signal_in=rec_readout[0])
    n_vs_pats, p_vs_pats = readout_vs_patterns(daq_params, n_readout)
    make_readout_vs_patterns_plot(p_vs_pats)

    return daq_params, rec_readout_vs_pats, n_vs_pats, p_vs_pats, bins_cntr, counts


if __name__ == "__main__":
    pass
