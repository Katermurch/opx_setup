# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 11:18:02 2020.

@author: P. M. Harrington, 27 January 2020
"""

import numpy as np
import os
import datetime
import analysis
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import integrate
import csv
from sklearn import svm


class Nop:
    pass


# def make_readme():
#    dir_save = daq_params.dir_save
#
#    if (not os.path.isdir(dir_save)):
#        os.mkdir(dir_save)


def readout_with_and_without_pi_and_cal(daq_params, p_readout):
    num_pats = daq_params.num_patterns
    num_pats_each = num_pats // 2  # (num_pats-3)//2

    p_readout.T[:51].T

    # rescale p_f

    #    p_f_all_raw = p_readout[2]

    #    # rescale p_f
    ##    scale = 0.212 #p_f_all_raw[-1] - p_f_all_raw[-2]
    ##    offset = 0.106#0.5*(p_f_all_raw[-1] + p_f_all_raw[-2]) - 0.5
    #    scale = p_f_all_raw[-1] - p_f_all_raw[-2]
    #    offset = 0.5*(p_f_all_raw[-1] + p_f_all_raw[-2]) - 0.5
    #
    #    print("Scale: {}".format(scale))
    #    print("Offset: {}".format(offset))
    #
    ##    p_f_all = p_f_all_raw
    #    p_f_all = (p_f_all_raw-offset)/(2*scale) + 0.5

    p_f = p_f_all[:num_pats_each]  # first sequence has no pi
    p_e = p_f_all[num_pats_each : 2 * num_pats_each]  # second sequence has pi

    p_e_post = p_e / (p_e + p_f)
    p_f_post = p_f / (p_e + p_f)

    p_readout_fnotf = []
    p_readout_fnotf.append(p_readout)
    p_readout_fnotf.append(0 * p_e)

    return (p_f_post, p_e_post, p_f, p_e)


def record_vs_patterns(daq_params=None, rec_readout=[None, None]):
    rec_a = rec_readout[0]
    rec_b = rec_readout[1]

    rec_vs_pats_ch_a = []
    rec_vs_pats_ch_b = []

    num_patterns = daq_params.num_patterns
    num_records_per_pattern = daq_params.num_records_per_pattern
    num_total_records = num_patterns * num_records_per_pattern

    index = 0
    for rec in (rec_a, rec_b):
        rec = rec[0:num_total_records]
        rec = np.reshape(rec, (num_records_per_pattern, num_patterns))
        if index == 0:
            rec_vs_pats_ch_a = rec
        elif index == 1:
            rec_vs_pats_ch_b = rec

        index += 1

    rec_readout_vs_pats = np.array([rec_vs_pats_ch_a, rec_vs_pats_ch_b])
    return rec_readout_vs_pats


def record_avg_vs_patterns(daq_params=None, rec=[]):
    rec_a = rec[0]
    rec_b = rec[1]

    rec_avg_vs_pats_channel_a = []
    rec_avg_vs_pats_channel_b = []

    index = 0
    for rec in (rec_a, rec_b):
        rec = np.mean(rec, axis=0)
        if index == 0:
            rec_avg_vs_pats_channel_a = rec
        elif index == 1:
            rec_avg_vs_pats_channel_b = rec

        index += 1

    return (rec_avg_vs_pats_channel_a, rec_avg_vs_pats_channel_b)


def readout_vs_patterns(daq_params=None, n_readout_all_thresholds=[]):
    n_vs_pats = []
    p_vs_pats = []

    num_patterns = daq_params.num_patterns
    num_records_per_pattern = daq_params.num_records_per_pattern
    num_total_records = num_patterns * num_records_per_pattern

    for n_readout in n_readout_all_thresholds:
        _n = n_readout[0:num_total_records]
        _n = np.reshape(_n, (num_records_per_pattern, num_patterns))
        n_vs_pats.append(_n)
        p_vs_pats.append(np.mean(_n, axis=0))

    n_vs_pats = np.array(n_vs_pats)
    p_vs_pats = np.array(p_vs_pats)

    return n_vs_pats, p_vs_pats


def make_readout_vs_patterns_plot(p_readout_vs_pats, ylabel_str=None):

    plt.figure()
    for _p in p_readout_vs_pats:
        if _p is None:
            continue

        plt.plot(_p)

    plt.ylim(0, 1)
    plt.xlabel("pattern #")
    if ylabel_str is None:
        plt.ylabel("Pbability")
    else:
        plt.ylabel(ylabel_str)

    plt.show()


def threshold_record_averages(daq_params, signal_in=[]):
    #
    thresholds = np.sort(daq_params.threshold)[::-1]
    thresholds = thresholds.tolist() + [-np.inf]

    n_readout = []
    prev_threshold = np.inf

    for threshold in thresholds:
        _n = [1 if threshold < s and s < prev_threshold else 0 for s in signal_in]

        n_readout.append(_n)
        prev_threshold = threshold

    n_readout = np.array(n_readout)

    return n_readout


# TODO finish cluster threshold
def cluster_threshold_record_averages(model, Ax, By, daq_params, signal_in=[]):
    IQdata = []
    for i in range(len(signal_in[0])):

        IQdata.append([signal_in[0][i], signal_in[1][i]])

    #    n_readout = []
    #    _n = []
    #    for i,s in enumerate(signal_in[1]): #|g>
    #        if -Ax[0]*signal_in[0][i]-By[0]*signal_in[1][i]<1 and -Ax[1]*signal_in[0][i]-By[1]*signal_in[1][i]>1:
    #            _n.append(1)
    #        else:
    #            _n.append(0)
    #    n_readout.append(_n)
    #    _n = []
    #    for i,s in enumerate(signal_in[1]): #|e>
    #        if -Ax[1]*signal_in[0][i]-By[1]*signal_in[1][i]>1 and -Ax[2]*signal_in[0][i]-By[2]*signal_in[1][i]<1:
    #            _n.append(1)
    #        else:
    #            _n.append(0)
    #    n_readout.append(_n)
    #    _n = []
    #    for i,s in enumerate(signal_in[1]): #|f>
    #        if -Ax[2]*signal_in[0][i]-By[2]*signal_in[1][i]<1 and -Ax[0]*signal_in[0][i]-By[0]*signal_in[1][i]<1:
    #            _n.append(1)
    #        else:
    #            _n.append(0)
    #    n_readout.append(_n)
    #
    #    n_readout = np.array(n_readout)

    ####trying with model predict
    n_readout = []
    _ng = []
    predictions = model.predict(IQdata)
    for i, s in enumerate(signal_in[1]):  # |g>
        if predictions[i] == 1:
            _ng.append(1)
        else:
            _ng.append(0)
    n_readout.append(_ng)
    _ne = []
    for i, s in enumerate(signal_in[1]):  # |e>
        if predictions[i] == 2:
            _ne.append(1)
        else:
            _ne.append(0)
    n_readout.append(_ne)
    _nf = []
    for i, s in enumerate(signal_in[1]):  # |f>
        if predictions[i] == 0:
            _nf.append(1)
        else:
            _nf.append(0)
    n_readout.append(_nf)

    n_readout = np.array(n_readout)

    return n_readout


def find_threshold(daq_params, signal_in=[]):
    #
    thresholds = np.sort(daq_params.threshold)[::-1]
    thresholds = thresholds.tolist() + [-np.inf]
    analysis.fit_readout_histogram(rec_readout, hist_bins, hist_counts, num_gaussians=3)
    n_readout = []
    prev_threshold = np.inf

    for threshold in thresholds:
        _n = [1 if threshold < s and s < prev_threshold else 0 for s in signal_in]

        n_readout.append(_n)
        prev_threshold = threshold

    n_readout = np.array(n_readout)

    return n_readout


def make_avg_record_vs_patterns_plot(rec_avg_over_reps_a, rec_avg_over_reps_b):
    #
    f, ax = plt.subplots(2, 1)
    ax[0].plot(rec_avg_over_reps_a)
    ax[1].plot(rec_avg_over_reps_b)


def make_readout_histogram_for_each_pattern(rec_vs_pats=[]):
    rec_vs_pats = np.vstack(rec_vs_pats).T

    # definitions for the axes
    #    left, width = 0.1, 0.65
    #    bottom, height = 0.1, 0.65
    #    spacing = 0.005
    left, width = 0.5, 0.5
    bottom, height = left, width
    spacing = 0.01
    rect_hist = [left, bottom + height + spacing, width, 0.25]

    # start with a rectangular Figure
    plt.figure(figsize=(8, 8))

    ax_hist = plt.axes(rect_hist)
    ax_hist.tick_params(direction="in", labelbottom=True)

    # determine limits
    binwidth = 0.25
    lim_min = np.min([rec_vs_pats])
    lim_max = np.max([rec_vs_pats])

    # define bins
    bins = np.arange(lim_min, lim_max, binwidth)
    bins_cntr = 0.5 * bins[:-1] + 0.5 * bins[1:]

    # plot histogram for each pattern
    for rec in rec_vs_pats:
        counts_bins = np.histogram(rec, bins=bins)
        counts = counts_bins[0]
        ax_hist.plot(bins_cntr, counts)

    #
    #    ax.set_aspect('auto')
    plt.show()

    return (bins_cntr, counts)


def make_iq_plot(channel_a_and_b, pltitle: str):
    #
    ch_a = channel_a_and_b[0]
    ch_b = channel_a_and_b[1]

    #
    plt.figure(figsize=(8, 8))

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(figsize=(4, 4))

    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction="in", top=True, right=True)
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction="in", labelbottom=False)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction="in", labelleft=False)

    # the scatter plot:
    ax_scatter.scatter(ch_a, ch_b, c="black", marker=".", alpha=0.5)

    # now determine nice limits by hand:
    binwidth = 2.5
    lim_x_min = np.min([ch_a])
    lim_x_max = np.max([ch_a])
    lim_y_min = np.min([ch_b])
    lim_y_max = np.max([ch_b])

    ax_scatter.set_xlim((lim_x_min, lim_x_max))
    ax_scatter.set_ylim((lim_y_min, lim_y_max))

    bins_x = np.arange(lim_x_min, lim_x_max, binwidth)
    bins_y = np.arange(lim_y_min, lim_y_max, binwidth)

    counts_x = np.histogram(ch_a)
    counts_y = np.histogram(ch_b)

    hist_out_x = ax_histx.hist(
        ch_a, bins=bins_x, histtype="step", orientation="vertical"
    )
    hist_out_y = ax_histy.hist(
        ch_b, bins=bins_y, histtype="step", orientation="horizontal"
    )
    ax_histx.set_title("Ch 1")
    ax_histy.set_title("Ch 2")

    counts_x = hist_out_x[0]
    counts_y = hist_out_y[0]

    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())
    plt.show()

    bins_cntr_x = 0.5 * bins_x[:-1] + 0.5 * bins_x[1:]
    bins_cntr_y = 0.5 * bins_y[:-1] + 0.5 * bins_y[1:]

    bins_cntr = [bins_cntr_x, bins_cntr_y]
    counts = [counts_x, counts_y]
    return (bins_cntr, counts)


def make_n_state_iq_plot(rec_readout_vs_pats):

    #
    plt.figure(figsize=(8, 8))

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(figsize=(8, 8))

    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction="in", top=True, right=True)
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction="in", labelbottom=False)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction="in", labelleft=False)

    # the scatter plot:
    ch_a = rec_readout_vs_pats[0].T
    ch_b = rec_readout_vs_pats[1].T
    num_pats = len(ch_a)
    for k in range(num_pats):
        ax_scatter.scatter(ch_a[k], ch_b[k], marker=".", alpha=0.5)

    # now determine nice limits by hand:
    binwidth = 0.25
    lim_x_min = np.min(ch_a)
    lim_x_max = np.max(ch_a)
    lim_y_min = np.min(ch_b)
    lim_y_max = np.max(ch_b)

    ax_scatter.set_xlim((lim_x_min, lim_x_max))
    ax_scatter.set_ylim((lim_y_min, lim_y_max))

    bins_x = np.arange(lim_x_min, lim_x_max, binwidth)
    bins_y = np.arange(lim_y_min, lim_y_max, binwidth)
    bins_cntr_x = 0.5 * bins_x[:-1] + 0.5 * bins_x[1:]
    bins_cntr_y = 0.5 * bins_y[:-1] + 0.5 * bins_y[1:]
    bins_cntr = [bins_cntr_x, bins_cntr_y]

    #    #
    #    counts_x = np.histogram(rec_readout_vs_pats[0])
    #    counts_y = np.histogram(rec_readout_vs_pats[1])

    #
    counts_x = []
    counts_y = []
    for rec_a, rec_b in zip(ch_a, ch_b):
        #
        h_x = ax_histx.hist(rec_a, bins=bins_x, histtype="step", orientation="vertical")
        h_y = ax_histy.hist(
            rec_b, bins=bins_y, histtype="step", orientation="horizontal"
        )

        #
        counts_x.append(h_x[0])
        counts_y.append(h_y[1])

    #
    counts = [counts_x, counts_y]

    #
    ax_histx.set_title("Ch 1")
    ax_histy.set_title("Ch 2")
    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())

    plt.show()

    return bins_cntr, counts


def make_all_record_average_plots(all_avg_channel_a, all_avg_channel_b):
    f, ax = plt.subplots(2, 1)
    ax[0].plot(all_avg_channel_a)
    ax[1].plot(all_avg_channel_b)
    plt.show()


def save_csv(folder_name, data_in):
    # transpose data for loading into Igor
    data_to_save = np.vstack(data_in).T

    #
    cwd = os.getcwd()
    path_data_folder = os.path.abspath(os.path.join(cwd, os.pardir)) + "/data"

    if not os.path.isdir(path_data_folder):
        os.mkdir(path_data_folder)

    path_name_folder = (
        os.path.abspath(os.path.join(cwd, os.pardir)) + "/data" + "/" + folder_name
    )

    if not os.path.isdir(path_name_folder):
        os.mkdir(path_name_folder)

    time_when_saving = datetime.datetime.now()
    file_name = folder_name + "_{}".format(time_when_saving.strftime("%Y%m%d_%H%M%S"))
    file_path = path_name_folder + "/" + file_name + ".txt"

    header = ()  # ("N_list", "e_gnd", "e_gap")
    table = []
    for idx, d in enumerate(data_to_save):
        table.append(d)

    table.insert(0, header)

    f = open(file_path, "w", newline="")
    writer = csv.writer(f)
    writer.writerows(table)


if __name__ == "__main__":
    pass
