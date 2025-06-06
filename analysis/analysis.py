# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 11:18:02 2020.

@author: P. M. Harrington, 27 January 2020
"""
import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import integrate
from scipy.signal import argrelextrema
import csv
import analysis.fit_functions as fitfun
from scipy.fft import fft, fftfreq


class Nop:
    pass


def readout_fnotf(p_readout):
    num_pats_each = p_readout.shape[1] // 2

    p_readout_pi_f = p_readout.T[:num_pats_each].T
    p_readout_pi_t = p_readout.T[num_pats_each:].T

    p_readout_fnotf = []
    p_readout_fnotf.append(p_readout_pi_t[0])
    p_readout_fnotf.append(p_readout_pi_t[2])
    p_readout_fnotf.append(p_readout_pi_f[2])
    p_readout_fnotf = np.array(p_readout_fnotf)

    #    pf = 0.5 + 0.5*(p_readout_fnotf[2]-p_readout_fnotf[1])/(p_readout_fnotf[2]+p_readout_fnotf[1])
    #    pe = 1-pf
    #    p_post_fnotf = []
    #    p_post_fnotf.append(0*pf)
    #    p_post_fnotf.append(pe)
    #    p_post_fnotf.append(pf)
    #    p_post_fnotf = np.array(p_post_fnotf)

    pf = p_readout_fnotf[2] / (p_readout_fnotf[2] + p_readout_fnotf[1])
    pe = 1 - pf
    p_post_fnotf = []
    p_post_fnotf.append(0 * pf)
    p_post_fnotf.append(pe)
    p_post_fnotf.append(pf)
    p_post_fnotf = np.array(p_post_fnotf)

    return p_readout_fnotf, p_post_fnotf


def p_readout_postselected(p_readout=[None, None, None]):
    p_readout_post = [None] * 3

    p_e = p_readout[1]
    p_f = p_readout[2]
    p_readout_post[0] = 0 * p_e
    p_readout_post[1] = p_e / (p_e + p_f)
    p_readout_post[2] = p_f / (p_e + p_f)

    return p_readout_post


def p_readout_postselected_pief(p_readout=[None, None, None]):
    p_readout_post = [None] * 3

    length = int(len(p_readout[0]))

    #    p_e = p_readout[1][0:int(length/2)]
    p_e = p_readout[2][
        int(length / 2) : int(length)
    ]  # use f state to readout e state pop
    p_f = p_readout[2][0 : int(length / 2)]
    p_readout_post[0] = 0 * p_e
    p_readout_post[1] = p_e / (p_e + p_f)
    p_readout_post[2] = p_f / (p_e + p_f)

    return p_readout_post


def p_readout_scaled(p_readout):

    p_readout_scale_matrix = []
    length = int(len(p_readout[0]))

    fidelity_matrix = np.zeros((3, 3))

    p_readout_three_pnt_msmt = p_readout[:, -3:]

    for i in range(3):
        fidelity_matrix[i, :] = p_readout_three_pnt_msmt[i, :] / np.sum(
            p_readout_three_pnt_msmt[i, :]
        )

    fidelity_matrix_inverse = np.linalg.inv(fidelity_matrix)

    for j in range(length):
        p_readout_scale_matrix.append(fidelity_matrix_inverse @ p_readout[j, :])

    p_readout_scale_matrix_array = np.array(p_readout_scale_matrix)

    return fidelity_matrix, fidelity_matrix_inverse, p_readout_scale_matrix_array


def get_threshold_value_from_gaussians(a_vals, b_vals):
    #
    w_a = abs(a_vals[0])
    mu_a = a_vals[1]
    s_a = a_vals[2]
    #
    w_b = abs(b_vals[0])
    mu_b = b_vals[1]
    s_b = b_vals[2]

    #
    a = 1 / (2 * s_a**2) - 1 / (2 * s_b**2)
    b = -(mu_a / s_a**2 - mu_b / s_b**2)
    c = mu_a**2 / (2 * s_a**2) - mu_b**2 / (2 * s_b**2) + np.log(w_b / w_a)

    #
    x0 = -b / (2 * a) - np.sqrt(b**2 - 4 * a * c) / (2 * a)
    x1 = -b / (2 * a) + np.sqrt(b**2 - 4 * a * c) / (2 * a)

    return x0, x1


def fit_parabola(x_vals, y_vals, guess_vals=None):
    if guess_vals is None:
        guess_a = 1.0
        guess_b = 1.0
        guess_c = 1.0
        guess_vals = [guess_a, guess_b, guess_c]

    #
    fit = curve_fit(fitfun.parabola, x_vals, y_vals, p0=guess_vals)
    y_vals_fit = fitfun.parabola(x_vals, *fit[0])

    plt.figure(figsize=(6, 4))
    plt.plot(x_vals, y_vals)
    plt.plot(x_vals, y_vals_fit)
    plt.show()

    print("\n")
    var_str = ["a", "b", "c"]
    for idx, v in enumerate(var_str):
        print(v + ": {}".format(fit[0][idx]))

    print("min (fit): {}".format(-fit[0][1] / (2 * fit[0][0])))

    return (fit, y_vals_fit)


def fit_exp_decay(x_vals, y_vals, guess_vals=None):
    if guess_vals is None:
        guess_amplitude = -1.0
        guess_gamma = 1.0
        guess_offset = 0
        guess_vals = [guess_amplitude, guess_gamma, guess_offset]

    #
    popt, pcov = curve_fit(fitfun.exp_decay, x_vals, y_vals, p0=guess_vals)
    perr = np.sqrt(np.diag(pcov))
    y_vals_fit = fitfun.exp_decay(x_vals, *popt)

    plt.figure(figsize=(6, 4))
    plt.plot(x_vals, y_vals)
    plt.plot(x_vals, y_vals_fit)
    plt.show()

    print("\n")
    var_str = ["   amp", " gamma", "offset"]
    for idx, v in enumerate(var_str):
        print(v + ": {}".format(popt[idx]))

    return (popt, perr, y_vals_fit, pcov)


# Gives us fit from the functions written in fit_functions python file
def fit_exp_10(x_vals, y_vals, guess_vals=None):
    if guess_vals is None:
        guess_amplitude = -1.0
        guess_gamma = -0.1
        guess_offset = 0
        guess_vals = [guess_amplitude, guess_gamma, guess_offset]

    # popt contains the constants of the fitted functions as an array
    # pcov contains the covariant matrix/array for the error of the fit.
    popt, pcov = curve_fit(fitfun.exp_10, x_vals, y_vals, p0=guess_vals, maxfev=3000)
    perr = np.sqrt(np.diag(pcov))

    # Using the values in popt, this plugs in different x values into the fitted equation and stores them
    y_vals_fit = fitfun.exp_10(x_vals, *popt)

    plt.figure(figsize=(6, 4))
    plt.plot(x_vals, y_vals)
    plt.plot(x_vals, y_vals_fit)
    plt.show()

    print("\n")
    var_str = ["   amp", " gamma", "offset"]
    for idx, v in enumerate(var_str):
        print(v + ": {}".format(popt[idx]))

    return (popt, perr, y_vals_fit, pcov)


def fit_exp_decay_with_decay_time(x_vals, y_vals, guess_vals=None):
    if guess_vals is None:
        guess_amplitude = -1.0
        guess_T = 1.0
        guess_offset = 0
        guess_vals = [guess_amplitude, guess_T, guess_offset]

    #
    popt, pcov = curve_fit(
        fitfun.exp_decay_with_decay_time, x_vals, y_vals, p0=guess_vals, maxfev=3000
    )
    perr = np.sqrt(np.diag(pcov))
    y_vals_fit = fitfun.exp_decay_with_decay_time(x_vals, *popt)

    plt.figure(figsize=(6, 4))
    plt.plot(x_vals, y_vals)
    plt.plot(x_vals, y_vals_fit)
    plt.show()

    print("\n")
    var_str = ["   amp", " decay time", "offset"]
    for idx, v in enumerate(var_str):
        print(v + ": {}".format(popt[idx]))

    return (popt, perr, y_vals_fit, pcov)


def fit_double_exp_decay(x_vals, y_vals, guess_vals=None):
    if guess_vals is None:
        guess_amplitude1 = -1.0
        guess_gamma1 = 3.0
        guess_amplitude2 = -1.0
        guess_gamma2 = 6.0
        guess_offset = 0
        guess_vals = [
            guess_amplitude1,
            guess_gamma1,
            guess_amplitude2,
            guess_gamma2,
            guess_offset,
        ]

    #
    popt, pcov = curve_fit(fitfun.double_exp_decay, x_vals, y_vals, p0=guess_vals)
    perr = np.sqrt(np.diag(pcov))
    y_vals_fit = fitfun.double_exp_decay(x_vals, *popt)

    plt.figure(figsize=(6, 4))
    plt.plot(x_vals, y_vals)
    plt.plot(x_vals, y_vals_fit)
    plt.show()
    print("test")
    print("\n")
    var_str = ["  amp1", "gamma1", "  amp2", "gamma2", "offset"]
    for idx, v in enumerate(var_str):
        print(v + ": {}".format(popt[idx]))

    return (popt, perr, y_vals_fit, pcov)


def fit_sine_decay(x_vals, y_vals, guess_vals=None):
    if guess_vals is None:
        guess_freq_Hz = 5
        guess_gamma = 5
        guess_amplitude = 0.5
        guess_phase_deg = -90
        guess_offset = 0.5

        guess_vals = [
            guess_freq_Hz,
            guess_gamma,
            guess_amplitude,
            guess_phase_deg,
            guess_offset,
        ]

    #
    popt = np.full(len(guess_vals), np.nan)
    pcov = np.full((len(guess_vals), len(guess_vals)), np.nan)
    y_vals_fit = np.full(len(x_vals), np.nan)
    try:
        popt, pcov = curve_fit(fitfun.sine_decay, x_vals, y_vals, p0=guess_vals)
        y_vals_fit = fitfun.sine_decay(x_vals, *popt)
    except RuntimeError:
        print("RuntimeError")

    perr = np.sqrt(np.diag(pcov))

    y_vals_fit = fitfun.sine_decay(x_vals, *popt)

    plt.figure(figsize=(6, 4))
    plt.plot(x_vals, y_vals)
    plt.plot(x_vals, y_vals_fit)
    plt.show()

    print("\n")
    var_str = ["  freq", " gamma", "   amp", " phase", "offset"]
    for idx, v in enumerate(var_str):
        print(v + ": {} +/- {}".format(popt[idx], perr[idx]))

    print("pi_pulse time" + ": {} +/- {}".format(1 / 2 / popt[0], perr[0]))
    return (popt, perr, y_vals_fit, pcov)

def fit_sine_square_decay(x_vals, y_vals, guess_vals=None,lower_bounds=[0, 0.3,0, -np.inf, -np.inf], upper_bounds=[.3,0.6,np.inf, np.inf, np.inf]):
    if guess_vals is None:
        guess_freq_Hz = 5
        guess_gamma = 5
        guess_amplitude = 0.5
        guess_phase_deg = -90
        guess_offset = 0.5

        guess_vals = [
            guess_freq_Hz,
            guess_gamma,
            guess_amplitude,
            guess_phase_deg,
            guess_offset,
        ]

    #,
    popt = np.full(len(guess_vals), np.nan)
    pcov = np.full((len(guess_vals), len(guess_vals)), np.nan)
    y_vals_fit = np.full(len(x_vals), np.nan)
    

    try:
        popt, pcov = curve_fit(fitfun.sine_square_decay, x_vals, y_vals,bounds=(lower_bounds, upper_bounds) )
        y_vals_fit = fitfun.sine_square_decay(x_vals, *popt)
    except RuntimeError:
        print("RuntimeError")

    perr = np.sqrt(np.diag(pcov))

    y_vals_fit = fitfun.sine_square_decay(x_vals, *popt)

    plt.figure(figsize=(3, 2))
    plt.plot(x_vals, y_vals)
    plt.plot(x_vals, y_vals_fit)
    plt.show()

    print("freq:{},gamma: {}, amp:{}, phase_deg: {},offset: {},".format(popt[0],popt[1],popt[2], popt[3], popt[4]))#


    
    return (popt, perr, y_vals_fit, pcov)

def fit_sine_decay_with_decay_time(x_vals, y_vals, guess_vals=None):
    if guess_vals is None:
        guess_freq_Hz = 5
        guess_T = 5
        guess_amplitude = 0.5
        guess_phase_deg = -90
        guess_offset = 0.5

        guess_vals = [
            guess_freq_Hz,
            guess_T,
            guess_amplitude,
            guess_phase_deg,
            guess_offset,
        ]

    #
    popt = np.full(len(guess_vals), np.nan)
    pcov = np.full((len(guess_vals), len(guess_vals)), np.nan)
    y_vals_fit = np.full(len(x_vals), np.nan)
    try:
        popt, pcov = curve_fit(
            fitfun.sine_decay_with_decay_time, x_vals, y_vals, p0=guess_vals
        )
        y_vals_fit = fitfun.sine_decay_with_decay_time(x_vals, *popt)
    except RuntimeError:
        print("RuntimeError")

    perr = np.sqrt(np.diag(pcov))

    y_vals_fit = fitfun.sine_decay_with_decay_time(x_vals, *popt)

    plt.figure(figsize=(6, 4))
    plt.plot(x_vals, y_vals)
    plt.plot(x_vals, y_vals_fit)
    plt.show()

    print("\n")
    var_str = ["  freq", " T2*", "   amp", " phase", "offset"]
    for idx, v in enumerate(var_str):
        print(v + ": {} +/- {}".format(popt[idx], perr[idx]))

    print("pi_pulse time" + ": {} +/- {}".format(1 / 2 / popt[0], perr[0]))
    return (popt, perr, y_vals_fit, pcov)


def fit_sine_decay_with_modulation(x_vals, y_vals, guess_vals=None):
    if guess_vals is None:
        guess_freq_Hz1 = 5
        guess_freq_Hz2 = 5
        guess_gamma = 5
        guess_amplitude1 = 0.5
        guess_amplitude2 = 0.5
        guess_phase_deg1 = -90
        guess_phase_deg2 = -90
        guess_offset = 0.5

        guess_vals = [
            guess_freq_Hz1,
            guess_freq_Hz2,
            guess_gamma,
            guess_amplitude1,
            guess_amplitude2,
            guess_phase_deg1,
            guess_phase_deg2,
            guess_offset,
        ]

    #
    popt = np.full(len(guess_vals), np.nan)
    pcov = np.full((len(guess_vals), len(guess_vals)), np.nan)
    y_vals_fit = np.full(len(x_vals), np.nan)
    try:
        popt, pcov = curve_fit(
            fitfun.sine_decay_with_modulation, x_vals, y_vals, p0=guess_vals
        )
        y_vals_fit = fitfun.sine_decay_with_modulation(x_vals, *popt)
    except RuntimeError:
        print("RuntimeError")

    perr = np.sqrt(np.diag(pcov))

    y_vals_fit = fitfun.sine_decay_with_modulation(x_vals, *popt)

    plt.figure(figsize=(6, 4))
    plt.plot(x_vals, y_vals)
    plt.plot(x_vals, y_vals_fit)
    plt.show()

    print("\n")
    var_str = [
        "  freq1",
        "  freq2",
        " gamma",
        "   amp1",
        "   amp2",
        " phase1",
        " phase2",
        "offset",
    ]
    for idx, v in enumerate(var_str):
        print(v + ": {} +/- {}".format(popt[idx], perr[idx]))

    return (popt, perr, y_vals_fit, pcov)


def fit_lorentzian(x_vals, y_vals, guess_vals=None):
    if guess_vals is None:
        guess_amplitude = -1.0
        guess_b = 1.0
        guess_offset = 0
        guess_freq = 7.3
        guess_vals = [guess_amplitude, guess_b, guess_offset, guess_freq]

    #
    popt, pcov = curve_fit(fitfun.lorentzian, x_vals, y_vals, p0=guess_vals)
    perr = np.sqrt(np.diag(pcov))
    y_vals_fit = fitfun.lorentzian(x_vals, *popt)

    plt.figure(figsize=(6, 4))
    plt.plot(x_vals, y_vals)
    plt.plot(x_vals, y_vals_fit)
    plt.show()

    print("\n")
    var_str = ["   amp", " b", "offset", "  freq"]
    for idx, v in enumerate(var_str):
        print(v + ": {}".format(popt[idx]))

    return (popt, perr, y_vals_fit, pcov)


def fit_line(x_vals, y_vals, guess_vals=None):
    if guess_vals is None:
        guess_m = -1.0
        guess_b = 1.0
        guess_vals = [guess_m, guess_b]

    #
    popt, pcov = curve_fit(fitfun.line, x_vals, y_vals, p0=guess_vals)
    perr = np.sqrt(np.diag(pcov))
    y_vals_fit = fitfun.line(x_vals, *popt)

    plt.figure(figsize=(6, 4))
    plt.plot(x_vals, y_vals)
    plt.plot(x_vals, y_vals_fit)
    plt.show()

    print("\n")
    var_str = ["m", "b"]
    for idx, v in enumerate(var_str):
        print(v + ": {}".format(popt[idx]))

    return (popt, perr, y_vals_fit, pcov)


def fit_readout_histogram(rec_readout, hist_bins, hist_counts, num_gaussians=3):
    #    w0 = max(hist_counts)/3
    #    w1 = max(hist_counts)/3
    #    w2 = max(hist_counts)/3
    w0 = 100
    w1 = 100
    w2 = 100
    s0 = 0.2 * np.std(rec_readout)
    s1 = 0.2 * np.std(rec_readout)
    s2 = 0.2 * np.std(rec_readout)
    mu0 = -160.0  # np.mean(rec_readout) - 1*s0
    mu1 = -153.0  # np.mean(rec_readout)
    mu2 = -142.0  # np.mean(rec_readout) + 1*s0
    #    mu0 = np.mean(rec_readout) - 0.5*s0
    #    mu1 = np.mean(rec_readout)
    #    mu2 = np.mean(rec_readout) + 0.5*s0

    #    guess_vals = [w0, w2, mu0, mu2, s0, s2]
    guess_vals = [w0, w1, w2, mu0, mu1, mu2, s0, s1, s2]

    #    #
    #    fit = curve_fit(fitfun.two_gaussians, hist_bins, hist_counts, p0=guess_vals)
    #    hist_fit = fitfun.two_gaussians(hist_bins, *fit[0])

    #
    fit = curve_fit(fitfun.three_gaussians, hist_bins, hist_counts, p0=guess_vals)
    hist_fit = fitfun.three_gaussians(hist_bins, *fit[0])

    #
    plt.figure(figsize=(6, 4))
    plt.plot(hist_bins, hist_counts)
    plt.plot(hist_bins, hist_fit)

    ## get individual gaussians
    for k in range(num_gaussians):
        f = fitfun.gaussian(hist_bins, *fit[0][k::3])
        plt.plot(hist_bins, f)

    get_threshold_value_from_gaussians(fit[0][0::3], fit[0][1::3])
    get_threshold_value_from_gaussians(fit[0][1::3], fit[0][2::3])

    return fit, hist_fit


def fit_three_gaussian(hist_bins, hist_counts):

    # things to do:

    # hist_counts can be first smoothed, to help to find the desired three peaks (otherwise may stuck to local minima with values close to zero)
    # doesn't work if only work in g-e manifold

    maxInd = argrelextrema(
        hist_counts, np.greater, order=10
    )  # serach a local maxima within [N-order, N+order] range

    if len(maxInd[0]) == 3:
        [w0, w1, w2] = hist_counts[maxInd]
        [mu0, mu1, mu2] = hist_bins[maxInd]

        s0 = 0.9
        s1 = 1
        s2 = 1.1
        guess_vals = [w0, w1, w2, mu0, mu1, mu2, s0, s1, s2]

    else:
        guess_vals = [3000, 2000, 5000, 143, 153, 162, 2, 1.5, 1.5]
    #    count = 0
    #    while len(maxInd[0]) < 3:
    #        maxInd = argrelextrema(hist_counts,np.greater, order = 8+count)
    #        count = count + 1
    #        if count == 10:
    #            break
    #    guess_vals = [3000,2000,5000,147,157,162,2,2.5,1.5]
    #
    fit = curve_fit(fitfun.three_gaussians, hist_bins, hist_counts, p0=guess_vals)
    hist_fit = fitfun.three_gaussians(hist_bins, *fit[0])

    #
    plt.figure(figsize=(6, 4))
    plt.plot(hist_bins, hist_counts)
    plt.plot(hist_bins, hist_fit)
    #    plt.ylim(0,2000)

    ## get individual gaussians
    for k in range(3):
        f = fitfun.gaussian(hist_bins, *fit[0][k::3])
        plt.plot(hist_bins, f)
    plt.show()

    # search for threshold (intersection point of the Gaussians)
    mu0_fit_index = np.argmin(np.abs(hist_bins - fit[0][3]))
    mu1_fit_index = np.argmin(np.abs(hist_bins - fit[0][4]))
    mu2_fit_index = np.argmin(np.abs(hist_bins - fit[0][5]))

    threshold_01_index = mu0_fit_index + np.argmin(
        abs(
            fitfun.gaussian(hist_bins[mu0_fit_index:mu1_fit_index], *fit[0][0::3])
            - fitfun.gaussian(hist_bins[mu0_fit_index:mu1_fit_index], *fit[0][1::3])
        )
    )
    threshold_12_index = mu1_fit_index + np.argmin(
        abs(
            fitfun.gaussian(hist_bins[mu1_fit_index:mu2_fit_index], *fit[0][1::3])
            - fitfun.gaussian(hist_bins[mu1_fit_index:mu2_fit_index], *fit[0][2::3])
        )
    )
    threshold = [hist_bins[threshold_01_index], hist_bins[threshold_12_index]]

    #    threshold_ge_index = maxInd[0][0] + np.argmin(abs(fitfun.gaussian(hist_bins[maxInd[0][0]:maxInd[0][1]], *fit[0][0::3])-fitfun.gaussian(hist_bins[maxInd[0][0]:maxInd[0][1]], *fit[0][1::3])))
    #    threshold_ef_index = maxInd[0][1] + np.argmin(abs(fitfun.gaussian(hist_bins[maxInd[0][1]:maxInd[0][2]], *fit[0][1::3])-fitfun.gaussian(hist_bins[maxInd[0][1]:maxInd[0][2]], *fit[0][2::3])))
    #    threshold = [hist_bins[threshold_ge_index],hist_bins[threshold_ef_index]]

    #    temp1 = fitfun.gaussian(hist_bins[maxInd[0][0]:maxInd[0][1]], *fit[0][0::3])-fitfun.gaussian(hist_bins[maxInd[0][0]:maxInd[0][1]], *fit[0][1::3])
    #    plt.plot(hist_bins[maxInd[0][0]:maxInd[0][1]],temp1)
    #    plt.show
    return threshold


def fit_gaussian(rec_readout, hist_bins, hist_counts):
    w0 = max(hist_counts)
    s0 = np.std(rec_readout)
    mu0 = np.mean(rec_readout)

    guess_vals = [w0, mu0, s0]

    #
    fit = curve_fit(fitfun.gaussian, hist_bins, hist_counts, p0=guess_vals)
    hist_fit = fitfun.gaussian(hist_bins, *fit[0])

    #
    plt.figure(figsize=(6, 4))
    plt.plot(hist_bins, hist_counts)
    plt.plot(hist_bins, hist_fit)

    ## get individual gaussians
    f = fitfun.gaussian(hist_bins, *fit[0])
    plt.plot(hist_bins, f)
    plt.show()

    return fit, hist_fit


def fit_gaussian_no_plot(rec_readout, hist_bins, hist_counts, ax_hist):
    w0 = max(hist_counts)
    s0 = np.std(rec_readout)
    mu0 = np.mean(rec_readout)

    guess_vals = [w0, mu0, s0]

    #
    popt, pcov = curve_fit(fitfun.gaussian, hist_bins, hist_counts, p0=guess_vals)
    perr = np.sqrt(np.diag(pcov))
    hist_fit = fitfun.gaussian(hist_bins, *popt)

    #
    # plt.figure(figsize=(6,4))
    # plt.plot(hist_bins, hist_counts)
    # ax_hist.plot(hist_bins, hist_fit)

    ## get individual gaussians
    f = fitfun.gaussian(hist_bins, *popt)
    ax_hist.plot(hist_bins, f)
    # plt.show()

    return popt


def fit_gaussian_points(x_vals, y_vals, guess_vals=None):
    if guess_vals is None:
        guess_w0 = 0.06
        guess_mu0 = 1.3
        guess_s0 = 0.0005
        guess_vals = [guess_w0, guess_mu0, guess_s0]
        # w0, mu0, s0

    popt, pcov = curve_fit(fitfun.gaussian, x_vals, y_vals, p0=guess_vals)

    perr = np.sqrt(np.diag(pcov))
    y_vals_fit = fitfun.gaussian(x_vals, *popt)

    plt.figure(figsize=(6, 4))
    plt.plot(x_vals, y_vals)
    plt.plot(x_vals, y_vals_fit)
    plt.show()

    print("\n")
    var_str = ["   amp", " mu", "sigma"]
    for idx, v in enumerate(var_str):
        print(v + ": {}".format(popt[idx]))

    return (popt, perr, y_vals_fit, pcov)


def save_csv(folder_name=None, file_name=None, data_in=[]):
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
    if file_name is not None:
        file_name = (
            folder_name
            + file_name
            + "_{}".format(time_when_saving.strftime("%Y%m%d_%H%M%S"))
        )
    else:
        file_name = folder_name + "_{}".format(
            time_when_saving.strftime("%Y%m%d_%H%M%S")
        )
    file_path = path_name_folder + "/" + file_name + ".txt"

    header = ()  # ("N_list", "e_gnd", "e_gap")
    table = []
    for idx, d in enumerate(data_to_save):
        table.append(d)

    table.insert(0, header)

    f = open(file_path, "w", newline="")
    writer = csv.writer(f)
    writer.writerows(table)


def fit_two_gaussian(hist_bins, hist_counts):
    maxInd = argrelextrema(
        hist_counts, np.greater, order=10
    )  # serach a local maxima within [N-order, N+order] range

    if len(maxInd[0]) == 2:
        [w0, w1] = hist_counts[maxInd]
        [mu0, mu1] = hist_bins[maxInd]

        s0 = 0.9
        s1 = 1
        guess_vals = [w0, w1, mu0, mu1, s0, s1]

    else:
        guess_vals = [
            2000,
            5000,
            155,
            162,
            2.5,
            1.5,
        ]  ##Not sure about this guess value, which ones to exclude

    #    count = 0
    #    while len(maxInd[0]) < 3:
    #        maxInd = argrelextrema(hist_counts,np.greater, order = 8+count)
    #        count = count + 1
    #        if count == 10:
    #            break

    #
    fit = curve_fit(fitfun.two_gaussians, hist_bins, hist_counts, p0=guess_vals)
    hist_fit = fitfun.two_gaussians(hist_bins, *fit[0])

    #
    plt.figure(figsize=(6, 4))
    plt.plot(hist_bins, hist_counts)  # Plot raw data
    plt.plot(hist_bins, hist_fit)  # Plot the joined gaussian

    ## get individual gaussians
    for k in range(2):
        f = fitfun.gaussian(hist_bins, *fit[0][k::2])
        plt.plot(hist_bins, f)
    plt.show()

    # search for threshold (intersection point of the Gaussians)
    mu0_fit_index = np.argmin(np.abs(hist_bins - fit[0][2]))
    mu1_fit_index = np.argmin(np.abs(hist_bins - fit[0][3]))

    threshold_01_index = mu0_fit_index + np.argmin(
        abs(
            fitfun.gaussian(hist_bins[mu0_fit_index:mu1_fit_index], *fit[0][0::2])
            - fitfun.gaussian(hist_bins[mu0_fit_index:mu1_fit_index], *fit[0][1::2])
        )
    )
    threshold = [0, hist_bins[threshold_01_index]]  # default 0

    #    threshold_ge_index = maxInd[0][0] + np.argmin(abs(fitfun.gaussian(hist_bins[maxInd[0][0]:maxInd[0][1]], *fit[0][0::3])-fitfun.gaussian(hist_bins[maxInd[0][0]:maxInd[0][1]], *fit[0][1::3])))
    #    threshold_ef_index = maxInd[0][1] + np.argmin(abs(fitfun.gaussian(hist_bins[maxInd[0][1]:maxInd[0][2]], *fit[0][1::3])-fitfun.gaussian(hist_bins[maxInd[0][1]:maxInd[0][2]], *fit[0][2::3])))
    #    threshold = [hist_bins[threshold_ge_index],hist_bins[threshold_ef_index]]

    #    temp1 = fitfun.gaussian(hist_bins[maxInd[0][0]:maxInd[0][1]], *fit[0][0::3])-fitfun.gaussian(hist_bins[maxInd[0][0]:maxInd[0][1]], *fit[0][1::3])
    #    plt.plot(hist_bins[maxInd[0][0]:maxInd[0][1]],temp1)
    #    plt.show
    return threshold


def fit_sine(x_vals, y_vals, guess_vals=None):
    if guess_vals is None:
        guess_freq_Hz = 1 / len(x_vals)
        guess_amplitude = 1.0
        guess_phase_deg = y_vals[0] * 180 / np.pi
        guess_offset = 0.5
        guess_vals = [guess_freq_Hz, guess_amplitude, guess_phase_deg, guess_offset]

    #

    popt, pcov = curve_fit(fitfun.sine, x_vals, y_vals, p0=guess_vals)
    perr = np.sqrt(np.diag(pcov))
    y_vals_fit = fitfun.sine(x_vals, *popt)

    plt.figure(figsize=(6, 4))
    plt.plot(x_vals, y_vals)
    plt.plot(x_vals, y_vals_fit)
    plt.show()

    print("\n")
    var_str = ["freq", " amp", "phase", "offset"]
    for idx, v in enumerate(var_str):
        print(v + ": {}".format(popt[idx]))

    return (popt, perr, y_vals_fit, pcov)


def fit_sine_fix_freq(x_vals, y_vals, guess_vals=None, fixed_freq=0.1):
    popt, pcov = curve_fit(
        lambda timep, amplitude2, phase_rad2, offset2: fitfun.sine(
            timep, fixed_freq, amplitude2, phase_rad2, offset2
        ),
        x_vals,
        y_vals,
        p0=guess_vals,
    )
    perr = np.sqrt(np.diag(pcov))
    y_vals_fit = fitfun.sine(x_vals, fixed_freq, *popt)

    plt.figure(figsize=(6, 4))
    plt.plot(x_vals, y_vals)
    plt.plot(x_vals, y_vals_fit)
    plt.show()

    print("\n")
    var_str = [" amp", "phase", "offset"]
    for idx, v in enumerate(var_str):
        print(v + ": {}".format(popt[idx]))

    return (popt, perr, y_vals_fit, pcov)


if __name__ == "__main__":
    pass
