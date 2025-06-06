import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import analysis.analysis as analysis
import analysis.fit_functions as fitfuncs
from scipy.optimize import curve_fit


def get_IQ_averages(values):
    """This function takes the values input, returns your IQ averages."""
    I1 = values.rec_avg_vs_pats_1[0]
    Q1 = values.rec_avg_vs_pats_1[1]
    I2 = values.rec_avg_vs_pats_2[0]
    Q2 = values.rec_avg_vs_pats_2[1]

    return pd.DataFrame({"I1": I1, "Q1": Q1, "I2": I2, "Q2": Q2})


def get_IQ_raw(values):
    """This function takes the values input, returns your IQ averages."""
    I1 = values.rec_readout_vs_pats_1[0]
    Q1 = values.rec_readout_vs_pats_2[1]
    I2 = values.rec_readout_vs_pats_2[0]
    Q2 = values.rec_readout_vs_pats_2[1]

    return pd.DataFrame({"I1": I1, "Q1": Q1, "I2": I2, "Q2": Q2})


def spectroscopy_plot(
    freq_list: list, values: dict, vert_line_value: list = None, qubit_num=1
):
    """
    For a given qubit spectroscopy, plot the IQ values on side-by-side subplots
    with optional vertical lines.

    Args:
        freq_list: list of frequencies used in spectroscopy.
        values: dictionary from readout.
        vert_line_value: optional list with two values. The first is the vertical line position for the I channel,
                         and the second is for the Q channel.
        qubit_num: qubit number to run spectroscopy on.
    """
    IQ_vals = get_IQ_averages(values)
    IQ_vals["freq"] = freq_list

    # Create a figure with two side-by-side subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), dpi=150)

    # Plot the I channel on the first subplot
    sns.lineplot(data=IQ_vals, x="freq", y=f"I{qubit_num}", ax=axs[0])
    axs[0].set_title(f"Qubit {qubit_num} I-channel")
    axs[0].set_xlabel("Frequency")
    axs[0].set_ylabel("I Value")

    # Optionally add a vertical line to the I channel plot
    if vert_line_value[0] is not None and len(vert_line_value) > 0:
        axs[0].axvline(
            x=vert_line_value[0],
            color="red",
            linestyle="--",
            label=f"ssm = {vert_line_value[0]}",
        )

    # Plot the Q channel on the second subplot
    sns.lineplot(data=IQ_vals, x="freq", y=f"Q{qubit_num}", ax=axs[1])
    axs[1].set_title(f"Qubit {qubit_num} Q-channel")
    axs[1].set_xlabel("Frequency")
    axs[1].set_ylabel("Q Value")

    # Optionally add a vertical line to the Q channel plot
    if vert_line_value[1] is not None and len(vert_line_value) > 1:
        axs[1].axvline(
            x=vert_line_value[1],
            color="red",
            linestyle="--",
            label=f"ssm = {vert_line_value[1]}",
        )

    plt.tight_layout()
    plt.show()


def mod_spectroscopy_plot(freq_list: list, values: dict, vert_line_value: list = None):
    """
    For a givencoupler spectroscopy, plot the IQ values on side-by-side
    subplots with optional vertical lines.

    Args:
        freq_list: list of frequencies used in spectroscopy.
        values: dictionary from readout.
        vert_line_value: optional list with two values. The first is the vertical line position for the I channel,
                         and the second is for the Q channel.
        qubit_num: qubit number to run spectroscopy on.
    """
    IQ_vals = get_IQ_averages(values)
    IQ_vals["freq"] = freq_list

    # Create a figure with two side-by-side subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), dpi=150)

    # Plot the I channel on the first subplot
    sns.lineplot(data=IQ_vals, x="freq", y=f"I{1}", ax=axs[0, 0])
    axs[0, 0].set_title(f"Qubit {1} I-channel")
    axs[0, 0].set_xlabel("Frequency")
    axs[0, 0].set_ylabel("I Value")

    # Optionally add a vertical line to the I channel plot
    if vert_line_value[0] is not None and len(vert_line_value) > 0:
        axs[0, 0].axvline(
            x=vert_line_value[0],
            color="red",
            linestyle="--",
            label=f"ssm = {vert_line_value[0]}",
        )

    # Plot the Q channel on the second subplot
    sns.lineplot(data=IQ_vals, x="freq", y=f"Q{1}", ax=axs[1, 0])
    axs[1, 0].set_title(f"Qubit {1} Q-channel")
    axs[1, 0].set_xlabel("Frequency")
    axs[1, 0].set_ylabel("Q Value")

    # Optionally add a vertical line to the Q channel plot
    if vert_line_value[1] is not None and len(vert_line_value) > 0:
        axs[1, 0].axvline(
            x=vert_line_value[1],
            color="red",
            linestyle="--",
            label=f"ssm = {vert_line_value[1]}",
        )

    # Plot the I channel on the first subplot
    sns.lineplot(data=IQ_vals, x="freq", y=f"I{2}", ax=axs[0, 1])
    axs[0, 1].set_title(f"Qubit {2} I-channel")
    axs[0, 1].set_xlabel("Frequency")
    axs[0, 1].set_ylabel("I Value")

    # Optionally add a vertical line to the I channel plot
    if vert_line_value[0] is not None and len(vert_line_value) > 0:
        axs[0, 1].axvline(
            x=vert_line_value[0],
            color="red",
            linestyle="--",
            label=f"ssm = {vert_line_value[0]}",
        )

    # Plot the Q channel on the second subplot
    sns.lineplot(data=IQ_vals, x="freq", y=f"Q{2}", ax=axs[1, 1])
    axs[1, 1].set_title(f"Qubit {2} Q-channel")
    axs[1, 1].set_xlabel("Frequency")
    axs[1, 1].set_ylabel("Q Value")

    # Optionally add a vertical line to the Q channel plot
    if vert_line_value[1] is not None and len(vert_line_value) > 1:
        axs[1, 1].axvline(
            x=vert_line_value[1],
            color="red",
            linestyle="--",
            label=f"ssm = {vert_line_value[1]}",
        )

    plt.tight_layout()
    plt.show()


def rabi_plot(sweep_time, num_steps, values, qubit_num=1):
    IQ_data = get_IQ_averages(values)
    Q = IQ_data[f"Q{qubit_num}"]
    I = IQ_data[f"I{qubit_num}"]
    Qrange = abs(np.max(Q) - np.min(Q))
    Irange = abs(np.max(I) - np.min(I))
    if Qrange > Irange:
        times = np.linspace(0, sweep_time / 1000, num_steps)
        pi_ge_fit_vals, _, _, _ = analysis.fit_sine_decay(
            times, Q, guess_vals=[11, 0.3, np.abs(np.max(Q) - np.min(Q)), 38, Q[0]]
        )
        pi_ge = abs((1 / 2 / pi_ge_fit_vals[0]) * 1000)
        print("\u03C0_ge time = {} ns".format(pi_ge))
    else:
        times = np.linspace(0, sweep_time / 1000, num_steps)
        pi_ge_fit_vals, _, _, _ = analysis.fit_sine_decay(
            times, I, guess_vals=[11, 0.3, np.abs(np.max(I) - np.min(I)), 38, I[0]]
        )
        pi_ge = abs((1 / 2 / pi_ge_fit_vals[0]) * 1000)
        print("\u03C0_ge time = {} ns".format(pi_ge))


def gaussian(x, mu, std, A):
    """Gaussian function for curve fitting."""
    return A * np.exp(-((x - mu) ** 2) / (2 * std**2))


def fit_gaussian(data, bins):
    """Fit a Gaussian to the data and return mu, std, and the fit curve."""
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    p0 = [np.mean(data), np.std(data), np.max(counts)]  # Initial guess for mu, std, A
    popt, _ = curve_fit(gaussian, bin_centers, counts, p0=p0)
    return (
        popt[0],
        popt[1],
        gaussian(bin_centers, *popt),
        bin_centers,
    )  # mu, std, fit curve


def plot_IQ_histograms(readout_vs_pats, qubit_num):
    """
    Plot I vs Q histograms for both ground (g) and excited (e) states on the
    same plot.

    Returns fit parameters (mu, std) for each distribution.
    """
    # Unpack the data
    I_g = readout_vs_pats[0, :, 0]  # I ground state
    I_e = readout_vs_pats[0, :, 1]  # I excited state
    Q_g = readout_vs_pats[1, :, 0]  # Q ground state
    Q_e = readout_vs_pats[1, :, 1]  # Q excited state

    # Fit Gaussians to the data
    mu_I_g, std_I_g, fit_Ig, bin_centers_Ig = fit_gaussian(I_g, bins=50)
    mu_I_e, std_I_e, fit_Ie, bin_centers_Ie = fit_gaussian(I_e, bins=50)
    mu_Q_g, std_Q_g, fit_Qg, bin_centers_Qg = fit_gaussian(Q_g, bins=50)
    mu_Q_e, std_Q_e, fit_Qe, bin_centers_Qe = fit_gaussian(Q_e, bins=50)

    # Return fit parameters
    fit_params = {
        "I_g": {"mu": mu_I_g, "std": std_I_g},
        "I_e": {"mu": mu_I_e, "std": std_I_e},
        "Q_g": {"mu": mu_Q_g, "std": std_Q_g},
        "Q_e": {"mu": mu_Q_e, "std": std_Q_e},
    }

    # Create figure and axes
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), dpi=150)

    # Plot I histograms
    axs[0].hist(I_g, bins=200, histtype="step", label=f"I{qubit_num} (g)", color="blue")
    axs[0].hist(I_e, bins=200, histtype="step", label=f"I{qubit_num} (e)", color="red")
    axs[0].plot(bin_centers_Ig, fit_Ig, color="blue")
    axs[0].plot(bin_centers_Ie, fit_Ie, color="red")
    axs[0].set_title(f"Qubit {qubit_num} I-channel Histogram")
    axs[0].set_xlabel("I Value")
    axs[0].set_ylabel("Counts")
    axs[0].legend()

    # Plot Q histograms
    axs[1].hist(Q_g, bins=200, histtype="step", label=f"Q{qubit_num} (g)", color="blue")
    axs[1].hist(Q_e, bins=200, histtype="step", label=f"Q{qubit_num} (e)", color="red")
    axs[1].plot(bin_centers_Qg, fit_Qg, color="blue")
    axs[1].plot(bin_centers_Qe, fit_Qe, color="red")
    axs[1].set_title(f"Qubit {qubit_num} Q-channel Histogram")
    axs[1].set_xlabel("Q Value")
    axs[1].set_ylabel("Counts")
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    return fit_params


def calculate_SNR(fit_params):
    """
    Calculate the Signal-to-Noise Ratio (SNR) using the provided fit parameters.

    The SNR is defined by the equation:
    
        SNR = sqrt((μ_I_g - μ_I_e)² + (μ_Q_g - μ_Q_e)²) / ((|σ_I_g| + |σ_I_e| + |σ_Q_g| + |σ_Q_e|) / 4)
      
    The numerator represents the magnitude of the difference vector between the ground
    and excited state means in the IQ-plane. The denominator is the average of the absolute
    standard deviations of all components, serving as an estimate of the noise.

    Parameters:
        fit_params (dict): Dictionary containing measurement fit parameters. Expected to have keys:
            "I_g", "I_e", "Q_g", "Q_e". Each of these is a dictionary with:
                - "mu": the mean value.
                - "std": the standard deviation value.
    
    Returns:
        float: The calculated signal-to-noise ratio (SNR).
    """
    mu_I_g = fit_params["I_g"]["mu"]
    mu_I_e = fit_params["I_e"]["mu"]
    mu_Q_g = fit_params["Q_g"]["mu"]
    mu_Q_e = fit_params["Q_e"]["mu"]
    std_I_g = fit_params["I_g"]["std"]
    std_I_e = fit_params["I_e"]["std"]
    std_Q_g = fit_params["Q_g"]["std"]
    std_Q_e = fit_params["Q_e"]["std"]

    signal = np.sqrt((mu_I_g - mu_I_e) ** 2 + (mu_Q_g - mu_Q_e) ** 2)
    noise = (np.abs(std_I_g) + np.abs(std_I_e) + np.abs(std_Q_g) + np.abs(std_Q_e)) / 4
    SNR = signal / noise
    return SNR
