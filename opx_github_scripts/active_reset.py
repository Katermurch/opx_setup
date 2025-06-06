"""
        ACTIVE RESET
This script is used to benchmark different types of q2_xy initialization including active reset protocols.
the different methods are written in macros for better readability.

Each protocol is detailed in the corresponding docstring, but the idea behind active reset is to first measure one
quadrature of the rr2 ("I") and compare it to one or two threshold in order to decide whether to apply a pi-pulse
(q2_xy in |e>), do nothing (q2_xy in |g>) or measure again if the q2_xy state is undetermined (active_reset_two_thresholds).

Then, after q2_xy initialization, the IQ blobs for |g> and |e> are measured again and the readout fidelity is derived
similarly to what is done in IQ_blobs.py.

Prerequisites:
    - Having found the resonance frequency of the rr2 coupled to the q2_xy under study (rr2_spectroscopy).
    - Having calibrated q2_xy pi pulse (x180) by running q2_xy, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - Having calibrated the IQ blobs (rotation_angle and ge_threshold).
    - (optional) Having calibrated the readout (readout_frequency_, _amplitude_, _duration_optimization).
    - Having updated the rotation angle (rotation_angle) and g -> e threshold (ge_threshold) in the configuration (IQ_blobs.py).
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig, LoopbackInterface
from configuration import *
from qualang_tools.analysis.discriminator import two_state_discriminator
import matplotlib.pyplot as plt
from qualang_tools.results.data_handler import DataHandler

##################
#   Parameters   #
##################
# Parameters Definition
initialization_method = "active_reset_one_threshold"  # "thermalization", "active_reset_one_threshold", "active_reset_two_thresholds", "active_reset_fast"
n_shot = 10000  # Number of acquired shots
# The thresholds ar calibrated with the IQ_blobs.py script:
# If I > threshold_e, then the q2_xy is assumed to be in |e> and a pi pulse is played to reset it.
# If I < threshold_g, then the q2_xy is assumed to be in |g>.
# else, the q2_xy state is not determined accurately enough, so we just measure again.
ge_threshold_g = ge_threshold_q2 * 0.5
ge_threshold_e = ge_threshold_q2
# Maximum number of tries for active reset
max_tries = 10

# Data to save
save_data_dict = {
    "n_shot": n_shot,
    "config": config,
}


###################################
# Helper functions and QUA macros #
###################################
def q2_xy_initialization(method: str = "thermalization"):
    """
    Allows to switch between several initialization methods.

    :param method: the desired initialization method among "thermalization", "active_reset_one_threshold", "active_reset_two_thresholds", "active_reset_fast".
    :return: the number of tries to reset the q2_xy.
    """
    if method == "thermalization":
        wait(thermalization_time * u.ns)
        return 1
    elif method == "active_reset_fast":
        return active_reset_fast(ge_threshold_e)
    elif method == "active_reset_one_threshold":
        return active_reset_one_threshold(ge_threshold_e, max_tries)
    elif method == "active_reset_two_thresholds":
        return active_reset_two_thresholds(ge_threshold_g, ge_threshold_e, max_tries)
    else:
        raise ValueError(f"method {method} is not implemented.")


def active_reset_one_threshold(threshold_g: float, max_tries: int):
    """
    Active reset protocol where the outcome of the measurement is compared to a pre-calibrated threshold (IQ_blobs.py).
    If the q2_xy is in |e> (I>threshold), then play a pi pulse and measure again, else (q2_xy in |g>) return the number
    of pi-pulses needed to reset the q2_xy.
    The program waits for the rr2 to deplete before playing the conditional pi-pulse so that the calibrated
    pi-pulse parameters are still valid.

    :param threshold_g: threshold between the |g> and |e> blobs - calibrated in IQ_blobs.py
    :param max_tries: maximum number of iterations needed to reset the q2_xy before exiting the loop anyway.
    :return: the number of tries to reset the q2_xy.
    """
    I_reset = declare(fixed, value = 1.396e-03 + 0.01)
    counter = declare(int)
    assign(counter, 0)
    align("rr2", "q2_xy")
    with while_((I_reset > threshold_g) & (counter < max_tries)):
        # Measure the state of the rr2
        measure("readout", "rr2", dual_demod.full("rotated_cos", "rotated_sin", I_reset))
        align("rr2", "q2_xy")
        # Wait for the rr2 to deplete
        wait(depletion_time * u.ns, "q2_xy")
        # Play a conditional pi-pulse to actively reset the q2_xy
        play("x180", "q2_xy", condition=(I_reset > threshold_g))
        # Update the counter for benchmarking purposes
        assign(counter, counter + 1)
    return counter


def active_reset_two_thresholds(threshold_g: float, threshold_e: float, max_tries: int):
    """
    Active reset protocol where the outcome of the measurement is compared to two pre-calibrated thresholds (IQ_blobs.py).
    If I > threshold_e, then the q2_xy is assumed to be in |e> and a pi pulse is played to reset it.
    If I < threshold_g, then the q2_xy is assumed to be in |g> and the loop can be exited.
    else, the q2_xy state is not determined accurately enough, so we just repeat the process.
    The program waits for the rr2 to deplete before playing the conditional pi-pulse so that the calibrated
    pi-pulse parameters are still valid.

    :param threshold_g: threshold "inside" the |g> blob, below which the q2_xy is in |g> with great certainty.
    :param threshold_e: threshold between the |g> and |e> blobs - calibrated in IQ_blobs.py
    :param max_tries: maximum number of iterations needed to reset the q2_xy before exiting the loop anyway.
    :return: the number of tries to reset the q2_xy.
    """
    I_reset = declare(fixed, value = 1.396e-03 + 0.01)
    counter = declare(int)
    assign(counter, 0)
    align("rr2", "q2_xy")
    with while_((I_reset > threshold_g) & (counter < max_tries)):
        # Measure the state of the rr2
        measure("readout", "rr2",  dual_demod.full("rotated_cos", "rotated_sin", I_reset))
        align("rr2", "q2_xy")
        # Wait for the rr2 to deplete
        wait(depletion_time * u.ns, "q2_xy")
        # Play a conditional pi-pulse to actively reset the q2_xy
        play("x180", "q2_xy", condition=(I_reset > threshold_e))
        # Update the counter for benchmarking purposes
        assign(counter, counter + 1)
    return counter


def active_reset_fast(threshold_g: float):
    """
    Active reset protocol where the outcome of the measurement is compared to a pre-calibrated threshold (IQ_blobs.py).
    If the q2_xy is in |e> (I>threshold), then play a pi pulse, else (q2_xy in |g>) do nothing and proceed to the sequence.
    The program waits for the rr2 to deplete before playing the conditional pi-pulse so that the calibrated
    pi-pulse parameters are still valid.

    :param threshold_g: threshold between the |g> and |e> blobs - calibrated in IQ_blobs.py.
    :return: 1
    """
    I_reset = declare(fixed)
    align("rr2", "q2_xy")
    # Measure the state of the rr2
    measure("readout", "rr2", None, dual_demod.full("rotated_cos", "rotated_sin", I_reset))
    align("rr2", "q2_xy")
    # Wait for the rr2 to deplete
    wait(depletion_time * u.ns, "q2_xy")
    # Play a conditional pi-pulse to actively reset the q2_xy
    play("x180", "q2_xy", condition=(I_reset > threshold_g))
    return 1


###################
# The QUA program #
###################
with program() as active_reset_prog:
    n = declare(int)  # Averaging index
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()
    I_g = declare(fixed)
    Q_g = declare(fixed)
    I_g_st = declare_stream()
    Q_g_st = declare_stream()
    I_e = declare(fixed)
    Q_e = declare(fixed)
    I_e_st = declare_stream()
    Q_e_st = declare_stream()

    cont_condition = declare(bool)
    tries_st = declare_stream()

    with for_(n, 0, n < n_shot, n + 1):
        # Active reset
        count = q2_xy_initialization(method=initialization_method)
        align()
        # Measure the state of the rr2 after reset, q2_xy should be in |g>
        measure(
            "readout",
            "rr2",
            None,
            dual_demod.full("rotated_cos", "rotated_sin", I_g),
            dual_demod.full("rotated_minus_sin", "rotated_cos", Q_g),
        )
        # Save the 'I' & 'Q' quadratures to their respective streams for the ground state
        save(I_g, I_g_st)
        save(Q_g, Q_g_st)
        with if_(count > 0):
            save(count, tries_st)

        align()  # global align
        # wait(thermalization_time*2)
        # Active reset
        count = q2_xy_initialization(method=initialization_method)
        align()
        # Play the x180 gate to put the q2_xy in the excited state
        play("x180", "q2_xy")
        # Align the two elements to measure after playing the q2_xy pulse.
        align("q2_xy", "rr2")
        # Measure the state of the rr2, q2_xy should be in |e>
        measure(
            "readout",
            "rr2",
            None,
            dual_demod.full("rotated_cos", "rotated_sin", I_e),
            dual_demod.full("rotated_minus_sin", "rotated_cos", Q_e),
        )
        # Save the 'I' & 'Q' quadratures to their respective streams for the excited state
        save(I_e, I_e_st)
        save(Q_e, Q_e_st)
        # Save only the count when the q2_xy was not directly measured in |g>
        with if_(count > 0):
            save(count, tries_st)

    with stream_processing():
        # Save all streamed points for plotting the IQ blobs
        I_g_st.save_all("I_g")
        Q_g_st.save_all("Q_g")
        I_e_st.save_all("I_e")
        Q_e_st.save_all("Q_e")
        tries_st.average().save("average_tries")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

simulation = False
if simulation:
    simulation_config = SimulationConfig(
        duration=28000, simulation_interface=LoopbackInterface([("con1", 3, "con1", 1)])
    )
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, active_reset_prog, simulation_config)
    # Get the simulated samples
    samples = job.get_simulated_samples()
    # Plot the simulated samples
    samples.con1.plot()
    # Get the waveform report object
    waveform_report = job.get_simulated_waveform_report()
    # Cast the waveform report to a python dictionary
    waveform_dict = waveform_report.to_dict()
    # Visualize and save the waveform report
    waveform_report.create_plot(samples, plot=True, save_path=str(Path(__file__).resolve()))

else:
    qm = qmm.open_qm(config)
    job = qm.execute(active_reset_prog)
    # Creates a result handle to fetch data from the OPX
    res_handles = job.result_handles
    # Waits (blocks the Python console) until all results have been acquired
    res_handles.wait_for_all_values()
    # Fetch the 'I' & 'Q' points for the q2_xy in the ground and excited states
    Ig = res_handles.get("I_g").fetch_all()["value"]
    Qg = res_handles.get("Q_g").fetch_all()["value"]
    Ie = res_handles.get("I_e").fetch_all()["value"]
    Qe = res_handles.get("Q_e").fetch_all()["value"]
    average_tries = res_handles.get("average_tries").fetch_all()
    # Plot the IQ blobs, rotate them to get the separation along the 'I' quadrature, estimate a threshold between them
    # for state discrimination and derive the fidelity matrix
    angle, threshold, fidelity, gg, ge, eg, ee = two_state_discriminator(Ig, Qg, Ie, Qe, b_print=True, b_plot=True)
    plt.suptitle(f"{average_tries=}")
    print(f"{average_tries=}")
    plt.show()
    # Save results
    script_name = Path(__file__).name
    data_handler = DataHandler(root_data_folder=save_dir)
    save_data_dict.update({"Ig_data": Ig})
    save_data_dict.update({"Qg_data": Qg})
    save_data_dict.update({"Ie_data": Ie})
    save_data_dict.update({"Qe_data": Qe})
    save_data_dict.update({"two_state_discriminator": [angle, threshold, fidelity, gg, ge, eg, ee]})
    data_handler.additional_files = {script_name: script_name, **default_additional_files}
    data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])