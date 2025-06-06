"""
        IQ BLOBS
This sequence involves measuring the state of the rr2 'N' times, first after thermalization (with the q2_xy
in the |g> state) and then after applying a pi pulse to the q2_xy (bringing the q2_xy to the |e> state) successively.
The resulting IQ blobs are displayed, and the data is processed to determine:
    - The rotation angle required for the integration weights, ensuring that the separation between |g> and |e> states
      aligns with the 'I' quadrature.
    - The threshold along the 'I' quadrature for effective q2_xy state discrimination.
    - The readout fidelity matrix, which is also influenced by the pi pulse fidelity.

Prerequisites:
    - Having found the resonance frequency of the rr2 coupled to the q2_xy under study (rr2_spectroscopy).
    - Having calibrated q2_xy pi pulse (x180) by running q2_xy, spectroscopy, rabi_chevron, power_rabi and updated the config.

Next steps before going to the next node:
    - Update the rotation angle (rotation_angle) in the configuration.
    - Update the g -> e threshold (ge_threshold) in the configuration.
"""

from qm.qua import *
from qm import SimulationConfig
from qm import QuantumMachinesManager
from configuration import *
from qualang_tools.analysis.discriminator import two_state_discriminator
from qualang_tools.results.data_handler import DataHandler

##################
#   Parameters   #
##################
# Parameters Definition
n_runs = 10000  # Number of runs

# Data to save
save_data_dict = {
    "n_runs": n_runs,
    "config": config,
}

###################
# The QUA program #
###################

n_runs = 10000  # Number of runs

# Data to save
save_data_dict = {
    "n_runs": n_runs,
    "config": config,
}
with program() as IQ_blobs:
    n = declare(int)
    I_g = declare(fixed)
    Q_g = declare(fixed)
    I_g_st = declare_stream()
    Q_g_st = declare_stream()
    I_e = declare(fixed)
    Q_e = declare(fixed)
    I_e_st = declare_stream()
    Q_e_st = declare_stream()
    I_f = declare(fixed)
    Q_f = declare(fixed)
    I_f_st = declare_stream()
    Q_f_st = declare_stream()

    with for_(n, 0, n < n_runs, n + 1):
        # Measure the state of the rr2
        
        measure(
            "readout",
            "rr1",
            None,
            dual_demod.full("rotated_cos", "rotated_sin", I_g),
            dual_demod.full("rotated_minus_sin", "rotated_cos", Q_g),
            # dual_demod.full("opt_cos", "opt_sin", I_g),
            # dual_demod.full("opt_minus_sin", "opt_cos", Q_g),

        )
        # Wait for the q2_xy to decay to the ground state in the case of measurement induced transitions
        wait(thermalization_time * u.ns, "rr1")
        # Save the 'I' & 'Q' quadratures to their respective streams for the ground state
        save(I_g, I_g_st)
        save(Q_g, Q_g_st)

        align()  # global align

        # reset_if_phase("rr2")
        # wait(400)
        # Play the x180 gate to put the q2_xy in the excited state
        play("x180", "q1_xy")
        # Align the two elements to measure after playing the q2_xy pulse.
        align("q1_xy", "rr1")
        
        # Measure the state of the rr2
        measure(
            "readout",
            "rr1",
            None,
            dual_demod.full("rotated_cos", "rotated_sin", I_e),
            dual_demod.full("rotated_minus_sin", "rotated_cos", Q_e),
            # dual_demod.full("opt_cos", "opt_sin", I_e),
            # dual_demod.full("opt_minus_sin", "opt_cos", Q_e),
        )
        # Wait for the q2_xy to decay to the ground state
        wait(thermalization_time * u.ns, "rr1")
        # Save the 'I' & 'Q' quadratures to their respective streams for the excited state
        save(I_e, I_e_st)
        save(Q_e, Q_e_st)

        align()

        # play('x180ef', 'q1_xy')
        # Play the x180 gate to put the q2_xy in the excited state
        play("x180", "q1_xy")
        align('q1_xy', 'q1_xy_ef')
        play('x180_ef', 'q1_xy_ef')
        # Align the two elements to measure after playing the q2_xy pulse.
        align("q1_xy_ef", "rr1")
        
        # Measure the state of the rr2
        measure(
            "readout",
            "rr1",
            None,
            dual_demod.full("rotated_cos", "rotated_sin", I_f),
            dual_demod.full("rotated_minus_sin", "rotated_cos", Q_f),
            # dual_demod.full("opt_cos", "opt_sin", I_e),
            # dual_demod.full("opt_minus_sin", "opt_cos", Q_e),
        )
        # Wait for the q2_xy to decay to the ground state
        wait(thermalization_time * u.ns, "rr1")
        # Save the 'I' & 'Q' quadratures to their respective streams for the excited state
        save(I_f, I_f_st)
        save(Q_f, Q_f_st)

        align()

    with stream_processing():
        # Save all streamed points for plotting the IQ blobs
        I_g_st.save_all("I_g")
        Q_g_st.save_all("Q_g")
        I_e_st.save_all("I_e")
        Q_e_st.save_all("Q_e")
        I_f_st.save_all("I_f")
        Q_f_st.save_all("Q_f")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

###########################
# Run or Simulate Program #
###########################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, IQ_blobs, simulation_config)
    # Get the simulated samples
    samples = job.get_simulated_samples()
    # Plot the simulated samples
    samples.con1.plot()
    plt.show()
    # Get the waveform report object
    waveform_report = job.get_simulated_waveform_report()
    # Cast the waveform report to a python dictionary
    waveform_dict = waveform_report.to_dict()
    # Visualize and save the waveform report
    waveform_report.create_plot(samples, plot=True, save_path=str(Path(__file__).resolve()))
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(IQ_blobs)
    # Creates a result handle to fetch data from the OPX
    res_handles = job.result_handles
    # Waits (blocks the Python console) until all results have been acquired
    res_handles.wait_for_all_values()
    # Fetch the 'I' & 'Q' points for the q2_xy in the ground and excited states
    Ig = res_handles.get("I_g").fetch_all()["value"]
    Qg = res_handles.get("Q_g").fetch_all()["value"]
    Ie = res_handles.get("I_e").fetch_all()["value"]
    Qe = res_handles.get("Q_e").fetch_all()["value"]
    If = res_handles.get("I_f").fetch_all()["value"]
    Qf = res_handles.get("Q_f").fetch_all()["value"]
    # Plot the IQ blobs, rotate them to get the separation along the 'I' quadrature, estimate a threshold between them
    # for state discrimination and derive the fidelity matrix
    angle, threshold, fidelity, gg, ge, eg, ee = two_state_discriminator(Ig, Qg, Ie, Qe, b_print=True, b_plot=True)
    angle2, threshold2, fidelity2, gg2, ge2, eg2, ee2 = two_state_discriminator(Ig, Qg, If, Qf, b_print=True, b_plot=True)
    plt.show()
    #########################################
    # The two_state_discriminator gives us the rotation angle which makes it such that all of the information will be in
    # the I axis. This is being done by setting the `rotation_angle` parameter in the configuration.
    # See this for more information: https://qm-docs.qualang.io/guides/demod#rotating-the-iq-plane
    # Once we do this, we can perform active reset using:
    #########################################
    #
    # # Active reset:
    # with if_(I > threshold):
    #     play("x180", "q2_xy")
    #
    #########################################
    #
    # # Active reset (faster):
    # play("x180", "q2_xy", condition=I > threshold)
    #
    #########################################
    #
    # # Repeat until success active reset
    # with while_(I > threshold):
    #     play("x180", "q2_xy")
    #     align("q2_xy", "rr2")
    #     measure("readout", "rr2", None,
    #                 dual_demod.full("rotated_cos", "rotated_sin", I))
    #
    #########################################
    #
    # # Repeat until success active reset, up to 3 iterations
    # count = declare(int)
    # assign(count, 0)
    # cont_condition = declare(bool)
    # assign(cont_condition, ((I > threshold) & (count < 3)))
    # with while_(cont_condition):
    #     play("x180", "q2_xy")
    #     align("q2_xy", "rr2")
    #     measure("readout", "rr2", None,
    #                 dual_demod.full("rotated_cos", "rotated_sin", I))
    #     assign(count, count + 1)
    #     assign(cont_condition, ((I > threshold) & (count < 3)))
    #
    #########################################
    # Save results
    script_name = Path(__file__).name
    data_handler = DataHandler(root_data_folder=save_dir)
    save_data_dict.update({"Ig_data": Ig})
    save_data_dict.update({"Qg_data": Qg})
    save_data_dict.update({"Ie_data": Ie})
    save_data_dict.update({"Qe_data": Qe})
    save_data_dict.update({"If_data": If})
    save_data_dict.update({"Qf_data": Qf})
    save_data_dict.update({"two_state_discriminator": [angle, threshold, fidelity, gg, ge, eg, ee]})
    data_handler.additional_files = {script_name: script_name, **default_additional_files}
    data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])