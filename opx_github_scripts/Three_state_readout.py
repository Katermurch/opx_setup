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
    I1_g = declare(fixed)
    Q1_g = declare(fixed)
    I2_g = declare(fixed)
    Q2_g = declare(fixed)
    I1_g_st = declare_stream()
    Q1_g_st = declare_stream()
    I2_g_st = declare_stream()
    Q2_g_st = declare_stream()
    I1_e = declare(fixed)
    Q1_e = declare(fixed)
    I2_e = declare(fixed)
    Q2_e = declare(fixed)
    I1_e_st = declare_stream()
    Q1_e_st = declare_stream()
    I2_e_st = declare_stream()
    Q2_e_st = declare_stream()
    I1_f = declare(fixed)
    Q1_f = declare(fixed)
    I2_f = declare(fixed)
    Q2_f = declare(fixed)
    I1_f_st = declare_stream()
    Q1_f_st = declare_stream()
    I2_f_st = declare_stream()
    Q2_f_st = declare_stream()

    with for_(n, 0, n < n_runs, n + 1):
        # Measure the state of the rr2
        play('const', 'coupler_1')
        align('coupler_1', 'rr1', 'rr2')
        measure(
            "readout",
            "rr1",
            None,
            dual_demod.full("rotated_cos_q1", "rotated_sin_q1", I1_g),
            dual_demod.full("rotated_minus_sin_q1", "rotated_cos_q1", Q1_g),
            # dual_demod.full("opt_cos", "opt_sin", I_g),
            # dual_demod.full("opt_minus_sin", "opt_cos", Q_g),

        )
        measure(
        "readout",
        "rr2",
        None,
        dual_demod.full("rotated_cos_q2", "rotated_sin_q2", I2_g),
        dual_demod.full("rotated_minus_sin_q2", "rotated_cos_q2", Q2_g),
        # dual_demod.full("opt_cos", "opt_sin", I_g),
        # dual_demod.full("opt_minus_sin", "opt_cos", Q_g),

        )
        # Wait for the q2_xy to decay to the ground state in the case of measurement induced transitions
        wait(thermalization_time * u.ns, "rr1")
        wait(thermalization_time * u.ns, "rr2")
        # Save the 'I' & 'Q' quadratures to their respective streams for the ground state
        save(I1_g, I1_g_st)
        save(Q1_g, Q1_g_st)
        save(I2_g, I2_g_st)
        save(Q2_g, Q2_g_st)

        align()  # global align

        # reset_if_phase("rr2")
        # wait(400)
        # Play the x180 gate to put the q2_xy in the excited state
        play("x180", "q1_xy")
        # Align the two elements to measure after playing the q2_xy pulse.
        align('coupler_1',"q1_xy")
        play('const', 'coupler_1')
        align()
        # Measure the state of the rr2
        measure(
            "readout",
            "rr1",
            None,
            dual_demod.full("rotated_cos_q1", "rotated_sin_q1", I1_e),
            dual_demod.full("rotated_minus_sin_q1", "rotated_cos_q1", Q1_e),
            # dual_demod.full("opt_cos", "opt_sin", I_e),
            # dual_demod.full("opt_minus_sin", "opt_cos", Q_e),
        )
        measure(
        "readout",
        "rr2",
        None,
        dual_demod.full("rotated_cos_q2", "rotated_sin_q2", I2_e),
        dual_demod.full("rotated_minus_sin_q2", "rotated_cos_q2", Q2_e),
        # dual_demod.full("opt_cos", "opt_sin", I_g),
        # dual_demod.full("opt_minus_sin", "opt_cos", Q_g),

        )
        # Wait for the q2_xy to decay to the ground state
        wait(thermalization_time * u.ns, "rr1")
        wait(thermalization_time * u.ns, "rr2")
        # Save the 'I' & 'Q' quadratures to their respective streams for the excited state
        save(I1_e, I1_e_st)
        save(Q1_e, Q1_e_st)
        save(I2_e, I2_e_st)
        save(Q2_e, Q2_e_st)

        align()

        # play('x180ef', 'q1_xy')
        # Play the x180 gate to put the q2_xy in the excited state
        play("x180", "q1_xy")
        align('q1_xy', 'q1_xy_ef')
        play('x180_ef', 'q1_xy_ef')
        # Align the two elements to measure after playing the q2_xy pulse.
        align("q1_xy_ef", "coupler_1")
        play('const', 'coupler_1')
        align()
        # Measure the state of the rr2
        measure(
            "readout",
            "rr1",
            None,
            dual_demod.full("rotated_cos_q1", "rotated_sin_q1", I1_f),
            dual_demod.full("rotated_minus_sin_q1", "rotated_cos_q1", Q1_f),
            # dual_demod.full("opt_cos", "opt_sin", I_e),
            # dual_demod.full("opt_minus_sin", "opt_cos", Q_e),
        )
        measure(
        "readout",
        "rr2",
        None,
        dual_demod.full("rotated_cos_q2", "rotated_sin_q2", I2_f),
        dual_demod.full("rotated_minus_sin_q2", "rotated_cos_q2", Q2_f),
        # dual_demod.full("opt_cos", "opt_sin", I_g),
        # dual_demod.full("opt_minus_sin", "opt_cos", Q_g),

        )
        # Wait for the q2_xy to decay to the ground state
        wait(thermalization_time * u.ns, "rr1")
        wait(thermalization_time * u.ns, "rr2")
        # Save the 'I' & 'Q' quadratures to their respective streams for the excited state
        save(I1_f, I1_f_st)
        save(Q1_f, Q1_f_st)
        save(I2_f, I2_f_st)
        save(Q2_f, Q2_f_st)
        align()

    with stream_processing():
        # Save all streamed points for plotting the IQ blobs
        I1_g_st.save_all("I1_g")
        Q1_g_st.save_all("Q1_g")
        I1_e_st.save_all("I1_e")
        Q1_e_st.save_all("Q1_e")
        I1_f_st.save_all("I1_f")
        Q1_f_st.save_all("Q1_f")
        I2_g_st.save_all("I2_g")
        Q2_g_st.save_all("Q2_g")
        I2_e_st.save_all("I2_e")
        Q2_e_st.save_all("Q2_e")
        I2_f_st.save_all("I2_f")
        Q2_f_st.save_all("Q2_f")


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
    simulation_config = SimulationConfig(duration=50_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, IQ_blobs, simulation_config)
    # Get the simulated samples
    samples = job.get_simulated_samples()
    # Plot the simulated samples
    samples.con1.plot()
    plt.show()
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
    I1g = res_handles.get("I1_g").fetch_all()["value"]
    Q1g = res_handles.get("Q1_g").fetch_all()["value"]
    I1e = res_handles.get("I1_e").fetch_all()["value"]
    Q1e = res_handles.get("Q1_e").fetch_all()["value"]
    I1f = res_handles.get("I1_f").fetch_all()["value"]
    Q1f = res_handles.get("Q1_f").fetch_all()["value"]
    I2g = res_handles.get("I2_g").fetch_all()["value"]
    Q2g = res_handles.get("Q2_g").fetch_all()["value"]
    I2e = res_handles.get("I2_e").fetch_all()["value"]
    Q2e = res_handles.get("Q2_e").fetch_all()["value"]
    I2f = res_handles.get("I2_f").fetch_all()["value"]
    Q2f = res_handles.get("Q2_f").fetch_all()["value"]
    # Plot the IQ blobs, rotate them to get the separation along the 'I' quadrature, estimate a threshold between them
    # for state discrimination and derive the fidelity matrix
    plt.subplot(1,2,1)
    plt.plot(I1g, Q1g, ".", alpha=0.5, label="G", markersize=2)
    plt.plot(I1e, Q1e, ".", alpha=0.5, label="E", markersize=2)
    plt.plot(I1f, Q1f, ".", alpha=0.5, label="F", markersize=2)
    
    plt.legend(["G", "E","F"])
    plt.xlabel("I1")
    plt.ylabel("Q1")
    plt.title("Qubit1")
    plt.subplot(1,2,2)
    plt.plot(I2g, Q2g, ".", alpha=0.5, label="G", markersize=2)
    plt.plot(I2e, Q2e, ".", alpha=0.5, label="E", markersize=2)
    plt.plot(I2f, Q2f, ".", alpha=0.5, label="F", markersize=2)
    plt.legend(["G", "E","F"])
    plt.xlabel("I2")
    plt.ylabel("Q2")
    plt.title("Qubit2")
    # angle, threshold, fidelity, gg, ge, eg, ee = two_state_discriminator(I1g, Q1g, Ie, Qe, b_print=True, b_plot=True)
    # angle2, threshold2, fidelity2, gg2, ge2, eg2, ee2 = two_state_discriminator(Ig, Qg, If, Qf, b_print=True, b_plot=True)
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
    save_data_dict.update({"I1g_data": I1g})
    save_data_dict.update({"Q1g_data": Q1g})
    save_data_dict.update({"I1e_data": I1e})
    save_data_dict.update({"Q1e_data": Q1e})
    save_data_dict.update({"I1f_data": I1f})
    save_data_dict.update({"Q1f_data": Q1f})
    save_data_dict.update({"I2g_data": I2g})
    save_data_dict.update({"Q2g_data": Q2g})
    save_data_dict.update({"I2e_data": I2e})
    save_data_dict.update({"Q2e_data": Q2e})
    save_data_dict.update({"I2f_data": I2f})
    save_data_dict.update({"Q2f_data": Q2f})
    # save_data_dict.update({"two_state_discriminator": [angle, threshold, fidelity, gg, ge, eg, ee]})
    data_handler.additional_files = {script_name: script_name, **default_additional_files}
    data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])