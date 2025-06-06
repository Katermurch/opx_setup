from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.analysis import two_state_discriminator
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt
from analysis import fit_line
from qualang_tools.results.data_handler import DataHandler

##################
#   Parameters   #
##################
# Parameters Definition
n_runs = 2000
# The readout amplitude sweep (as a pre-factor of the readout amplitude) - must be within [-2; 2)
a_min = 0
a_max = 0.2
da = 0.01
amplitudes = np.arange(a_min, a_max + da / 2, da)  # The amplitude vector +da/2 to add a_max to the scan
print(amplitudes)
# Data to save
save_data_dict = {
    "n_runs": n_runs,
    "amplitudes": amplitudes,
    "config": config,
}

###################
# The QUA program #
###################
with program() as ro_amp_opt:
    n = declare(int)  # QUA variable for the number of runs
    counter = declare(int, value=0)  # Counter for the progress bar
    a = declare(fixed)  # QUA variable for the readout amplitude
    I_g = declare(fixed)  # QUA variable for the 'I' quadrature when the qubit is in |g>
    Q_g = declare(fixed)  # QUA variable for the 'Q' quadrature when the qubit is in |g>
    I_g_st = declare_stream()
    Q_g_st = declare_stream()
    I_e = declare(fixed)  # QUA variable for the 'I' quadrature when the qubit is in |e>
    Q_e = declare(fixed)  # QUA variable for the 'Q' quadrature when the qubit is in |e>
    I_e_st = declare_stream()
    Q_e_st = declare_stream()
    n_st = declare_stream()

    with for_(n, 0, n < n_runs, n + 1):
        save(n, n_st)
        with for_(*from_array(a, amplitudes)):
            measure(
                "readout" * amp(a),
                "rr2",
                dual_demod.full("rotated_cos", "rotated_sin", I_g),
                dual_demod.full("rotated_minus_sin", "rotated_cos", Q_g),
            )
            # Wait for the qubit to decay to the ground state
            wait(thermalization_time * u.ns, "rr2")
            # Save the 'I_e' & 'Q_e' quadratures to their respective streams
            save(I_g, I_g_st)
            save(Q_g, Q_g_st)

            align()  # global align
            # Play the x180 gate to put the qubit in the excited state
            play("x180", "q2_xy")
            # Align the two elements to measure after playing the qubit pulse.
            align("q2_xy", "rr2")
            # Measure the state of the resonator
            measure(
                "readout" * amp(a),
                "rr2",
                dual_demod.full("rotated_cos", "rotated_sin", I_e),
                dual_demod.full("rotated_minus_sin", "rotated_cos", Q_e),
            )
            # Wait for the qubit to decay to the ground state
            wait(thermalization_time * u.ns, "rr2")
            # Save the 'I_e' & 'Q_e' quadratures to their respective streams
            save(I_e, I_e_st)
            save(Q_e, Q_e_st)


    with stream_processing():
        n_st.save("iteration")
            # mean values
        I_g_st.buffer(len(amplitudes)).average().save("Ig_avg")
        Q_g_st.buffer(len(amplitudes)).average().save("Qg_avg")
        I_e_st.buffer(len(amplitudes)).average().save("Ie_avg")
        Q_e_st.buffer(len(amplitudes)).average().save("Qe_avg")
        # variances to get the SNR
        (
            ((I_g_st.buffer(len(amplitudes)) * I_g_st.buffer(len(amplitudes))).average())
            - (I_g_st.buffer(len(amplitudes)).average() * I_g_st.buffer(len(amplitudes)).average())
        ).save("Ig_var")
        (
            ((Q_g_st.buffer(len(amplitudes)) * Q_g_st.buffer(len(amplitudes))).average())
            - (Q_g_st.buffer(len(amplitudes)).average() * Q_g_st.buffer(len(amplitudes)).average())
        ).save("Qg_var")
        (
            ((I_e_st.buffer(len(amplitudes)) * I_e_st.buffer(len(amplitudes))).average())
            - (I_e_st.buffer(len(amplitudes)).average() * I_e_st.buffer(len(amplitudes)).average())
        ).save("Ie_var")
        (
            ((Q_e_st.buffer(len(amplitudes)) * Q_e_st.buffer(len(amplitudes))).average())
            - (Q_e_st.buffer(len(amplitudes)).average() * Q_e_st.buffer(len(amplitudes)).average())
        ).save("Qe_var")
                

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
    job = qmm.simulate(config, ro_amp_opt, simulation_config)
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
    job = qm.execute(ro_amp_opt)  # execute QUA program
    # Get results from QUA program
    results = fetching_tool(job, data_list=["iteration"], mode="live")
    # Get progress counter to monitor runtime of the program
    while results.is_processing():
        # Fetch results
        iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration[0], n_runs, start_time=results.get_start_time())

    # Fetch the results at the end
    results = fetching_tool(
        job,
        data_list=["Ig_avg", "Qg_avg", "Ie_avg", "Qe_avg", "Ig_var", "Qg_var", "Ie_var", "Qe_var", "iteration"],
    )
    Ig_avg, Qg_avg, Ie_avg, Qe_avg, Ig_var, Qg_var, Ie_var, Qe_var, iteration = results.fetch_all()
    
    Z = (Ie_avg - Ig_avg) + 1j * (Qe_avg - Qg_avg)
    var = (Ig_var + Qg_var + Ie_var + Qe_var) / 4
    SNR = np.sqrt(((np.abs(Z)) ** 2) / (2 * var))
 
  

    (popt, perr, y_vals_fit, pcov)=fit_line(amplitudes * readout_amp_q2,SNR)  
    
    # Plot the data
    fig = plt.figure()
    plt.scatter(amplitudes * readout_amp_q2, SNR, label="SNR")
    plt.plot(amplitudes * readout_amp_q2, y_vals_fit, "b", label='Fit line')
    plt.title("SNR vs Readout Amplitude")
    plt.xlabel("Readout amplitude [V]")
    plt.ylabel("SNR")
    plt.legend()
    plt.show()
    

    # # Save results
    # script_name = Path(__file__).name
    # data_handler = DataHandler(root_data_folder=save_dir)
    # save_data_dict.update({"Ig_data": I_g})
    # save_data_dict.update({"Qg_data": Q_g})
    # save_data_dict.update({"Ie_data": I_e})
    # save_data_dict.update({"Qe_data": Q_e})
    # save_data_dict.update({"fig_live": fig})
    # data_handler.additional_files = {script_name: script_name, **default_additional_files}
    # data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])