"""
        RAMSEY CHEVRON (IDLE TIME VS FREQUENCY)
The program consists in playing a Ramsey sequence (x90 - idle_time - x90 - measurement) for different q2_xy intermediate
frequencies and idle times.
From the results, one can estimate the q2_xy frequency more precisely than by doing Rabi and also gets a rough estimate
of the q2_xy coherence time.

Prerequisites:
    - Having found the resonance frequency of the rr2 coupled to the q2_xy under study (rr2_spectroscopy).
    - Having calibrated q2_xy pi pulse (x180) by running q2_xy, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.

Next steps before going to the next node:
    - Update the q2_xy frequency (q2_xy_IF) in the configuration.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt
from qualang_tools.results.data_handler import DataHandler

##################
#   Parameters   #
##################
# Parameters Definition
n_avg = 20  # Number of averaging loops

# Frequency sweep in Hz
freq_span = 5 * u.MHz
df = 100 * u.kHz
dfs = np.arange(-freq_span, freq_span, df)

# Idle time sweep (Needs to be a list of integers) - in clock cycles (4ns)
tau_max = 2000 // 4
d_tau = 40 // 4
taus = np.arange(0, tau_max, d_tau)
if len(np.where((taus > 0) & (taus < 4))[0]) > 0:
    raise Exception("Delay must be either 0 or an integer larger than 4.")

# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "IF_frequencies": dfs,
    "taus": taus,
    "config": config,
}

###################
# The QUA program #
###################
with program() as ramsey_freq_duration:
    n = declare(int)  # QUA variable for the averaging loop
    df = declare(int)  # QUA variable for the q2_xy detuning
    delay = declare(int)  # QUA variable for the idle time
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    n_st = declare_stream()  # Stream for the averaging iteration 'n'

    with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
        with for_(*from_array(delay, taus)):  # QUA for_ loop for sweeping the idle time
            with for_(*from_array(df, dfs)):  # QUA for_ loop for sweeping the q2_xy frequency
                # Update the frequency of the digital oscillator linked to the q2_xy element
                update_frequency("q2_xy", df + qubit_IF_q2)
                # Adjust the idle time
                with if_(delay >= 4):
                    play("x90", "q2_xy")
                    wait(delay, "q2_xy")
                    play("x90", "q2_xy")
                with else_():
                    play("x90", "q2_xy")
                    play("x90", "q2_xy")
                align("q2_xy", "rr2")
                # Measure the state of the rr2.
                measure(
                    "readout",
                    "rr2",
                    None,
                    dual_demod.full("rotated_cos", "rotated_sin", I),
                    dual_demod.full("rotated_minus_sin", "rotated_cos", Q),
                )
                # Wait for the q2_xy to decay to the ground state
                wait(thermalization_time * u.ns, "rr2")
                # Save the 'I' & 'Q' quadratures to their respective streams
                save(I, I_st)
                save(Q, Q_st)
        # Save the averaging iteration to get the progress bar
        save(n, n_st)

    with stream_processing():
        # Cast the data into a 2D matrix, average the 2D matrices together and store the results on the OPX processor
        I_st.buffer(len(dfs)).buffer(len(taus)).average().save("I")
        Q_st.buffer(len(dfs)).buffer(len(taus)).average().save("Q")
        n_st.save("iteration")


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
    job = qmm.simulate(config, ramsey_freq_duration, simulation_config)
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
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(ramsey_freq_duration)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  #  Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        I, Q, iteration = results.fetch_all()
        # Convert the results into Volts
        I, Q = u.demod2volts(I, readout_len), u.demod2volts(Q, readout_len)
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot results
        plt.suptitle("Ramsey chevron")
        plt.subplot(211)
        plt.cla()
        plt.title("I quadrature [V]")
        plt.pcolor(dfs / u.MHz, taus * 4, I)
        plt.xlabel("q2_xy detuning [MHz]")
        plt.ylabel("Idle time [ns]")
        plt.subplot(212)
        plt.cla()
        plt.title("Q quadrature [V]")
        plt.pcolor(dfs / u.MHz, taus * 4, Q)
        plt.xlabel("q2_xy detuning [MHz]")
        plt.ylabel("Idle time [ns]")
        plt.tight_layout()
        plt.pause(0.01)
    plt.show()
    # Save results
    script_name = Path(__file__).name
    data_handler = DataHandler(root_data_folder=save_dir)
    save_data_dict.update({"I_data": I})
    save_data_dict.update({"Q_data": Q})
    save_data_dict.update({"fig_live": fig})
    data_handler.additional_files = {script_name: script_name, **default_additional_files}
    data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])