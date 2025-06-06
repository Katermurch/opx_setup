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
from analysis import fit_sine,fit_sine_fix_freq,fit_gaussian_points

##################
#   Parameters   #
##################
# Parameters Definition
n_avg = 100  # Number of averaging loops

# readout amp sweep in Hz
a_min = 0
a_max = .2
da = 0.005
amplitudes = np.arange(a_min, a_max + da / 2, da)

# phase sweewp of second 90 pulse 
phi_max = 1
d_phi = 0.02
phi_array = np.arange(0, phi_max, d_phi)


# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "ro_amp": amplitudes,
    "phi_array": phi_array,
    "config": config,
}

###################
# The QUA program #
###################
with program() as ramsey_freq_duration:
    n = declare(int)  # QUA variable for the averaging loop
    a = declare(fixed)  # QUA variable for the q2_xy detuning
    phi = declare(fixed)  # QUA variable for the idle time
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    n_st = declare_stream()  # Stream for the averaging iteration 'n'

    with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
        with for_(*from_array(a, amplitudes)): 
            reset_if_phase("q2_xy") 
            reset_frame("q2_xy")# QUA for_ loop for sweeping the idle time
            with for_(*from_array(phi, phi_array)):
                reset_if_phase("q2_xy")   # QUA for_ loop for sweeping the q2_xy frequency
                play("x90", "q2_xy")
                align("q2_xy", "rr2") 
                play("readout"*amp(a),"rr2") # Align the q2_xy and rr2 elements
                align("rr2","q2_xy") 
                frame_rotation_2pi(phi, "q2_xy")  # Rotate the frame of the q2_xy element by phi
                wait(100, "q2_xy")
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
        I_st.buffer(len(phi_array)).buffer(len(amplitudes)).average().save("I")
        Q_st.buffer(len(phi_array)).buffer(len(amplitudes)).average().save("Q")
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
        plt.pcolor(phi_array , amplitudes * readout_amp_q2, I)
        plt.xlabel("phase")
        plt.ylabel("amplitude")
        plt.subplot(212)
        plt.cla()
        plt.title("Q quadrature [V]")
        plt.pcolor(phi_array , amplitudes * readout_amp_q2, Q)
        plt.xlabel("phase")
        plt.ylabel("amplitude")
        plt.tight_layout()
        plt.pause(0.01)
    plt.show()
    plt.plot(phi_array,I[0])
    plt.show()
    fit_vals_sine,_,_,_ = fit_sine(phi_array,I[0],guess_vals=[.16 ,100,-67.76128980276253,-263.4006117568909])
    ramsey_amp = np.zeros(I.shape[0])
    for i in range(I.shape[0]):
        fit_data = I[i] 
        fit_vals, _, _, _ = fit_sine_fix_freq(
            phi_array, fit_data, guess_vals=[.01, -64.10931689530595, 0.00012702578101351036], fixed_freq=fit_vals_sine[0]
        )
        ramsey_amp[i] = abs(fit_vals[0])
        print(f'Index {i}: Amp={fit_vals[0]:.5f}')
    Gaussian_fit_vals,_,y_vals,_ = fit_gaussian_points(amplitudes * readout_amp_q2,ramsey_amp,guess_vals=[0.03,0,.1])
    plt.scatter(amplitudes * readout_amp_q2,ramsey_amp)
    plt.plot(amplitudes * readout_amp_q2, y_vals)
    plt.xlabel("RO_ram_amp")
    plt.ylabel("Ramsey Amplitude")
    plt.title("Ramsey Amplitude vs RO_ram_amp")

    # Save results
    script_name = Path(__file__).name
    data_handler = DataHandler(root_data_folder=save_dir)
    save_data_dict.update({"I_data": I})
    save_data_dict.update({"Q_data": Q})
    save_data_dict.update({"fig_live": fig})
    data_handler.additional_files = {script_name: script_name, **default_additional_files}
    data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])

    