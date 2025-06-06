from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt
from qualang_tools.results.data_handler import DataHandler
import sys 
import os
import time 

n_avg = 1000


readout_len = readout_len_q2
tau_min = 1 #16 ns
tau_max = 50
tau_step = 1
tau_list = np.arange(tau_min, tau_max, tau_step)

save_data_dict = {
    'n_avgs': n_avg,
    'taus': tau_list,
    'config': config,
}

with program() as swap_test:
    n = declare(int)  # QUA variable for the averaging loop

    tau = declare(int) #drive time

    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature

    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    n_st = declare_stream()  # Stream for the averaging iteration 'n'

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        
        update_frequency("coupler_1", coupler_drive_freq)

        with for_(*from_array(tau, tau_list)):

            play("x180", "q1_xy")
            align()
            play('const', 'coupler_1', duration = tau)
            align()
            measure(
                "readout",
                "rr2",
                dual_demod.full("rotated_cos_q2", "rotated_sin_q2", I),
                dual_demod.full("rotated_minus_sin_q2", "rotated_cos_q2", Q),
            )
            wait(thermalization_time * u.ns)
            save(I, I_st)
            save(Q, Q_st)
        

    with stream_processing():
        I_st.buffer(len(tau_list)).average().save('I')
        Q_st.buffer(len(tau_list)).average().save('Q')
        n_st.save("n")

qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name)

###########################
# Run or Simulate Program #
###########################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, swap_test, simulation_config)
    # Get the simulated samples
    samples = job.get_simulated_samples()
    # Plot the simulated samples
    samples.con1.plot()
    plt.show()
    # Get the waveform report object
    # waveform_report = job.get_simulated_waveform_report()
    # # Cast the waveform report to a python dictionary
    # waveform_dict = waveform_report.to_dict()
    # # Visualize and save the waveform report
    # waveform_report.create_plot(samples, plot=True, save_path=str(Path(__file__).resolve()))
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(swap_test)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["I", "Q", "n"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        I, Q, iteration = results.fetch_all()
        # Convert the results into Volts
        I, Q = u.demod2volts(I, readout_len), u.demod2volts(Q, readout_len)
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        
        
        #plot results 
    plt.plot(tau_list*4,I)
    plt.xlabel("time (ns)")
    plt.ylabel("Frequency (MHz)")
    plt.title("Swap Test Results")

        
    plt.show()

    # Save results
    script_name = Path(__file__).name
    data_handler = DataHandler(root_data_folder=save_dir)
    save_data_dict.update({"I_data": I})
    save_data_dict.update({"Q_data": Q})
    save_data_dict.update({"fig_live": fig})
    data_handler.additional_files = {script_name: script_name, **default_additional_files}
    data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])