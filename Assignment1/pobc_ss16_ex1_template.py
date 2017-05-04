#*******************************************************************
#   PoBC, SS16
#
#   Template script for Task 1B 
#
#   A leaky integrate-and-fire neuron is created.
#   It gets a spike input at time t=105 ms 
#   There is also an optional step current input at time t=100 ms that forces the neuron to spike
#   
#
#
#*******************************************************************
# At the beginning we import the necessary Python packages
from numpy import *       # for numerical operations
import  pylab     # for plotting (matplotlib)
import nest
import nest.voltage_trace




###########################################
#  Parameter for the LIF neuron
###########################################
Rm = 10.0 # [MOhms]
Cm = 3000. # [pF]
tau_m = Rm*Cm/1000.0  # membrane time constant [ms]
tau_s = 7.0;     # synaptic time constant [ms]
Trefract = 10.   # The refractory period of the LIF neuron [ms]
Vthresh = -45.   # The threshold potential of the LIF neuron [mV]
Vresting = -60.  # The resting potential of the LIF neuron [mV]

t_spike_input = 105.
t_step = 100.
step_duration = 0.5
step_amplitude = 90752.5


nrn_parameter_dict = {"V_m": Vresting,     # Membrane potential in mV
                      "E_L": Vresting,     # Resting membrane potential in mV
                      "C_m": Cm,           # Capacity of the membrane in pF
                      "tau_m": tau_m,      # Membrane time constant in ms
                      "t_ref": Trefract,   # Duration of refractory period in ms
                      "V_th": Vthresh,     # Spike threshold in mV
                      "V_reset": Vresting, # Reset potential of the membrane in mV

                      "tau_syn_ex": tau_s, # Time constant of the excitatory synaptic current in ms
                      "I_e": 0.0           # No constant external input current
                      }
# The other neuron parameters have default values.


# Reset the NEST Simulator
nest.ResetKernel()

###################################
# Create nodes
###################################
    
# Create the IAF neuron, see http://www.nest-simulator.org/cc/iaf_psc_exp/
neuron = nest.Create("iaf_psc_exp", params=nrn_parameter_dict)

# Create inputs
spike_gen = nest.Create("spike_generator", params = {"spike_times": array([t_spike_input])})
step_gen = nest.Create('step_current_generator')
nest.SetStatus(step_gen, {'amplitude_times':array([t_step,t_step+step_duration]),'amplitude_values':array([step_amplitude, 0.])})

# step_amplitude is in [pA]
# needs a large value to produce a spike

# Create voltmeter and spike recorder
multim = nest.Create('multimeter')
nest.SetStatus(multim, {"withtime":True, "record_from":["V_m"]})
spike_det = nest.Create('spike_detector', params={"withgid": True, "withtime": True})

###################################
# Connect nodes
###################################
     
# Connect spike generator to neuron
# Note: The connection weight is given in [pA]
nest.Connect(spike_gen, neuron)
# Connect current step input step_gen to the neuron
# Note: The current amplitudes as defined above are multiplied with the weight.
nest.Connect(step_gen, neuron)
# Connect voltmeter and spike recorder to neuron

nest.Connect(neuron, spike_det)
nest.Connect(multim, neuron)
 
###################################
# Now simulate 
###################################
    
nest.Simulate(2.)
###################################
# Analyze results and make plots
###################################


# Extract spikes voltage
dmm = nest.GetStatus(multim)[0]
Vms = dmm["events"]["V_m"]
ts = dmm["events"]["times"]

pylab.figure(1)
pylab.plot(ts, Vms)

dSD = nest.GetStatus(spike_det,keys="events")[0]
evs = dSD["senders"]
ts = dSD["times"]

pylab.figure(2)
pylab.plot(ts, evs, ".")

pylab.show()
# etc.