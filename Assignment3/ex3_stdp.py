#*******************************************************************
#   Principles of Brain Computation, SS17
#
#   Template script for Exercise 3
#
#       Robert Legenstein, April 2017
#
#*******************************************************************
# At the beginning we import the necessary Python packages
import nest
from numpy import *       # for numerical operations
from pylab import *       # for plotting (matplotlib)
from pobc_utils import *       # for generating poisson spike trains
import nest.voltage_trace

nest.set_verbosity("M_WARNING") # surpress too much text output

DT = 0.1       # The time step of the simulation [msec]

def generate_stimulus(nchannels, Rs, jitter, Rbase, Tsim):
    # used by construct_input_population
    if Rs == 0.0:
        Soccur = array([])
    else:
        Soccur = poisson_generator(Rs, t_stop=Tsim * 1000) / 1000.0
    spikes = []
    for i in range(nchannels):
        s = append(poisson_generator(Rbase, t_stop=Tsim * 1000) / 1000.0, Soccur + jitter * random.randn(len(Soccur)))
        s.sort()
        s*=1000.0  # times in ms for NEST
        # round to simulation precision
        s *= 10
        s = s.round()+1.0
        s = s/10.0
        spikes.append(s)
    return spikes, Soccur

def generate_stimulus_sequence(nchannels, Rs, jitter, Rbase, Tsim):
    #  used by construct_input_population
    if Rs == 0.0:
        Soccur = array([])
    else:
        Soccur = poisson_generator(Rs, t_stop = Tsim*1000)/1000.0
    spikes = []
    #inp_neuron = SpikingInputNeuron()
    for i in range(nchannels):
        s = append(poisson_generator(Rbase, t_stop = Tsim*1000)/1000.0,i*0.001+Soccur+jitter*random.randn(len(Soccur)))
        s.sort()
        s*=1000.0  # times in ms for NEST
        # round to simulation precision
        s *= 10
        s = s.round()+1.0
        s = s/10.0
        spikes.append(s)
    return spikes, Soccur


def construct_input_population(Nin, jitter, Tsim, sequence):
    # This is a hack.
    # Because in Nest, one cannot connect spike generators with other
    # neurons with STDP synapses, we need to first connect them to a
    # pool of iaf_psc_exp neurons which are then serving as the input pool
    # The pool will produce approximately Poissonian spike trains with rate Rin 
    # Nin...number of input neurons
    # jitter...jitter of population spikes
    # Tsim.....Total simulation time
    # sequence....if True, the stimulus will be shifted in neuron i by i msec
    # Returns:
    # spike_generators...the spike generators' GIDs
    # input_neurons...the input neurons' GIDs
    
    # create input population
        if sequence:
           inp_spikes, s_occur = generate_stimulus_sequence(int(Nin/2), 2.0, jitter, 8.0, Tsim)
        else:
           inp_spikes, s_occur = generate_stimulus(int(Nin/2), 2.0, jitter, 8.0, Tsim)
        inp_spikes_2, s_occur_2 = generate_stimulus(int(Nin/2), 0.0, 0e-3, 8.0, Tsim)
        
        inp_spikes += inp_spikes_2
        
        spike_generators = nest.Create("spike_generator", Nin)
        for (sg, sp) in zip(spike_generators, inp_spikes):
                sp = sp[sp>0]
                nest.SetStatus([sg],{'spike_times': sp})

        input_neurons = nest.Create("iaf_psc_delta",Nin)
        # Choose threshold very close to resting potential so that each spike in a Poisson generator
        # elicits one spike in the corresponding input neuron
        Vresting = -60.0
        nrn_params =  {"V_m": Vresting,     # Membrane potential in mV
                      "E_L": Vresting,      # Resting membrane potential in mV
                      "C_m": 1.0e4/40,      # Capacity of the membrane in pF
                      "tau_m": 0.5,         # Membrane time constant in ms
                      "V_th": -59.9999,     # Spike threshold in mV
                      "V_reset": Vresting,  # Reset potential of the membrane in mV
                      "t_ref": .2           # refractory time in ms
                      }
        nest.SetStatus(input_neurons,nrn_params)
        # Connect Poisson generators to input neurons "one-to-one"
        nest.Connect(spike_generators,input_neurons,{'rule':'one_to_one'},syn_spec={'weight':0.1})
        return spike_generators, input_neurons


def perform_simulation(sequence, jitter=0.0, alpha=1.1, Wmax_fact=2, Tsim=200000.0, W = 20.0e2):
    """
    Performs the network simulation.
    sequence...If True, stimulus in input population will be sequential
    jitter...Jitter on input population events
    alpha....Scaling factor of negative STDP window size A- = -alpha*A+
    W........Initial weight of synapses
    Wmax_fact.....Maximal synaptic weight is given by Wmax = W * Wmax_fact
    Tsim.....Simulation time
    """

    # initializing the network
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 0.1})

    N = 200               # number of input neurons     
    
    #########################################
    # create any neurons, recorders etc. here
    #########################################
    # neurons 
    # V

    # parameters of the neuron
    Vresting = -60.
    nrn_params =  {
                  "V_m": Vresting,      # Membrane potential in mV
                  "E_L": Vresting,      # Resting membrane potential in mV
                  "C_m": 30000.,        # Capacity of the membrane in pF
                  "tau_m": 30.,         # Membrane time constant in ms, R_m*tau_m
                  "V_th": -45.,         # Spike threshold in mV
                  "V_reset": Vresting,  # Reset potential of the membrane in mV
                  "t_ref": 2.,         # refractory time in ms
                  "tau_minus": 30.      # Membrane time constant in ms
                  }

    # creating the neuron
    iaf_neuron = nest.Create("iaf_psc_exp", 1)

    #setting it's parameters
    nest.SetStatus(iaf_neuron, nrn_params)
    
    # recorders 
    spike_detector1 = nest.Create("spike_detector",params={"withgid": True, "withtime": True})
    spike_detector2 = nest.Create("spike_detector", params={"withgid": True, "withtime": True})
    volts = nest.Create("voltmeter")
    nest.SetStatus(volts, {"label": "voltmeter", "withtime": True, "withgid": True})

    # the follwoing creates N input neurons and sets their spike trains during simulation
    spike_generators,input_neurons = construct_input_population(N, jitter, Tsim, sequence)
    
    #########################################
    # Connect nodes, simulate
    #########################################

    # creating the synapses
    # parameters
    syn_param = {
                "alpha": alpha, 
                "lambda": 0.005,
                "tau_plus": 30., 
                "mu_plus": 0.,
                "mu_minus": 0.,
                "Wmax": Wmax_fact * W, 
                "weight": W,
                }

    # set the parameters 
    nest.CopyModel("stdp_synapse", "syn", syn_param)

    # connect the nodes
    nest.Connect(input_neurons, iaf_neuron, {"rule": "all_to_all"} , syn_spec="syn")
    nest.Connect(input_neurons, spike_detector1)
    nest.Connect(iaf_neuron, spike_detector2)
    nest.Connect(volts, iaf_neuron)
    # run the simulation
    weights = zeros((int(Tsim/1000),N+1))
    for i in range(int(ceil(Tsim/1000))):
        nest.Simulate(1000)
        a = nest.GetConnections(target = iaf_neuron)
        weights[i] = nest.GetStatus(a,"weight")


    # To extract spikes of input neuons as a list of numpy-arrays, use the
    # following function provided in nnb_utils:

    spikes_in1 = get_spike_times(spike_detector1)
    spikes_in2 = get_spike_times(spike_detector2)

    #plot the spikes
    # figure(1)
    # plot_raster(spikes_in1[:100], Tsim)
    # figure(2)
    # plot_raster(spikes_in1[100:], Tsim)
    # figure(3)
    # plot_raster(spikes_in2, Tsim)
    # figure(4)
    # nest.voltage_trace.from_device(volts)
    plot_figures(1, 2, spikes_in2 , weights, spikes_in1 , Tsim, "mean weight to time ", "spikes correlqtions " , Tmax_spikes=25)
    show()
    return spikes_in1 # spikes, weight_evolution

def plot_raster(spikes,tmax):
    """
    Spike raster plot for spikes in 'spikes' up to time tmax [in sec]
    spikes[i]: spike times of neuron i in seconds
    """
    i = 0
    for spks in spikes:
        sp = spks[spks<tmax]
        ns = len(sp)
        plot(sp,i*ones(ns),'b.')
        i=i+1


def main():
    print("FUCK PYNEST")
    perform_simulation(False, jitter=0.0, alpha=1.1, Wmax_fact=2, Tsim=20000.0, W=2e3)


main()




