## PoBC 2017, ex4

import nest
import nest.raster_plot
import pylab
import numpy as np
from pobc_utils import get_spike_times, poisson_generator


def generate_stimuls_mem(dt_stim, stim_len, Rs, Tsim):
    # Creates stimulus spikes for two input neurons
    # dt_stim...stimulus bursts come everey dt_stim ms
    # stim_len..length of stimulus burst in [ms]
    # Rs........rate of stimulus burst [Hz]
    # Tsim......simulation time [ms]
    # returns
    # spikes....s[i] spike times of i-th neuron [ms]
    spikes = [np.array([]), np.array([])]
    Nstim = int(np.floor((Tsim - stim_len) / dt_stim))
    targets = np.random.randint(2, size=Nstim)
    t = dt_stim
    for rb in targets:
        spikes[rb] = np.append(spikes[rb], t + poisson_generator(Rs, t_stop=stim_len))
        t = t + dt_stim
    # round to simulation precision
    for i in range(len(spikes)):
        spikes[i] *= 10
        spikes[i] = spikes[i].round() + 1.0
        spikes[i] = spikes[i] / 10.0

    return spikes, targets


def generate_stimuls_xor(dt_stim, stim_len, Rs, Tsim):
    # Creates stimulus spikes for two input neurons
    # dt_stim...stimulus bursts come everey dt_stim ms
    # stim_len..length of stimulus burst in [ms]
    # Rs........rate of stimulus burst [Hz]
    # Tsim......simulation time [ms]
    # returns
    # spikes....s[i] spike times of i-th neuron [ms]
    spikes = [np.array([]), np.array([])]
    Nstim = int(np.floor((Tsim - stim_len) / dt_stim))
    in1 = np.random.randint(2, size=Nstim)
    in2 = np.random.randint(2, size=Nstim)
    t = dt_stim
    for rb in in1:
        if rb == 1:
            spikes[0] = np.append(spikes[0], t + poisson_generator(Rs, t_stop=stim_len))
        t = t + dt_stim
    t = dt_stim
    for rb in in2:
        if rb == 1:
            spikes[1] = np.append(spikes[1], t + poisson_generator(Rs, t_stop=stim_len))
        t = t + dt_stim
    # round to simulation precision
    for i in range(len(spikes)):
        spikes[i] *= 10
        spikes[i] = spikes[i].round() + 1.0
        spikes[i] = spikes[i] / 10.0
    targets_bin = np.logical_xor(in1, in2)
    targets = np.zeros(len(targets_bin))
    targets[targets_bin] = 1

    return spikes, targets


def get_liquid_states(spike_times, times, tau):
    # returns the liquid states
    # spike_times[i]...numpy-array of spike-times of neuron i in [sec]
    # times............tunes when liquid states should be extracted [sec]
    # tau..............time constant for liquid state filter [sec]
    # returns:
    # states... numpy array with states[i,j] the state of neuron j in example i
    N = np.size(spike_times, 0)
    T = np.size(times, 0)
    states = np.zeros((T, N))
    t_window = 3 * tau
    n = 0
    for spt in spike_times:

        spt2 = spt.__copy__()

        t_idx = T - 1
        for t in reversed(times):
            spt2 = spt2[spt2 < t]
            cur_times = spt2[spt2 >= t - t_window]
            states[t_idx, n] = sum(np.exp(-(t - cur_times) / tau))
            t_idx -= 1
        n += 1
    return states


def divide_train_test(states, targets, train_frac):
    # divides liquid states and targets into
    # training set and test set
    # randomly chooses round(train_frac*len(targets)) exmaples for training, rest for testing
    # states... numpy array with states[i,j] the state of neuron j in example i
    # targets.. the targets for training/testing. targets[i] is target of example i
    # train fraction...fraction in (0,1) of training examples
    # returns:
    #    states_train..training states in same format as states
    #    states_test...test states in same format as states
    #    targets_train..training targets in same format as targets
    #    targets_test...test targets in same format as targets
    Nstates = np.size(states, 0)
    Ntrain = round(Nstates * train_frac)
    # Ntest = Nstates - Ntrain
    idx_states = np.random.permutation(Nstates)
    idx_train = idx_states[:Ntrain]
    idx_test = idx_states[Ntrain:]
    states_train = states[idx_train, :]
    states_test = states[idx_test, :]
    targets_train = targets[idx_train]
    targets_test = targets[idx_test]
    return states_train, states_test, targets_train, targets_test


def train_readout(states, targets, reg_fact=0):
    # train readout with linear regression
    # states... numpy array with states[i,j] the state of neuron j in example i
    # targets.. the targets for training/testing. targets[i] is target of example i
    # reg_fact..regularization factor. If set to 0, no regularization is performed
    # returns:
    #    w...weight vector
    if reg_fact == 0:
        w = np.linalg.lstsq(states, targets)[0]
    else:
        w = np.dot(np.dot(pylab.inv(reg_fact * pylab.eye(np.size(states, 1)) + np.dot(states.T, states)), states.T),
                   targets)

    return w


def test_readout(w, states, targets):
    # compute misclassification rate of linear readout with weights w
    # states... numpy array with states[i,j] the state of neuron j in example i
    # targets.. the targets for training/testing. targets[i] is target of example i
    # returns:
    #   err...the misclassification rate
    yr = np.dot(states, w)  # compute prediction
    # compute error
    y = np.zeros(np.size(yr))
    y[yr >= 0.5] = 1
    err = (1. * sum(y != targets)) / len(targets)
    return err


def main():
    simtime = 40000.  # how long shall we simulate [ms]

    N_rec = 500  # Number of neurons to record from

    # Network parameters.
    delay_dict = dict(distribution='normal_clipped', mu=10., sigma=20., low=3., high=200.)

    N_E = 1000  # 1000  # number of excitatory neurons
    N_I = 250  # 250  # number of inhibitory neurons
    N_neurons = N_E + N_I  # total number of neurons

    C_E = 2  # number of excitatory synapses per neuron
    C_I = 1  # number of inhibitory synapses per neuron
    C_inp = 100  # number of outgoing input synapses per input neuron

    w_scale = 10.0
    J_EE = w_scale * 5.0  # strength of E->E synapses [pA]
    J_EI = w_scale * 25.0  # strength of E->I synapses [pA]
    J_IE = w_scale * -20.0  # strength of inhibitory synapses [pA]
    J_II = w_scale * -20.0  # strength of inhibitory synapses [pA]
    J_noise = 5.0  # strength of synapses from noise input [pA]

    p_rate = 100.0  # this is used to simulate input from neurons around the populations

    # Set parameters of the NEST simulation kernel
    nest.SetKernelStatus({'print_time': True,
                          'local_num_threads': 1}) # increase if you can use more threads

    # Create nodes -------------------------------------------------

    nest.SetDefaults('iaf_psc_exp',
                     {'C_m': 30.0,  # 1.0,
                      'tau_m': 30.0,
                      'E_L': 0.0,
                      'V_th': 15.0,
                      'tau_syn_ex': 3.0,
                      'tau_syn_in': 2.0,
                      'V_reset': 13.8,
                      'I_e':14.5})

    # Create excitatory and inhibitory populations
    exc = nest.Create('iaf_psc_exp',N_E)
    inh = nest.Create('iaf_psc_exp',N_I)
    # Create noise input
    noise = nest.Create('poisson_generator', 1, {'rate': p_rate})

    # create spike detectors from excitatory and inhibitory populations
    spike_detector_I = nest.Create("spike_detector",params={"withgid": True, "withtime": True})
    spike_detector_E = nest.Create("spike_detector",params={"withgid": True, "withtime": True})

    # create input generators
    dt_stim = 300.  #[ms]
    stim_len = 50.  #[ms]
    Rs = 200.  #[Hz]
    # this is for ex 4A
    inp_spikes, targets = generate_stimuls_xor(dt_stim, stim_len, Rs, simtime)
    # this is for ex 4B
    # inp_spikes, targets = generate_stimuls_mem(dt_stim, stim_len, Rs, simtime)

    # create two spike generators,
    # set their spike_times of i-th generator to inp_spikes[i]
    spike_generators = nest.Create("spike_generator", 2)
    for (sg, sp) in zip(spike_generators, inp_spikes):
        nest.SetStatus([sg], {'spike_times': sp})

    # Connect nodes ------------------------------------------------

    # dynamic parameters
    f0 = 10.

    def get_u_0(U, D, F):
        return U / (1 - (1 - U) * np.exp(-1 / (f0 * F)))

    def get_x_0(U, D, F):
        return (1 - np.exp(-1 / (f0 * D))) / (1 - (1 - get_u_0(U, D, F)) * np.exp(-1 / (f0 * D)))


    syn_param_EE = {"tau_psc": 2.0,
                    "tau_fac": 1.,  # facilitation time constant in ms
                    "tau_rec": 813.,  # recovery time constant in ms
                    "U": 0.59,  # utilization
                    "u": get_u_0(0.59, 813., 1.),
                    "x": get_x_0(0.59, 813., 1.)
                    }
    nest.CopyModel("tsodyks_synapse", "EE", syn_param_EE)  # synapse model for E->E connections
    # connect E to E with EE synapse model and fixed indegree C_E. Specify the delay and weight distribution here.
    connection_rule_ex = {'rule': 'fixed_indegree', 'indegree': C_E}


    nest.Connect(exc,exc,connection_rule_ex,{'model':"EE",'weight':{'distribution': 'normal',
                                                                    'mu': J_EE,'sigma': 0.7*J_EE },
                                             'delay':delay_dict})

    syn_param_EI = {"tau_psc": 2.0,
                    "tau_fac": 1790.,  # facilitation time constant in ms
                    "tau_rec": 399.,  # recovery time constant in ms
                    "U": 0.049,  # utilization
                    "u": get_u_0(0.049, 399., 1790.),
                    "x": get_x_0(0.049, 399., 1790.)
                    }
    nest.CopyModel("tsodyks_synapse", "EI", syn_param_EI)  # synapse model for E->I connections
    # connect E to I with EI synapse model and fixed indegree C_E. Specify the delay and weight distribution here.
    nest.Connect(exc,inh,connection_rule_ex,{'model':"EI",'weight':{'distribution': 'normal',
                                                                    'mu': J_EI,'sigma': 0.7*J_EI },
                                             'delay':delay_dict})


    syn_param_IE = {"tau_psc": 2.0,
                    "tau_fac": 376.,  # facilitation time constant in ms
                    "tau_rec": 45.,  # recovery time constant in ms
                    "U": 0.016,  # utilization
                    "u": get_u_0(0.016, 45., 376.),
                    "x": get_x_0(0.016, 45., 376.)
                    }
    nest.CopyModel("tsodyks_synapse", "IE", syn_param_IE)  # synapse model for I->E connections
    # connect I to E with IE model and fixed indegree C_I. Specify the delay and weight distribution here.
    connection_rule_in = {'rule': 'fixed_indegree', 'indegree': C_I}

    nest.Connect(inh,exc,connection_rule_in,{'model':"IE",'weight':{'distribution': 'normal',
                                                                    'mu': J_IE,'sigma': -0.7*J_IE },
                                             'delay':delay_dict})

    syn_param_II = {"tau_psc": 2.0,
                    "tau_fac": 21.,  # facilitation time constant in ms
                    "tau_rec": 706.,  # recovery time constant in ms
                    "U": 0.25,  # utilization
                    "u": get_u_0(0.25, 706., 21.),
                    "x": get_x_0(0.25, 706., 21.)
                    }
    nest.CopyModel("tsodyks_synapse", "II", syn_param_II)  # synapse model for I->I connections
    # connect I to I with II model and fixed indegree C_E. Specify the delay and weight distribution here.
    nest.Connect(inh,inh,connection_rule_in,{'model':"II",'weight':{'distribution': 'normal',
                                                                    'mu': J_II,'sigma': -0.7*J_II },
                                             'delay':delay_dict})

    # connect one noise generator to all neurons
    nodes = exc+inh
    nest.CopyModel('static_synapse_hom_w', 'excitatory_noise', {'weight': J_noise})
    nest.Connect(noise, nodes, syn_spec={'model': 'excitatory_noise', 'delay': delay_dict})

    # connect input neurons to E-pool
    # Each input neuron makes C_input synapses
    # distribute weights uniformly in (2.5*J_EE, 7.5*J_EE)

    connection_rule_input = {'rule': 'fixed_outdegree', 'outdegree':C_inp}
    nest.Connect(spike_generators,exc,connection_rule_input,{'model':'static_synapse',
                                                                  "weight": {"distribution": "uniform", "low": 125., "high": 375.},
                                                                  'delay':delay_dict})
    # connect all recorded E/I neurons to the respective detector
    nest.Connect(exc,spike_detector_E)
    nest.Connect(inh, spike_detector_I)
    # SIMULATE!! -----------------------------------------------------
    nest.Simulate(simtime)




    #compute excitatory rate

    spikes_E = get_spike_times(spike_detector_E)
    number_of_spikes_E = 0
    for i in range(N_E):
        number_of_spikes_E += len(spikes_E[i])
    rate_ex = number_of_spikes_E/N_E/simtime*1000.
    print(('Excitatory rate   : {:.2f} Hz'.format(rate_ex)))

    #compute inhibitory rate

    spikes_I = get_spike_times(spike_detector_I)
    number_of_spikes_I = 0
    for i in range(N_I):
        number_of_spikes_I += len(spikes_I[i])
    rate_in = number_of_spikes_I / N_I / simtime * 1000.
    print(('Inhibitory rate   : {:.2f} Hz'.format(rate_in)))



    # To plot network activity
    # nest.raster_plot.from_device(spike_detector_E, hist=False, title='')
    # pylab.show()

    # train the readout on 20 randomly chosen training sets

    NUM_TRAIN = 20
    error = np.zeros((NUM_TRAIN,))
    train_error = np.zeros((NUM_TRAIN,))
    TRAIN_READOUT = True

    tau_lsm = 0.50  # [sec]
    # readout_delay = 0.01  # [sec]
    spike_times = spikes_E  # returns spike times in seconds

    readout_delay = np.linspace(150,150,1)
    mean_error = np.zeros(len(readout_delay), )
    std_error = np.zeros(len(readout_delay), )

    for l, k in enumerate(readout_delay):

        rec_time_start = (dt_stim / 1000 + stim_len / 1000 + k/1000)  # time of first liquid state [sec]
        times = np.arange(rec_time_start, simtime / 1000, dt_stim / 1000)  # times when liquid states are extracted [sec]

        print("Extract Liquid States...")

        states = get_liquid_states(spike_times, times, tau_lsm)
        print(np.trace(np.dot(states.T, states)))
        np.save("states",states)
        refactor = np.array(np.linspace(0.,5000,10))
        mean_error = np.zeros(len(refactor), )
        std_error = np.zeros(len(refactor), )
        mean_error_train= np.zeros(len(refactor), )
        std_error_train= np.zeros(len(refactor), )
        for j,reg in enumerate(refactor):
            for i in range(NUM_TRAIN):
                if TRAIN_READOUT:


                    states_train, states_test, targets_train, targets_test = divide_train_test(states, targets, train_frac= 0.8)

                    w = train_readout(states_train, targets_train, reg)
                    trainerror = test_readout(w, states_train, targets_train)
                    err = test_readout(w, states_test, targets_test)
                    # print(err)
                    train_error[i] = trainerror
                    error[i] = err
                    # don't forget to add constant component to states for bias

            mean_error[j] =np.mean(error)
            std_error[j] = np.std(error)
            mean_error_train[j] = np.mean(train_error)
            std_error_train[j] = np.std(train_error)
            # print(refactor[j], mean_error[j], std_error[j])

    pylab.plt.plot(refactor, mean_error_train, 'o', linestyle='-')
    pylab.plt.fill_between(refactor, mean_error_train - std_error_train, mean_error_train + std_error_train,alpha=.2)
    pylab.plt.plot(refactor, mean_error, 'o', linestyle='-')
    pylab.plt.fill_between(refactor, mean_error - std_error, mean_error + std_error, alpha=.2)

    pylab.plt.show()
if __name__ == "__main__":
    main()
