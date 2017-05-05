#from numpy import array, log
import numpy
from pylab import *
from numpy import *
import nest

def get_spike_times(spike_rec):
    """
       Takes a spike recorder spike_rec and returns the spikes in a list of numpy arrays.
       Each array has all spike times of one sender (neuron) in units of [sec]
    """
    events = nest.GetStatus(spike_rec)[0]['events']
    min_idx = min(events['senders'])
    max_idx = max(events['senders'])
    spikes = []
    for i in range(min_idx,max_idx+1):
	    idx = find(events['senders']==i)
	    spikes.append(events['times'][idx]/1000.0)  # convert times to [sec]
    return spikes
    
def cross_correlate_spikes(s1, s2, binsize, corr_range):
    # Compute cross-correlation between two spike trains
    # The implementation is rather inefficient
    cr_lo = corr_range[0]
    cr_hi = corr_range[1]
    ttt = corr_range[1]-corr_range[0]
    Nbins = int(ceil(ttt/binsize))
    Nbins_h = round(Nbins/2)
    corr = zeros(Nbins+1)
    s1a = append(s1,inf)

    for t in s2:
	    idx = 0
	    while s1a[idx]<t+cr_lo:
		    idx +=1
	    while s1a[idx]<t+cr_hi:
		    idxc = int((t-s1a[idx])/binsize+Nbins_h) #HERE
		    corr[idxc] += 1
		    idx +=1
    return corr
    
def avg_cross_correlate_spikes(spikes, num_pairs, binsize, corr_range):
    """
       computes average cross-crrelation between pairs of spike trains in spikes in the
       range defince by corr_range and with bin-size defined by binsize.
    """
    i = random.randint(len(spikes))
    j = random.randint(len(spikes))
    if i == j: 
        j = (i + 1) % len(spikes)
    s1 = spikes[i]
    s2 = spikes[j]
    corr = cross_correlate_spikes(s1, s2, binsize, corr_range)
    #for p in xrange(1, num_pairs): #Python2
    for p in range(1, num_pairs):
            i = random.randint(len(spikes))
            j = random.randint(len(spikes))
            if i == j: 
                j = (i + 1) % len(spikes)
            s1 = spikes[i]
            s2 = spikes[j]
            corr += cross_correlate_spikes(s1, s2, binsize, corr_range)
    return corr


def avg_cross_correlate_spikes_2sets(spikes1,spikes2, binsize, corr_range):
    s1 = spikes1[0]
    s2 = spikes2    #HERE
    corr = cross_correlate_spikes(s1, s2, binsize, corr_range)
    for i in range(1,len(spikes1)):
        for j in range(1,len(spikes2)):
            s1 = spikes1[i]
            s2 = spikes2     #HERE
            corr += cross_correlate_spikes(s1, s2, binsize, corr_range)
    return corr

def poisson_generator(rate, t_start=0.0, t_stop=1000.0, rng = None):
    """
    Returns a SpikeTrain whose spikes are a realization of a Poisson process
    with the given rate (Hz) and stopping time t_stop (milliseconds).

    Note: t_start is always 0.0, thus all realizations are as if 
    they spiked at t=0.0, though this spike is not included in the SpikeList.

    Inputs:
        rate    - the rate of the discharge (in Hz)
        t_start - the beginning of the SpikeTrain (in ms)
        t_stop  - the end of the SpikeTrain (in ms)
        array   - if True, a numpy array of sorted spikes is returned,
                  rather than a SpikeTrain object.

    Examples:
        >> gen.poisson_generator(50, 0, 1000)
        >> gen.poisson_generator(20, 5000, 10000, array=True)
     
    See also:
        inh_poisson_generator
    """
    
    if rng==None:
        rng = random

    #number = int((t_stop-t_start)/1000.0*2.0*rate)
    
    # less wasteful than double length method above
    n = (t_stop-t_start)/1000.0*rate
    number = int(numpy.ceil(n+3*numpy.sqrt(n)))
    if number<100:
        number = min(5+numpy.ceil(2*n),100)
    
    if number > 0:
        isi = rng.exponential(1.0/rate, number)*1000.0
        if number > 1:
            spikes = numpy.add.accumulate(isi)
        else:
            spikes = isi
    else:
        spikes = numpy.array([])

    spikes+=t_start
    i = numpy.searchsorted(spikes, t_stop)

    extra_spikes = []
    if i==len(spikes):
        # ISI buf overrun
        
        t_last = spikes[-1] + rng.exponential(1.0/rate, 1)[0]*1000.0

        while (t_last<t_stop):
            extra_spikes.append(t_last)
            t_last += rng.exponential(1.0/rate, 1)[0]*1000.0
        
        spikes = numpy.concatenate((spikes,extra_spikes))

    else:
        spikes = numpy.resize(spikes,(i,))

    return spikes

def plot_figures(fig1,fig2, spikes, weights, inp_spikes, Tsim, filename_fig1, filename_fig2, Tmax_spikes=25):
    """
    This function plots two figures for analysis of results
    fig1,fig2....figure identifiers
    spikes.......spikes of the output neuron
    weights......recorded weights over time (column t is the weight vector at recording time index t
                 The function assumes that weights are recorded every second.
    inp_spikes...spikes of input neurons as list of numpy arrays (use get_spike_times)
    Tsim.........simulation time
    filename_figX...filenames of figures to save figures to file
    Tmax_spikes.....Computation of spike cross-correlations may take some time
                    Use Tmax_spikes to compute the cc only over time (0,Tmax_spike)
    """
    # crop spike times in order to save time during convolution:
    weights = transpose(weights)  # HERE


    Nin = len(weights)
    Nin2 = int(Nin/2)


    spikes = transpose(spikes)
    spikes = spikes[:Tmax_spikes] #HERE keep the first Tmax spikes
    for i in range(inp_spikes.__len__()):
        inp_spikes[i] = inp_spikes[i][:Tmax_spikes] #HERE

    f = figure(fig1, figsize = (8,3.6   ))
    f.subplots_adjust(top= 0.89, left = 0.09, bottom = 0.15, right = 0.93, hspace = 0.30, wspace = 0.40)



    ax = subplot(1,2,1)
    imshow(weights, aspect = 'auto')
    xlim(0,Tsim/1000.)
    xlabel('time [sec]')
    colorbar()
    ylabel('synapse id.')
    text(-0.19, 1.07, 'A', fontsize = 'large', transform = ax.transAxes)

    ax = subplot(1,2,2)
    mean_up = mean(weights[0:Nin2,:], axis = 0)
    mean_down = mean(weights[Nin2:Nin,:], axis = 0)
    std_up = std(weights[0:Nin2,:], axis = 0)
    std_down = std(weights[Nin2:Nin,:], axis = 0)
    plot(linspace(0,Tsim,len(mean_up)), mean_up, color = 'b')

    plot(linspace(0,Tsim,len(mean_down)), mean_down, color = 'r')
    errorbar(linspace(0, Tsim, len(mean_up))[::20], mean_up[::20], std_up[::20], fmt = 'b.')
    errorbar(linspace(0, Tsim, len(mean_down))[::20], mean_down[::20], std_down[::20], fmt = 'r.')
    xlabel('time [sec]')
    ylabel('avg. syn. weight')
    text(-0.19, 1.07, 'B', fontsize = 'large', transform = ax.transAxes)
    
    savefig(filename_fig1)       

    f = figure(fig2, figsize = (8,8))
    f.subplots_adjust(top= 0.93, left = 0.09, bottom = 0.12, right = 0.95, hspace = 0.40, wspace = 0.40)
    ax = subplot(2,2,1)
    corr = avg_cross_correlate_spikes(inp_spikes[0:Nin2], 200, binsize = 5e-3, corr_range = (-100e-3,100e-3))
    plot(arange(-100e-3,101e-3, 5e-3), corr, marker = 'o')
    xlim(-100e-3,100e-3)
    xticks(list(arange(-0.1,0.101,0.05)), [ '-0.1', '-0.05', '0', '0.05', '0.1' ] )
    xlabel('time lag [sec]')
    axvline(0.0)
    title('input correl. first group', fontsize = 15)
    ylabel('counts/bin')
    text(-0.19, 1.07, 'A', fontsize = 'large', transform = ax.transAxes)

    
    
    ax = subplot(2,2,2)
    corr = avg_cross_correlate_spikes(inp_spikes[Nin2:Nin], 200, binsize = 5e-3, corr_range = (-100e-3,100e-3))
    plot(arange(-100e-3,101e-3, 5e-3), corr, marker = 'o')
    xlim(-100e-3,100e-3)
    xticks(list(arange(-0.1,0.101,0.05)), [ '-0.1', '-0.05', '0', '0.05', '0.1' ] )
    xlabel('time lag [sec]')
    ylabel('counts/bin')
    title('input correl. second group', fontsize = 15)
    axvline(0.0)
    text(-0.19, 1.07, 'B', fontsize = 'large', transform = ax.transAxes)

    
    ax = subplot(2,2,3)
    corr = avg_cross_correlate_spikes_2sets(inp_spikes[0:Nin2], spikes, binsize = 5e-3, corr_range = (-100e-3,100e-3))#HERE
    plot(arange(-100e-3,101e-3, 5e-3), corr, marker = 'o')
    xlim(-100e-3,100e-3)
    xticks(list(arange(-0.1,0.101,0.05)), [ '-0.1', '-0.05', '0', '0.05', '0.1' ] )
    xlabel('time lag [sec]')
    title('input-output correl. first group', fontsize = 15)
    ylabel('counts/bin')
    axvline(0.0)
    text(-0.19, 1.07, 'C', fontsize = 'large', transform = ax.transAxes)

    
    ax = subplot(2,2,4)
    corr = avg_cross_correlate_spikes_2sets(inp_spikes[Nin2:Nin], spikes, binsize = 5e-3, corr_range = (-100e-3,100e-3)) #HERE
    plot(arange(-100e-3,101e-3, 5e-3), corr, marker = 'o')
    xlim(-100e-3,95e-3)
    xticks(list(arange(-0.1,0.101,0.05)), [ '-0.1', '-0.05', '0', '0.05', '0.1' ] )
    title('input-output correl. second group', fontsize = 15)
    xlabel('time lag [sec]')
    ylabel('counts/bin')
    axvline(0.0)
    text(-0.19, 1.07, 'D', fontsize = 'large', transform = ax.transAxes)

    savefig(filename_fig2)

