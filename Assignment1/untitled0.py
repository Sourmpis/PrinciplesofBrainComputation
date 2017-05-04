import nest
import pylab


neuron = nest.Create("iaf_psc_alpha")
nest.SetStatus(neuron, {"I_e": 376.0})
multimeter = nest.Create("multimeter")
nest.SetStatus(multimeter, {"withtime":True, "record_from":["V_m"]})
spikedetector = nest.Create("spike_detector", params={"withgid": True, "withtime": True})
nest.Connect(multimeter, neuron)
nest.Connect(neuron, spikedetector)

neuron2 = nest.Create("iaf_neuron")
nest.SetStatus(neuron2 , {"I_e": 375.0})
nest.Connect(multimeter, neuron2)





#------------------------------
nest.Simulate(1000.)
    
dmm = nest.GetStatus(multimeter)[0]
Vms = dmm["events"]["V_m"]
ts = dmm["events"]["times"]

dSD = nest.GetStatus(spike_det,keys="events")[0]
evs = dSD["senders"]
ts = dSD["times"]

pylab.figure(2)
pylab.plot(ts, evs, ".")

pylab.figure(1)


Vms1 = dmm["events"]["V_m"][::2] # start at index 0: till the end: each second entry
ts1 = dmm["events"]["times"][::2]
pylab.plot(ts1, Vms1)



Vms2 = dmm["events"]["V_m"][1::2] # start at index 1: till the end: each second entry
ts2 = dmm["events"]["times"][1::2]

pylab.plot(ts2, Vms2)
pylab.show()    