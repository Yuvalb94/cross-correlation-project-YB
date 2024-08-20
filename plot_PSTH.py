import numpy as np
import matplotlib.pyplot as plt
import fnmatch
import seaborn as sns
from scipy import signal

events_1 = np.loadtxt("ND9_Hab1_g0_tcat.nidq.XD_0_1_0_corr.txt")
events_3 = np.loadtxt("ND9_Hab1_g0_tcat.nidq.XD_0_3_0_corr.txt")
events_4 = np.loadtxt("ND9_Hab1_g0_tcat.nidq.XD_0_4_0_corr.txt")


cluster_info = np.genfromtxt(fname="cluster_info.tsv", skip_header=1)
KSLabel = np.genfromtxt(fname="cluster_info.tsv", usecols=3, dtype='str', skip_header=1)
good_idx = KSLabel=='good'
good_cluster_ID = cluster_info[good_idx, 0] # This will assign values only from the '0' column of cluster_info(IDs), and only if they are labeled 'good'.


spike_sec = np.load("spike_seconds.npy")
spike_clusters = np.load("spike_clusters.npy")


def plot_PSTH(spike_times, spike_clusters, cluster_IDs, events, start, end, binsize): # we want to avoid using variables calles end because it is a used word
    start_timeframe = start
    end_timeframe = end # we dont need those lines, we can use the name from when we call the function
    yaxis_scale = 1000 / (((end - start) * 1000) / binsize) # divide in number of events and divide in the width of the bin for time scaling (second)


    for ID in cluster_IDs:
        print("currunt neuron:", ID)
        current_neuron_spikes = spike_times[np.where(spike_clusters == ID)]
        print("number of spikes related to this neuron:", len(current_neuron_spikes))
        sum_of_spikes = np.array([])
        for event in events:
            working_spike_sec = current_neuron_spikes[np.logical_and(current_neuron_spikes > (event + start_timeframe), current_neuron_spikes < (event + end_timeframe))]
            sum_of_spikes = np.append(sum_of_spikes, working_spike_sec-event)
        print("total spike count within events timeframe for this neuron:", len(sum_of_spikes))
        # if(len(sum_of_spikes) > 40):
        #     fig, ax = plt.subplots()
        #     ax.hist((sum_of_spikes) * 1000, bins=binsize, linewidth=0.5, edgecolor="white")
        #     ax.set_xlabel('Post stimulus time (ms)')
        #     ax.set_ylabel('firing rate (spike/s)')
        #     ax.set_title('neuron %s average response to stimulus' %ID)
        #     plt.show()

if __name__ == "__main__":
    plot_PSTH(spike_sec, spike_clusters, good_cluster_ID, events_3, -0.5, 3, 10)

# 1. write event and good cluster ID data reading in distinct functions
# 2. make function more general, the for loops should be outside of the function in accordance to the specific data needs.
# 3. utility functions --> higher level functions
# 4. spikeGLX - format of saving data.
# 5. the task:
#     a. change the code so that it contains little low-level functions for each (action)
#     b. make smooth histogram instead of bins. //savitzky-golay filter?
#     c. then make higher level function(loops)
#     d. data is path to folder