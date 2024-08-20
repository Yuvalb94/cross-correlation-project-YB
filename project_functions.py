import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import fnmatch
import os
import seaborn as sns
from scipy import signal

# events_1 = np.loadtxt("ND9_Hab1_g0_tcat.nidq.XD_0_1_0_corr.txt")
# events_3 = np.loadtxt("ND9_Hab1_g0_tcat.nidq.XD_0_3_0_corr.txt")
# events_4 = np.loadtxt("ND9_Hab1_g0_tcat.nidq.XD_0_4_0_corr.txt")

path = r"C:\Users\yuval\OneDrive\Desktop\לימודים\פרויקט מעבדה\plot_psth"


def getEvents(path):
    # this path receives path to all relevant files and returns a dictionary that contains all existing event numbers as keys (0-5), and the list of event times as values for each key.
    os.chdir(path)  # make path location the desired working directory
    all_files = os.listdir(path)  # insert all files and subfolders in path to a list
    pattern = "*_0_corr*" # pattern that appears only in event files' names in path
    events = dict()
    for filename in fnmatch.filter(all_files, pattern):
        # loop that iterates through all files in path and runs through only event files which contain the pattern
        # then the loop also finds the number that appears before the pattern that identifies the taste of the event (sugar/Nacl/acid/quanine/
        #first we need to find the event number:
        event_num = filename[filename.find(pattern.replace("*", "")) - 1]
        events[f"events_{event_num}"] = list(np.loadtxt(filename))
    print("event numbers in this project:", events.keys())
    return events




def getGoodClusterIds(path):
    # this function receives the path to relevant files and returns a list of good cluster ID's
    os.chdir(path)  # make path location the desired working directory
    cluster_info = np.genfromtxt(fname="cluster_info.tsv", skip_header=1)
    KSLabel = np.genfromtxt(fname="cluster_info.tsv", usecols=3, dtype='str', skip_header=1)
    good_idx = KSLabel=='good'
    good_cluster_ID = cluster_info[good_idx, 0] # This will assign values only from the '0' column of cluster_info(IDs), and only if they are labeled 'good'.

    return good_cluster_ID

def getGoodClusterIdsAndDepth(path):
    os.chdir(path)  # make path location the desired working directory
    cluster_info = np.genfromtxt(fname="cluster_info.tsv", skip_header=1)
    KSLabel = np.genfromtxt(fname="cluster_info.tsv", usecols=3, dtype='str', skip_header=1)
    good_idx = KSLabel == 'good'
    good_cluster_ID = cluster_info[good_idx, 0]  # This will assign values only from the '0' column of cluster_info(IDs), and only if they are labeled 'good'.
    good_cluster_depths = cluster_info[good_idx, 6]
    return good_cluster_ID, good_cluster_depths

def getSpikeData(path):
    #this function receives the path to all relevant files, and returns a dictionary and it's keys
    # the dictionary contains list of spike seconds with a corresponding cluster ID for each spike, list of good neuron ID's, and another dictionary with event times
    os.chdir(path)  # make path location the desired working directory
    spike_data = dict() # this dictionary will hold all the data and will be the returned object
    spike_data["spike_sec"] = np.load("spike_seconds.npy")
    spike_data["spike_clusters"] = np.load("spike_clusters.npy")
    spike_data["good_cluster_ID"] = getGoodClusterIds(path)
    spike_data["event_times"] = getEvents(path)
    # spike_data["depth"] =
    print("Returned data:", spike_data.keys())
    return spike_data

def checkThreshold(x, thr1, thr2): # check if more than x(thr2) elements have crossed t(thr1) in list of elements(x)
    # function returns the amount of elements that cross the threshold, Nan if there is none.
    c = [n > thr1 for n in x]
    if sum(c) >= thr2:
        return sum(c)
    else:
        return 0

def findBigNeurons(spike_sec, spike_clusters, good_neurons, threshold):
    big_clusters = []
    for neuron in good_neurons:
        neuron_spikes = spike_sec[spike_clusters == neuron]
        if len(neuron_spikes) > threshold:
            big_clusters.append(neuron)
    return big_clusters

# def plot_PSTH(spike_times, events, start_time, end_time, binsize):
#
#     total_spikes_in_timeframe = []
#     for event in events:
#         working_spike_sec = spike_times[np.logical_and(spike_times > (event + start_time), spike_times < (event + end_time))]
#         total_spikes_in_timeframe = np.append(total_spikes_in_timeframe, (working_spike_sec - event))
#
#     fig, ax = plt.subplots()
#     # n, bins, patches = plt.hist((total_spikes_in_timeframe) * 1000, density=False, bins=binsize, linewidth=0.5, edgecolor="white", alpha=0.5)
#     # ax.hist((total_spikes_in_timeframe) * 1000, density=False, bins=binsize, linewidth=0.5, edgecolor="white")
#     n, bins = np.histogram((total_spikes_in_timeframe) * 1000, bins=binsize)
#     bin_center = bins[:-1] + np.diff(bins) / 2
#     # bin_center2 = bins2[:-1] + np.diff(bins) / 2
#     y_smooth = signal.savgol_filter(n, window_length=7, polyorder=3, mode="nearest")
#     plt.plot(bin_center, y_smooth, linewidth=1, linestyle='-', color='green')
#     plt.vlines(0, ymin=0, ymax=np.max(n), linewidth=0.5, linestyle='dashed', color='black')
#     # plt.vlines(0, ymin=0, ymax=np.max(bins), linewidth=0.5, linestyle='dashed', color='black')
#     plt.xlabel('Post stimulus time (ms)')
#     plt.ylabel('firing rate (spike/s)')
#     # # ax.set_title('neuron %s average response to stimulus' %)
#     y_vals = ax.get_yticks()
#     bin_size_in_seconds = (end_time - start_time) / binsize
#     num_of_events = len(events)
#     yscale = bin_size_in_seconds * num_of_events
#     ax.set_yticklabels(['{:3.0f}'.format(x / yscale) for x in y_vals])
#     # plt.show()
#     # print(f"n: {n} \n size of n:{len(n)} type:{type(n[0])} \n n2: {n2} \n size of n2:{len(n2)} type:{type(n2[0])}")
#     # print(f"bins:{bins} \n size of bins:{len(bins)}\n bins2:{bins2} \n size of bins2:{len(bins2)}")
#     # print(f"bin_center:{bin_center} \n size:{len(bin_center)}  \n bin_center2:{bin_center2} \n size2:{len(bin_center2)}")
#
#     return fig


def plot_PSTH(spike_times, tastes, start_time, end_time, binsize):
    fig, ax = plt.subplots()
    colors = ['blue', 'brown', 'green']
    legend = ['water', 'Nacl', 'Citric acid']
    for i, taste in enumerate(tastes):
        total_spikes_in_timeframe = []
        for event in taste:
            working_spike_sec = spike_times[np.logical_and(spike_times > (event + start_time), spike_times < (event + end_time))]
            total_spikes_in_timeframe = np.append(total_spikes_in_timeframe, (working_spike_sec - event))

        n, bins, patches = plt.hist((total_spikes_in_timeframe) * 1000, density=False, bins=binsize, linewidth=0.5, edgecolor="white", alpha=0)
        # ax.hist((total_spikes_in_timeframe) * 1000, density=False, bins=binsize, linewidth=0.5, edgecolor="white")
        # n, bins = np.histogram((total_spikes_in_timeframe) * 1000, bins=binsize)
        bin_center = bins[:-1] + np.diff(bins) / 2
        y_smooth = signal.savgol_filter(n, window_length=7, polyorder=3, mode="nearest")
        color = colors[i]
        plt.plot(bin_center, y_smooth, linewidth=1, linestyle='-', color=color)
        # plt.vlines(0, ymin=0, ymax=np.max(bins), linewidth=0.5, linestyle='dashed', color='black')
        plt.xlabel('Post stimulus time (ms)')
        plt.ylabel('firing rate (spike/s)')
        # # ax.set_title('neuron %s average response to stimulus' %)
        y_vals = ax.get_yticks()
        bin_size_in_seconds = (end_time - start_time) / binsize
        num_of_events = len(taste)
        yscale = bin_size_in_seconds * num_of_events
        ax.set_yticklabels(['{:3.0f}'.format(x / yscale) for x in y_vals])
    plt.vlines(0, ymin=0, ymax=np.max(n), linewidth=0.5, linestyle='dashed', color='black')
    ax.legend(legend)
        # plt.show()
        # change color
    return fig

# def plot_PSTH_seaborn(spike_times, events, start_time, end_time, binsize):
#     bin_size_in_seconds = (end_time - start_time) / binsize
#     num_of_events = len(events)
#     yscale = bin_size_in_seconds * num_of_events
#     total_spikes_in_timeframe = []
#     for event in events:
#         working_spike_sec = spike_times[np.logical_and(spike_times > (event + start_time), spike_times < (event + end_time))]
#         total_spikes_in_timeframe = np.append(total_spikes_in_timeframe, (working_spike_sec - event))
#     fig, ax = plt.subplots()
#     plot = sns.histplot(x=total_spikes_in_timeframe*1000, kde=True, stat='count', bins = binsize)
#     plot.set(xlabel='Post stimulus time (ms)', ylabel='firing rate (spike/s)', title='single neuron average response to stimulus')
#     y_vals = ax.get_yticks()
#     ax.set_yticklabels(['{:3.0f}'.format(x / yscale) for x in y_vals])
#     plt.show()



if __name__ == "__main__":
    spike_data = getSpikeData(path)
    times_of_events = []
    for key in spike_data["event_times"]:
        times_of_events.append(spike_data["event_times"][key])
    # event_times = spike_data["event_times"]["event_1"]
    spike_times = spike_data["spike_sec"][np.where(spike_data["spike_clusters"] == 134)]
    fig1 = plot_PSTH(spike_times, times_of_events, -0.001, 0.001, 100)
    plt.show()
#      events = getEvents(path)
    # print(events)
#     good_cluster_IDs = getGoodClusterIds(path)
#     spike_sec, spike_clusters = getSpikeData(path)
# print(spike_sec[1:2000], "\n", spike_clusters[1:2000])

