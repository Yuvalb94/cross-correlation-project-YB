import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from project_functions import getSpikeData
from project_functions import checkThreshold

path = r"C:\Users\yuval\OneDrive\Desktop\לימודים\פרויקט מעבדה\plot_psth"
## abbreviations:
## / ntc = neuron to correlate
## / c = counts, b = bins, b_e = bin edges, b_c = bin centers, s = total spikes.

def getNeuronSpikes(neuron_id, spike_sec, spike_clusters):
    neuron_spikes = spike_sec[np.where(spike_clusters == neuron_id)]
    return neuron_spikes

def getCorrelatedData(base_neuron_spikes, neuron_to_correlate, time_interval, binwidth=0.001):
    num_bins = int(2*time_interval/binwidth) # set number of bins so that each bin will be 1ms
    bincount = np.zeros(num_bins) # this variable will count and hold the additions to each bin (sort of a magazine)
    bins = np.linspace(-(time_interval+binwidth), time_interval, num_bins+1) # bin edges
    total_spikes = []
    for spike in base_neuron_spikes:
        spikes_in_timeframe = neuron_to_correlate[np.logical_and(neuron_to_correlate > (spike - time_interval), neuron_to_correlate < (spike + time_interval))]
        total_spikes = np.append(total_spikes, (spikes_in_timeframe - spike))

    bincount, bin_edges = np.histogram(total_spikes, bins)
    # norm_bincount = bincount / len(base_neuron_spikes) # normalization of the bin values so that they represent the probability of spikes in that bin

    # for spike in base_neuron_spikes:
    #     spikes_in_timeframe = neuron_to_correlate[np.logical_and(neuron_to_correlate > (spike - time_interval), neuron_to_correlate < (spike + time_interval))]
    #     bincount += np.histogram((spikes_in_timeframe-spike), num_bins)[0]
    # norm_bincount = bincount / len(base_neuron_spikes)

    # return norm_bincount, bin_edges, total_spikes
    return bincount, bin_edges, total_spikes

def calcBinCenters(bin_edges):
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
    return bin_centers


def plotCorrelationBehaviour(corr_data, bin_edges, base_neuron, ntc):
    c = corr_data
    b = bin_edges
    b_c = calcBinCenters(b)
    fig, ax = plt.subplots()
    y_smooth = signal.savgol_filter(c, window_length=200, polyorder=5, mode="nearest")
    plt.vlines(0, ymin=0, ymax=np.max(c)/2, linewidth=0.5, linestyle='dashed', color='black')
    plt.plot(b_c, y_smooth, linewidth=1, linestyle='-', color='green')
    plt.xlabel('time interval(seconds)')
    plt.ylabel('spike probability in time interval')
    plt.title(f"neurons {base_neuron} with {ntc} correlation behaviour")
    plt.show()

def plotCorrelatedData(corr_data, bin_edges, base_neuron, ntc):
    fig1, ax1 = plt.subplots()
    bin_centers = calcBinCenters(bin_edges)
    ax1.bar(bin_centers, corr_data, width=0.001, facecolor='black', edgecolor='black', linewidth=0.2, align='center')
    # y_smooth = signal.savgol_filter(corr_data, window_length=100, polyorder=7, mode="nearest")
    # ax1.plot(bin_centers, y_smooth, linewidth=1, linestyle='-', color='green')
    plt.xlabel('time interval(seconds)')
    plt.ylabel('spike count in time interval')
    plt.title(f"neuron {base_neuron} correlation with neuron {ntc}")
    # plotCorrelationBehaviour(corr_data, bin_edges, base_neuron, ntc)
    return fig1


def crossCorrelateData(path):
    spike_data = getSpikeData(path)
    spike_sec = spike_data["spike_sec"]
    spike_clusters = spike_data["spike_clusters"]
    good_clusters = spike_data["good_cluster_ID"]
    # good_cluster_2 = spike_data["good_cluster_ID"][0:40]
    correlation_data = dict()
    correlation_data["norm_counts"] = []

    for neuron in good_clusters:
        base_neuron_spikes = getNeuronSpikes(neuron, spike_sec, spike_clusters)
        good_clusters_2 = good_clusters[good_clusters != neuron]
        print(f"correlating for neuron{neuron}:")
        for ntc in good_clusters_2:
            ntc_spikes = getNeuronSpikes(ntc, spike_sec, spike_clusters)
            c, b, s = getCorrelatedData(base_neuron_spikes, ntc_spikes, 0.05)
            correlation_data["norm_counts"].append(c)

    return correlation_data

    # 1. run a loop through all neurons. for every neuron get its spikes and correlate with all other neuron spikes
    # 2. put the bincount data in a new array
    # 3. this array is a matrix so that for every iteration a new row is filled with corr_datas (arrays)?
    # 4.
    # return


if __name__=='__main__':
    spike_data = getSpikeData(path)
    spike_sec = spike_data["spike_sec"]
    spike_clusters = spike_data["spike_clusters"]
    good_clusters = spike_data["good_cluster_ID"]
    # base_neuron = spike_data["good_cluster_ID"][0]
    # base_neuron_spikes = getNeuronSpikes(base_neuron, spike_sec, spike_clusters)

# check for good neurons:
#     big_clusters = []
#     for neuron in good_clusters:
#         neuron_spikes = spike_sec[spike_clusters == neuron]
#         if len(neuron_spikes) > 8000:
#             big_clusters.append(neuron)
#     print(big_clusters)
# test for crosscorrelation data
#     good_cluster_2 = spike_data["good_cluster_ID"][0:50]

    # #
    # correlation_Data = crossCorrelateData(path)
    # norm_bincounts = crossCorrelateData(path)[0]
    # bin_sizes = crossCorrelateData(path)[1]
    # total_spikes = crossCorrelateData(path)[2]
    # for counts in correlation_Data["norm_counts"]:
    #     corr_str = checkThreshold(counts, 0.01, 100)
    #     if int(corr_str) > 500:
    #         print(corr_str)
    #         plotCorrelatedData(c, b, neuron, ntc)




    # example for plotCorrelatedBehaviour:
    # test_neuron_spikes = []
    # test_neuron_spikes.append(getNeuronSpikes(spike_data["good_cluster_ID"][8], spike_sec, spike_clusters))
    # test_neuron_spikes.append(getNeuronSpikes(16, spike_sec, spike_clusters))
    # plotCorrelationBehaviour(base_neuron_spikes, test_neuron_spikes)
    #
    #
    # # example for several neurons in correlation to base neuron - getCorrelatedData, and plotting them
    # corr_data = []
    # bins_edges = []
    # total_spikes = []
    # fig, axs = plt.subplots(3, 2)
    # k = 0
    # j = 0
    # for i, cluster in enumerate(good_clusters[1:7]):
    #     neuron_spikes = getNeuronSpikes(cluster, spike_sec, spike_clusters)
    #     c, b, s = getCorrelatedData(base_neuron_spikes, neuron_spikes, 1)
    #     axs[k, j].bar(b[:-1], c, width=0.001, edgecolor='white', linewidth=0.1, align='edge')
    #     axs[k, j].set_title(f"with neuron{cluster}:")
    #     j = j + 1
    #     if j == 2:
    #         k = k + 1
    #         j = 0
    # fig.suptitle(f"neuron{base_neuron} correlation:")
    # plt.show()
    # #

# general example for use in getNeuronSpikes and getCorrelatedData and then plotting it:
# the function getCorrelatedData returns bin count, bin edges and center, and the array of total spikes(not normalized to density)
    neuron_164 = getNeuronSpikes(164, spike_sec, spike_clusters)
    neuron_163 = getNeuronSpikes(163, spike_sec, spike_clusters)
    corr_data, bin_edges, total_spikes = getCorrelatedData(neuron_164, neuron_163, 0.05)
    # print("neuron 163 legth:", len(neuron_163), "neuron 164 length:", len(neuron_164))
#     # print([neuron_15, neuron_16])
#     print("length:", len(corr_data), "\ntype:", type(corr_data), "\ndata:", corr_data, "\ngreater than 0:", len(corr_data[corr_data>0]))
#     print(f"total spikes: {len(total_spikes)}")
    plotCorrelatedData(corr_data, bin_edges, 164, 163)
    plt.show()
    # fig, ax = plt.subplots()
    # bin_centers = calcBinCenters(bin_edges)
    # ax.bar(bin_centers, corr_data, width=0.001, edgecolor='white', linewidth=0.1, align='center')
    # n, bins, patches = plt.hist(total_spikes, density=True, bins=4000, edgecolor="white", linewidth=0.1)
    # plt.hist(corr_data, bins=bins_center, edgecolor='silver')
    # plotCorrelationBehaviour(neuron_50, neuron_51, 50, 51)
    # plt.show()




# 1. check normalization