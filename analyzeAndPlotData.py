import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import csv


path = r'C:\Users\Yuval\Desktop\Yuval_project\data_analysis\ND7'
os.chdir(path)
W_file = "results_ND7_YB_border_93.csv"
W_df = pd.read_csv(W_file)
W_df.columns.values[0] = 'time'
n = 132
border = 93 # write here the border from file name
# print(W_df.to_string())
time = W_df.time.tolist()
# fig, ax = plt.subplots()
# 1
# ax.plot(time, W_df.num_con_diff, color='black', label='all')
# ax.plot(time, W_df.tot_con_diff_e, color='magenta', label='excitatory')
# ax.plot(time, [abs(val) for val in W_df.num_con_diff_i], color='cyan', label='inhibitory')
# plt.show()

# con_type = ['connection type: all', 'gc-gc', 's2-s2', 'gc<->s2']
# fig, axes = plt.subplots(2, 2)
# j = 1
# for i in range(4):
#     plt.subplot(2, 2, (i+1))
#     ind = j + i*6
#     plt.plot(time, W_df.iloc[:, ind], color='black', label='all')
#     plt.plot(time, W_df.iloc[:, (ind + 2)], color='magenta', label='excitatory')
#     plt.plot(time, W_df.iloc[:, ind+4], color='cyan', label='inhibitory')
#     plt.title(f"{con_type[i]}")
#     if i==0 or i==2:
#         plt.ylabel('Number of connections')
#     if i==2 or i==3:
#         plt.xlabel('time from start(hours)')
#     if i==3:
#         plt.legend(loc='upper right')
#     if i==0 or i==1:
#         plt.xticks(color='w')
#
# plt.suptitle("Number of connections by orientation(e/i) and location(GC/S2)")
# plt.savefig("Number of connections by orientation and location")
# plt.show()


fig, axes = plt.subplots(1, 2)
# ind = 1
axes[0].plot(time, [val/n for val in W_df.iloc[:, 1]], color='grey', label="all")
axes[0].plot(time, [val/border for val in W_df.iloc[:, 7]], color='green', label="gc")
axes[0].plot(time, [val/(n-border) for val in W_df.iloc[:, 13]], color='blue', label='s2')
# axes[0].plot(time, [val/(n-border) for val in W_df.iloc[:, 17]], color='cyan', label='s2 / E', ls='--')

axes[0].set_title("Normalized # Connections per location", fontsize=15, fontweight='bold')
axes[0].set_box_aspect(1)
axes[0].set_xlabel('time from start(hours)', fontsize=14)
labels = axes[0].get_yticks()
labels = [round(l, 1) for l in labels]
print(labels)
axes[0].set_yticklabels(labels, fontsize=13)
axes[0].set_xticklabels(time, fontsize=13)
axes[0].set_ylabel('Norm number of connections', fontsize=13)
axes[0].legend(loc='upper right')


# ind = 2
axes[1].plot(time, [val/border for val in W_df.iloc[:, 25]], color='grey', label='gc->s2')
axes[1].plot(time, [val/border for val in W_df.iloc[:, 27]], color='magenta', label='gc->s2/e')
axes[1].plot(time, [val/border for val in W_df.iloc[:, 29]], color='cyan', label='gc->s2/i')
axes[1].plot(time, [val/(n-border) for val in W_df.iloc[:, 31]], color='grey', label='s2->gc', ls='--')
axes[1].plot(time, [val/(n-border) for val in W_df.iloc[:, 33]], color='purple', label='s2->gc/e', ls='--')
axes[1].plot(time, [val/(n-border) for val in W_df.iloc[:, 35]], color='aqua', label='s2->gc/i', ls='--')

# axes[1].plot(time, [val/(n-border) for val in W_df.iloc[:, 17]], color='cyan', label='inhibitory')

# axes[1].plot(time, [abs(val) for val in W_df.iloc[:, ind+4]], color='cyan', label='inhibitory')
axes[1].set_title("Normalized GC-S2 Connections over time", fontsize=15, fontweight='bold')
# axes[1].set_title("Mean connectivity strength", fontsize=16, fontweight='bold')
axes[1].set_xlabel('time from start(hours)', fontsize=14)
labels = axes[1].get_yticks()
labels = list(np.unique([round(l, 1) for l in labels]))
print(labels)
axes[1].set_yticks(labels)
axes[1].set_yticklabels(labels, fontsize=13)
axes[1].set_xticklabels(time, fontsize=13)
axes[1].set_ylabel('Norm number of connections', fontsize=13)
axes[1].set_box_aspect(1)


axes[1].legend(loc='upper right')


plt.suptitle("Connectivity Data Plot", fontsize=20)
# plt.savefig("Connectivity data plot")
plt.show()



# W_files = os.listdir(f"{path}/W""
# print(W_files[0])
# W = pd.read_csv(f"{path}/W/{W_files[0]}", header=None)
# print(W)
# p_files = os.listdir(f"{path}/params")
# os.chdir(f"{path}/params")
# borders = []
# cell_lists = []
# params=[]
# with open(p_files[0], 'r') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         params.append(list(row))
# borders = [float(item) for item in params[0]]
# cell_lists = params[1]
# print("borders:", borders)
# print('cells:', cell_lists)


# grouped = con_df.groupby(['locs', 'con_type', 'time'], as_index=True)['mean'].sum()
# # grouped = con_df[np.logical_and(con_df['con_type']=='inhib', con_df['locs']=='GC')].groupby(['locs', 'con_type'])['mean'].sum()
# list = grouped.index[]
# # times = grouped.index.
# print(grouped.head(25),  "\n", list)
# print(times)

# locs = [np.unique(df['locs'].values)][0]
# print(locs)
# colors = {'both': 'blue', 'excitatory': 'green', 'inhibitory': 'red'}
# con_type = np.unique(df['con_type'].values)
# colors = ['blue', 'green', 'red']

# create a figure and axes
# fig, axs = plt.subplots(2, 2, sharex=True)
# # legend_ax = fig.add_axes([0.9, 1, 0.0, 0.0])
#
# #iterate over locs and make subfigure for each one
# for i, loc in enumerate(locs):
#
#     loc_df = df[df['locs'] == loc].reset_index(drop=True)
#     plt.subplot(2, 2, (i + 1))
#
#     # iterate over the connectivity types and plot a line for each one
#     for con_type, color in colors.items():
#
#         # filter the dataframe for the current connectivity type
#         df_filtered = loc_df[loc_df['con_type'] == con_type].reset_index(drop=True)
#         print(df_filtered)
#         print(df_filtered['time'].tolist())
#         # get the mean values for the current connectivity type
#         means = df_filtered['mean'].values
#         if means.size != 0 and means[0] < 0:
#             means = means*(-1)
#         print(means)
#         # plot the means on the y-axis and the time on the x-axis
#         plt.plot(df_filtered['time'].tolist(), means, label=con_type, color=color)
#
#     plt.title(f"{loc}")
#     # set the x-ticks and labels
#     xticks = np.arange(0, 8)
#     xticklabels = ['0-2', '2-4', '4-6', '6-8', '8-10', '10-12', '12-14', '14-16']
#     plt.xticks(xticks, xticklabels)
#
#
#     # set the y-label
#
#     if i==0 or i==2:
#         plt.ylabel('Mean connectivity')
#     if i==2 or i==3:
#         plt.xlabel('time from start(hours)')
#     if i==3:
#         plt.legend(loc='lower right')
#
# # save the figure:
# plt.suptitle(f"Mean connectivity over time in different connectivity groups:")
#
# plt.savefig('mean_connectivity_time_and_loc_YB')
# # show the plot
# plt.show()

