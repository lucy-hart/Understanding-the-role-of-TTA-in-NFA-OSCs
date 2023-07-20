#%% import stuff
import os
import numpy as np
from scipy import interpolate as intp
import matplotlib.pyplot as plt
import pandas as pd

#%% 
#Read in data
dir = os.getcwd()
YSeries = 11

if YSeries == 11:
    fluences = ['3p2', '6p4', '13', '26', '51']
    labels = ['3.2', '6,4', '13', '26', '51']
    trip_lim = [1425, 1435]
elif YSeries == 6:
    fluences = ['3p8', '7p6', '15', '31', '61']
    labels = ['3.8', '7.6', '15', '31', '61']
    trip_lim = [1465, 1475]

filename = dir + '\data\FullTAS_Y{}_IR_AllFluences.xlsx'.format(YSeries)

times = {}
triplet_mean = {}

for i in range(len(fluences)):   

    TASData = pd.read_excel(filename, sheet_name = fluences[i], header = 0, index_col = 0, dtype = np.float64)
    time = np.array([i for i in TASData.columns if i >= 1])
    TASData = TASData[time]
    wl = TASData.index[:]
    times[i] = time

    #Calculate sigma for the triplet region 
    #axis 0 along rows ad axis 1 along columns
    argmin = np.argmin(abs(wl-trip_lim[0]))
    argmax = np.argmin(abs(wl-trip_lim[1]))

    triplet_mean[i] = TASData.iloc[argmin:argmax].mean(axis = 0).to_numpy()
    print(len(triplet_mean[i]))

#%%
#Interploate data so they all have the same time values for the fitting program
triplet_mean_interp = {}
lengths = np.zeros(len(fluences))
#keep time vaues for sample which has most datapoints
for i in range(len(fluences)):
    lengths[i] = len(times[i])
argkeep = np.argmax(lengths)

for i in range(len(fluences)):
    if i == argkeep:
        triplet_mean_interp[i] = triplet_mean[i]
    else:
        func = intp.interp1d(times[i], triplet_mean[i], fill_value = 'extrapolate')
        triplet_mean_interp[i] = func(times[argkeep])

#%% 
#Plot interpolation and raw data to make sure it looks ok
fig, ax = plt.subplots(2,2, facecolor='white', figsize = (15,15))
k = 0
for i in range(len(fluences)):
    if i != argkeep:
        if k <= 1:
            j = 0
        else:
            j = 1
        ax[j,k%2].semilogx(times[i], triplet_mean[i], color='cornflowerblue', label = 'data', alpha = 0.5)
        ax[j,k%2].semilogx(times[argkeep], triplet_mean_interp[i], color='midnightblue', label = 'interp', linestyle = '--')
        k += 1 
ax[1,1].legend()

plt.show()

# %%
#Save interpolated values as a .txt file 
ar = np.zeros((len(triplet_mean_interp[0][:]), len(fluences)+1))
ar[:,0] = times[argkeep]
for i in range(len(fluences)):
    ar[:,i+1] = triplet_mean_interp[i][:]
filename_save = dir + '\data\Y{}Interp.txt'.format(YSeries)
np.savetxt(filename_save, ar)

# %%
