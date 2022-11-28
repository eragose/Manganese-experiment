import numpy as np
import glob as glob
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

skip = 5               #For filer i Xray mappe brug 38 ellers 12
stop = 4108             #For filer i Xray mappe brug 4134 eller 4108
folder = 'Kali'

names = glob.glob(os.path.join(folder,'*.txt'))
files = []
for i in names[0:]:
    k = np.loadtxt(i, skiprows=skip)
    
    files.append(k)

coenergy = [1173.228, 1332.492]
csenergy = [661.657]
raenergy = [186.211, 241.997, 295.224, 351.932, 609.312, 768.356, 934.061, 1120.287, 1238.11, 1377.669, 1729.59, 1764.49, 1847.42, 2118.55, 2204.21, 2447.86]

ch_count = []
for i in files:
    channel_count = np.unique(i[:,1], return_counts=True)
    ch_count.append(channel_count)
print(ch_count[0][0], ch_count[0][1])

def plot_data(data):
    n = len(data)
    max_id = 3500
    fig, ax = plt.subplots(n)
    for i in range(n):
        x, y = data[i][0][:], data[i][1][:]
        x1 = convert(x)
        #print(y)
        peak = peaks(y[:max_id], 50)#max(y[:max_id])/30)
        print(peak)
        ax[i].scatter(convert(peak),y[peak], color='red')
        ax[i].plot(x1,y, label=names[i])
        ax[i].legend()
        ax[i].set_xlim(-100, max_id)
        ax[i].set_ylim(0, max(y[:max_id])+500)

def peaks(data, height):
    peak, prop = find_peaks(data, prominence=height)
    return peak




prom = [0, 100, 0]     #0 betyder den ikke er bestemt

plot_data(ch_count)
print(ch_count[0][1][4094])
plt.show()