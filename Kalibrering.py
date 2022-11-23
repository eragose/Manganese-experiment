import numpy as np
import glob as glob
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

skip = 5               #For filer i Xray mappe brug 38 ellers 12
stop = 4108             #For filer i Xray mappe brug 4134 eller 4108
folder = 'Kali'

#plt.rcParams['figure.format'] = 'svg'

names = glob.glob(os.path.join(folder,'*.txt'))
files = []
for i in names[0:]:
    k = np.loadtxt(i, skiprows=skip)
    
    files.append(k)


def plot_data(data):
    n = len(data)
    max_id = 1500
    fig, ax = plt.subplots(n)
    for i in range(n):
        x, y = np.unique(data[i][:,1], return_counts=True)
        #print(y)
        peak = peaks(y[:max_id], max(y[:max_id])/10)
        print(peak)
        ax[i].scatter(peak,y[peak], color='red')
        ax[i].plot(x,y, label=names[i])
        ax[i].legend()
        ax[i].set_xlim(-100, max_id)
        ax[i].set_ylim(0, max(y[:max_id])+500)

def peaks(data, height):
    peak, prop = find_peaks(data, height=height)
    return peak

#plot_data(files)
plt.show(format='svg')