import numpy as np
import matplotlib.pyplot as plt

#Defines the file to be split and reads it
file = 'C:/Users/Erik/OneDrive/Skrivebord/Eksperimentiel 3 ø1/Mg_1_maalning_dag2_ch000.txt'
#skiprows may have to be varied
dat = np.loadtxt(file, skiprows=4)
print('loaded')

#
def getCounts(data, lc: int = 1, hc: int = 3500):
    #data = np.loadtxt("kali/Kali" + name + "_ch000.txt", skiprows=5)
    counts = data[:, 1]
    dat = np.unique(counts, return_counts=True)
    lI = np.where(dat[0] <= lc)
    hI = np.where(dat[0] >= hc)
    #print(lI)
    #dat = np.delete(dat, lI, axis=1)

    plt.plot(dat[0], dat[1])
    #plt.title(name)
    plt.show()
    return dat

#define an interval and split the file into said interval
resolution = 10**(-8)
timeInterval = 10*60/resolution

#finding indexes
indexes = [0]
for i in range(len(dat)):
    i = int(i)
    if dat[i][0] >= dat[indexes[-1]][0]+timeInterval:
        indexes += [i]

intervals = []
for i in range(len(indexes)-1):
    i = int(i)
    intervals += [[indexes[i], indexes[i+1]]]

dats = []
for i in intervals:
    dats += [dat[i[0]:i[1]]]

for i in range(len(dats)):
    i = int(i)
    np.savetxt(f'Dag 2/Split Mg data/manganese1 counts {i}.txt', getCounts(dats[i]), delimiter=" ", fmt='%s')

