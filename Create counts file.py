import numpy as np
import matplotlib.pyplot as plt

folderin = "C:/Users/eriko/Desktop´/Eksperimentiel 3 ø1/"
folderout = "Kali/"


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



ra = np.loadtxt("C:/Users/eriko/Desktop/Eksperimentiel 3 ø1/ra_cali2_4.8_ch000.txt", skiprows=4)
print("loaded")
np.savetxt(folderout+'ra_4.8_counts.txt', getCounts(ra), delimiter=" ", fmt='%s')