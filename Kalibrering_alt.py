import numpy as np
import pandas as pd
import glob
import os
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

folder = 'kalibrering'
seperator = ' '
file_type = '*.txt'



read_files = glob.glob(os.path.join(folder, file_type))

#Creates a list of all file names
np_array_values = []
for files in read_files:
    pdfile = pd.read_csv(files, sep=seperator, skiprows=3)           #Specify seperator
    np_array_values.append(pdfile)

#print(np_array_values[0])

def getCounts(name: str, lc: int = 20, hc: int = 6000):
    data = np.loadtxt("kali/Kali" + name + "_ch000.txt", skiprows=5)
    counts = data[:, 1]
    (x, y) = np.unique(counts, return_counts=True)
    lI = np.where(x >= lc)[0][0]
    hI = np.where(x >= hc)[0][0]
    x = x[lI:hI]
    y = y[lI:hI]
    plt.plot(x, y)
    plt.title(name)
    #plt.show()
    return (x, y)


def gaussFit(x, mu, sig, a, b, c):
    lny = np.log(a) - ((x-mu)**2)/(2*sig**2)
    return np.exp(lny) - (b*x+c)


def getChannel(name: str, data: tuple, lower_limit: int, upper_limit: int, guess: [int, int, int], guess2 = [0,0]):
    x = data[0][lower_limit:upper_limit]
    y = data[1][lower_limit:upper_limit]
    yler = np.sqrt(y)
    pinit = guess + guess2
    xhelp = np.linspace(lower_limit, upper_limit, 500)
    popt, pcov = curve_fit(gaussFit, x, y, p0=pinit, sigma=yler, absolute_sigma=True)
    print(name)
    print('mu :', popt[0])
    print('sigma :', popt[1])
    print('scaling', popt[2])
    print('background', popt[3], popt[4])
    perr = np.sqrt(np.diag(pcov))
    print('usikkerheder:', perr)
    chmin = np.sum(((y - gaussFit(x, *popt)) / yler) ** 2)
    print('chi2:', chmin, ' ---> p:', ss.chi2.cdf(chmin, 4))

    plt.plot(x, y, color="r", label="data")
    plt.plot(xhelp, gaussFit(xhelp, *popt), 'k-.', label="gaussfit")
    plt.legend()

    plt.title(name)
    plt.show()

    return [popt, perr]


Cs = getCounts("Cs")
Co = getCounts("Co")
Ra = getCounts("Ra")

chs = []
chs += [getChannel("Cs E=661", Cs, 700, 2000, [900, 10, 200])]
chs += [getChannel("Co E=1173", Co, 1400, 1700, [1580, 10, 200])]
chs += [getChannel("Co E=1332", Co, 1700, 1900, [1800, 10, 200])]
chs += [getChannel("Ra E=186", Ra, 150, 350, [250, 10, 1000])]
chs += [getChannel("Ra E=241", Ra, 225, 425, [325, 10, 1000])]
chs += [getChannel("Ra E=295", Ra, 280, 480, [390, 10, 8000])]
chs += [getChannel("Ra E=351", Ra, 370, 570, [470, 10, 8000])]
chs += [getChannel("Ra E=609", Ra, 720, 920, [820, 10, 8000])]
chs += [getChannel("Ra E=768", Ra, 940, 1140, [1040, 10, 1000])]
chs += [getChannel("Ra E=934", Ra, 1100, 1400, [1260, 10, 500])]
chs += [getChannel("Ra E=1120", Ra, 1400, 1600, [1500, 10, 1000])]
chs += [getChannel("Ra E=1238", Ra, 1550, 1800, [1700, 10, 200])]
chs += [getChannel("Ra E=1377", Ra, 1760, 1960, [1860, 10, 200])]
chs += [getChannel("Ra E=1407", Ra, 1800, 2000, [1900, 10, 200])]
chs += [getChannel("Ra E=1729", Ra, 2230, 2430, [2330, 10, 200])]
chs += [getChannel("Ra E=1764", Ra, 2290, 2490, [2390, 10, 200])]
chs += [getChannel("Ra E=1847", Ra, 2430, 2630, [2500, 10, 100])]
chs += [getChannel("Ra E=2118", Ra, 2700, 3100, [2880, 10, 200])]
chs += [getChannel("Ra E=2204", Ra, 2890, 3190, [2990, 10, 200])]
chs += [getChannel("Ra E=2447", Ra, 3200, 3400, [3300, 10, 200])]
#getChannel("Ra E=768", Ra, 280, 480, [390, 10, 8000])

x = np.array([])
xler = np.array([])
for i in chs:
    x = np.append(x, [i[0][0]])
    xler = np.append(xler, [i[1][0]])

y = [661.657, 1173.228, 1332.492, 186.211, 241.997, 295.224, 351.932, 609.312, 768.356, 934.061, 1120.287, 1238.110, 1377.669, 1407.98, 1729.595, 1764.494, 1847.420, 2118.55, 2204.21, 2447.86] #KeV
yler = [0.003, 0.003, 0.003, 0.013, 0.003, 0.002, 0.002, 0.007, 0.01, 0.012, 0.01, 0.012, 0.012, 0.04, 0.015, 0.014, 0.025, 0.003, 0.004, 0.01]
#xler =[10, 10, 10]
def funlin(x, a, b):
    return a*x+b
#yler = np.sqrt(y)
pinit = [1,1]
xhelp = np.linspace(0, 3400, 500)
popt, pcov = curve_fit(funlin, x, y, p0=pinit, sigma=yler, absolute_sigma=True)
print("\n energy fit")
print('a hældning:', popt[0])
print('b forskydning:', popt[1])
perr = np.sqrt(np.diag(pcov))
print('usikkerheder:', perr, "\n")

#print(x)
chmin = np.sum(((y - funlin(x, *popt)) / yler) ** 2)
print('chi2:', chmin, ' ---> p:', ss.chi2.cdf(chmin, 4))
plt.scatter(x, y, label="data")
plt.plot(xhelp, funlin(xhelp, *popt), label="fit")
plt.legend()
#plt.show()

print("xler: ", xler)











