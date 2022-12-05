import numpy as np
import pandas as pd
import glob
import os
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from zipfile import ZipFile



#specifying the zip file name
#for i in range(5):
#    file_name = f'Dag 2/Mg_data/Mg_1_maalning_dag2_ch000{i}.zip'

    # opening the zip file in READ mode
#    with ZipFile(file_name, 'r') as zip:
        # printing all the contents of the zip file
#        zip.printdir()

        # extracting all the files
#        zip.extractall()

a = 0.73971712
ae = 9.9e-7
b = 0.3785
be = 1.4e-3

#data = []
#for i in range(5):
#    data.append(np.loadtxt(f'Mg_1_maalning_dag2_ch000{i}.txt')[1])

def getCounts(i, lc: int = 20, hc: int = 6000):
    print('File:', str(i))
    data = np.loadtxt(f"Mg_1_maalning_dag2_ch000{i}.txt")
    counts = data[:, 1]
    (x, y) = np.unique(counts, return_counts=True)
    lI = np.where(x >= lc)[0][0]
    #print(lI)
    hI = np.where(x >= hc)[0][0]
    x = x[lI:hI]
    y = y[lI:hI]
    
    a = 0.73971712
    ae = 9.9e-7
    b = 0.3785
    be = 1.4e-3
    x  = x * a + b
    
    return (x, y)

Line = [3445.279,3369.91,3122.908,2959.935,2657.547,2085.064,846.77]
line = [2598.438, 3369.81,2523.06,1037.833,2113.092,1810.726, 1238.27]


plt.vlines(Line, 0, 10000, colors='r', linestyles='dashed')
plt.vlines(line, 0, 10000, colors='b', linestyles='dashed')



Mg1 = np.array(getCounts(0))
Mg2 = np.array(getCounts(1))
Mg3 = np.array(getCounts(2))
Mg4 = np.array(getCounts(3))
Mg5 = np.array(getCounts(4))


Mg = Mg1[1] + Mg2[1] + Mg3[1] + Mg4[1] + Mg5[1]
print(Mg[0])

plt.plot(Mg[0][800:], Mg[1][800:])
plt.show()

plt.vlines(Line, 0, 10000, colors='r', linestyles='dashed')
plt.vlines(line, 0, 10000, colors='b', linestyles='dashed')

plt.plot(Mg1[0], Mg1[1])
plt.title('Mg')

plt.show()




def gaussFit(x, mu, sig, a, b, c):
    lny = np.log(a) - ((x-mu)**2)/(2*sig**2)
    return np.exp(lny) - (b*x+c)


def getChannel(name: str, data: tuple, lower_limit: int, upper_limit: int, guess: [int, int, int], guess2 = [0,0]):
    ll = np.where(data[:, 0] > lower_limit)[0][0]
    ul = np.where(data[:, 0] > upper_limit)[0][0]
    x = data[:, 0][ll:ul]
    y = data[:, 1][ll:ul]
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

# def t(x):
    # return int(x*a + b)
    # 
chs = []
chs += [getChannel("Mg E=846.77", Mg1,  700, 900, [850, 10, 200])]
# chs += [getChannel("Mg E=2085.0", Mg, 1400, 1700, [1580, 10, 200])]
# chs += [getChannel("Mg E=1238.27", Mg, 1700, 1900, [1800, 10, 200])]
# chs += [getChannel("Mg E=2657.54", Mg, 150, 350, [250, 10, 1000])]
# chs += [getChannel("Mg E=1810.72", Mg, 225, 425, [325, 10, 1000])]
# chs += [getChannel("¨Mg E=2959.03", Mg, 280, 480, [390, 10, 8000])]
# chs += [getChannel("Ra E=3122.908", Mg, 370, 570, [470, 10, 8000])]
# chs += [getChannel("Ra E=2113.09", Mg, 720, 920, [820, 10, 8000])]
# chs += [getChannel("Ra E=1037.833", Mg, 940, 1140, [1040, 10, 1000])]
# chs += [getChannel("Ra E=3369.9", Mg, 1100, 1400, [1260, 10, 500])]
# chs += [getChannel("Ra E=2523.06", Mg, 1400, 1600, [1500, 10, 1000])]
# chs += [getChannel("Ra E=2598.43", Mg, 1550, 1800, [1700, 10, 200])]
#


# chs += [getChannel("Ra E=1377", Ra, 1800, 2000, [1900, 10, 200])]
# chs += [getChannel("Ra E=1729", Ra, 2230, 2430, [2330, 10, 200])]
# chs += [getChannel("Ra E=1764", Ra, 2290, 2490, [2390, 10, 200])]
# chs += [getChannel("Ra E=1847", Ra, 2430, 2630, [2500, 10, 100])]
# chs += [getChannel("Ra E=2118", Ra, 2700, 3100, [2880, 10, 200])]
# chs += [getChannel("Ra E=2204", Ra, 2890, 3190, [2990, 10, 200])]
# chs += [getChannel("Ra E=2447", Ra, 3200, 3400, [3300, 10, 200])]
# #getChannel("Ra E=768", Ra, 280, 480, [390, 10, 8000])

x = np.array([])
xler = np.array([])
for i in chs:
    x = np.append(x, [i[0][0]])
    xler = np.append(xler, [i[1][0]])

y = [661.657, 1173.228, 1332.492, 186.211, 241.997, 295.224, 351.932, 609.312, 768.356, 934.061, 1120.287, 1238.110, 1377.669, 1407.98, 1729.595, 1764.494, 1847.420, 2118.55, 2204.21, 2447.86] #KeV
yler = [0.003, 0.003, 0.003, 0.013, 0.003, 0.002, 0.002, 0.007, 0.01, 0.012, 0.01, 0.012, 0.012, 0.04, 0.015, 0.014, 0.025, 0.003, 0.004, 0.01]
#xler =[10, 10, 10]