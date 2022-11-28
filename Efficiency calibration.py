import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

folder = 'Kali/'
seperator = ' '
file_type = '.txt'

data = np.loadtxt(folder + "ra_4.8_counts.txt")
#data = data[1:]


def converttoenergy(dat):
    a = 0.73971712
    ae = 9.9e-7
    b = 0.3785
    be = 1.4e-3
    #chs1 = a*chs+b
    newDat = np.array([])
    for j in range(len(dat[:,0])):
        j = (int(j))
        if j==0:
            newDat = np.array([[a*dat[j, 0]+b, dat[j, 1]]])
        else:
            newDat = np.append(newDat, [[a*dat[j, 0]+b, dat[j, 1]]], axis=0)

    Ee = np.sqrt((ae*dat[:,0])**2 + be**2)
    return newDat, Ee

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

data = np.transpose(data)
data, Ee = converttoenergy(data)
data = data[np.where(data[:,0]>1)]
data = data[np.where(data[:,0]<5000)]
#plt.plot(data[:,0], data[:,1])
#plt.show()

chs = []
chs += [getChannel("Ra E=186", data,186-100, 186+100, [186, 10, 1000])]
chs += [getChannel("Ra E=241", data,241-100, 241+100, [241, 10, 1000])]
chs += [getChannel("Ra E=295", data, 295-100, 295+100, [295, 10, 8000])]
chs += [getChannel("Ra E=351", data, 351-100, 351+100, [351, 10, 8000])]
chs += [getChannel("Ra E=609", data, 609-100, 609+100, [609, 10, 8000])]
chs += [getChannel("Ra E=768", data, 768-100, 768+100, [768, 10, 1000])]
chs += [getChannel("Ra E=934", data, 934-100, 934+100, [934, 10, 500])]
chs += [getChannel("Ra E=1120", data, 1120-100, 1120+100, [1120, 10, 1000])]
chs += [getChannel("Ra E=1238", data, 1238-100, 1238+100, [1238, 10, 200])]
chs += [getChannel("Ra E=1377", data, 1337-100, 1337+100, [1377, 10, 200])]
chs += [getChannel("Ra E=1407", data, 1407-100, 1407+100, [1407, 10, 200])]
chs += [getChannel("Ra E=1729", data, 1729-100, 1729+100, [1729, 10, 200])]
chs += [getChannel("Ra E=1764", data, 1764-100, 1764+100, [1764, 10, 200])]
chs += [getChannel("Ra E=1847", data, 1847-100, 1847+100, [1847, 10, 100])]
chs += [getChannel("Ra E=2118", data, 2118-100, 2118+100, [2118, 10, 200])]
chs += [getChannel("Ra E=2204", data, 2204-100, 2204+100, [2204, 10, 200])]
chs += [getChannel("Ra E=2447", data, 2447-100, 2447+100, [2447, 10, 200])]