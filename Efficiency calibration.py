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
plt.plot(data[:,0], data[:,1])
plt.show()