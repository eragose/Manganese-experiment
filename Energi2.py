import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


data = np.loadtxt('Dag 2/manganese1 all counts.txt')
def chtoen(data):
    a = 0.73971712
    ae = 9.9e-7
    b = 0.3785
    be = 1.4e-3
    newdat = np.array([data[0]*a+b, data[1]])
    errEnergy = np.array([((ae*data[0])**2+be**2)])
    return newdat, errEnergy


data, errEnergy = chtoen(data)
data = np.transpose(data)


Line = [3445.279, 3369.91, 3122.908, 2959.935, 2657.547, 2085.064, 846.77]
line = [2598.438, 3369.81, 2523.06, 1037.833, 2113.092, 1810.726, 1238.27]

plt.vlines(Line, 0, 10000, colors='r', linestyles='dashed')
plt.vlines(line, 0, 10000, colors='b', linestyles='dashed')

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


