import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.stats as ss

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


def gaussFit(x, mu, sig, a, b, c):
    lny = np.log(a) - ((x-mu)**2)/(2*sig**2)
    return np.exp(lny) - (b*x+c)


def getChannel(name: str, data: tuple, lower_limit: int, upper_limit: int, guess: [int, int, int], guess2=[0.0,0.0]):
    ll = np.where(data[:,0]>lower_limit)[0][0]
    ul = np.where(data[:,0]>upper_limit)[0][0]
    x = data[:,0][ll:ul]
    y = data[:,1][ll:ul]
    yler = np.sqrt(y)
    pinit = guess + guess2
    xhelp = np.linspace(lower_limit, upper_limit, 500)
    popt, pcov = curve_fit(gaussFit, x, y, p0=pinit, sigma=yler, absolute_sigma=True, bounds=([0, 0, 0, -np.inf, -np.inf], np.inf))
    print('\n',name)
    print('mu :', popt[0])
    print('sigma :', popt[1])
    print('scaling', popt[2])
    print('background', popt[3], popt[4])
    perr = np.sqrt(np.diag(pcov))
    print('usikkerheder:', perr)
    chmin = np.sum(((y - gaussFit(x, *popt)) / yler) ** 2)
    print('chi2:', chmin, ' ---> p:', ss.chi2.cdf(chmin, 4), '\n')

    #plt.plot(x, y, color="r", label="data")
    #plt.plot(xhelp, gaussFit(xhelp, *popt), 'k-.', label="gaussfit")
    #plt.legend()

    #plt.title(name)
    #plt.show()

    return [popt, perr]

data = np.transpose(data)
data, Ee = converttoenergy(data)
print(data)
data = data[np.where(data[:,0]>1)]
data = data[np.where(data[:,0]<5000)]
#plt.plot(data[:,0], data[:,1])
#plt.show()

chs = []
chs += [getChannel("Ra E=186", data,186-100, 186+100, [186, 2, 35000])]
chs += [getChannel("Ra E=241", data,241-100, 241+100, [241, 10, 60000])]
chs += [getChannel("Ra E=295", data, 295-100, 295+100, [295, 10, 100000])]
chs += [getChannel("Ra E=351", data, 351-100, 351+100, [351, 10, 150000])]
chs += [getChannel("Ra E=609", data, 609-100, 609+100, [609, 4, 7000])]
chs += [getChannel("Ra E=768", data, 768-100, 768+100, [768, 4, 1000])]
chs += [getChannel("Ra E=934", data, 934-100, 934+100, [934, 5, 1200])]
chs += [getChannel("Ra E=1120", data, 1120-100, 1120+100, [1120, 5, 1000])]
chs += [getChannel("Ra E=1238", data, 1238-100, 1238+100, [1238, 6.1, 310])]
chs += [getChannel("Ra E=1377", data, 1377-100, 1377+100, [1377, 7.16413, 3500], [-0.0539055, -8.036607])]
chs += [getChannel("Ra E=1764", data, 1764-100, 1764+100, [1764, 7.8658, 489], [0.09, -248])]
chs += [getChannel("Ra E=2204", data, 2204-100, 2204+100, [2204, 10, 100])]
chs += [getChannel("Ra E=2447", data, 2447-100, 2447+100, [2447, 11, 20])]
chs = np.array(chs)

abundances = np.array([3.95, 7.43, 19.3, 37.6, 46.1, 4.94, 3.03, 15.1, 5.79, 4.00, 15.4, 5.08, 1.57])
errAbund = np.array([0.06, 0.11, 0.2, 0.4, 0.5, 0.06, 0.04, 0.2, 0.08, 0.06, 0.2, 0.04, 0.2])
t1 = 776
t2 = 1.37463282e+11
time = (t2-t1)*10**(-8) #seconds
sourceactivity = 4.8 #micro Ci
rad = 2.3 #cm
radErr = 0.3 #+-cm
detRad = 7.8 #cm
detRadErr = 0.5 #+-cm

expectations = abundances * 3/4 * detRad**2/rad**2 * sourceactivity
expectationsUpper = (abundances+errAbund) * 3/4 * (detRad+detRadErr)**2/(rad-radErr)**2 * sourceactivity
expectationsLower = (abundances-errAbund) * 3/4 * (detRad-detRadErr)**2/(rad+radErr)**2 * sourceactivity
measured = chs[:, 0][:, 1] * chs[:, 0][:, 2] * np.sqrt(2*np.pi)/time
plt.scatter(chs[:,0][:,0], measured/expectations, label='mean efficiency')
plt.scatter(chs[:,0][:,0], measured/expectationsUpper, label='lower efficiency')
plt.scatter(chs[:,0][:,0], measured/expectationsLower, label='upper efficiency')
#plt.scatter(chs[:,1][:,0], expectations, label='expected')
plt.legend()
plt.show()


