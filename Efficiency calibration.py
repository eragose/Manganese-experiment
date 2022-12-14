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
    popt, pcov = curve_fit(gaussFit, x, y, p0=pinit, sigma=yler, absolute_sigma=True, bounds=([0,0,0,-np.inf, -np.inf],np.inf))
    print('\n',name)
    print('mu :', popt[0])
    print('sigma :', popt[1])
    print('scaling', popt[2])
    print('background', popt[3], popt[4])
    perr = np.sqrt(np.diag(pcov))
    print('usikkerheder:', perr)
    chmin = np.sum(((y - gaussFit(x, *popt)) / yler) ** 2)
    print('chi2:', chmin, ' ---> p:', ss.chi2.cdf(chmin, 4), '\n')
    # plt.plot(x, y, color="r", label="data")
    # plt.plot(xhelp, gaussFit(xhelp, *popt), 'k-.', label="gaussfit")
    # plt.legend()
    # plt.xlabel('Energy (keV)')
    # plt.ylabel('Counts')
    # plt.title(name)
    # plt.show()

    return [popt, perr]


data = np.transpose(data)
data, Ee = converttoenergy(data)
#print(data)
data = data[np.where(data[:,0]>1)]
data = data[np.where(data[:,0]<5000)]
#plt.plot(data[:,0], data[:,1])
#plt.show()

chs = []
chs += [getChannel("Ra E=186", data,186-20, 186+20, [186, 2, 35000])]
chs += [getChannel("Ra E=241", data,241-50, 241+50, [241, 10, 60000])]
chs += [getChannel("Ra E=295", data, 295-40, 295+40, [295, 10, 100000])]
chs += [getChannel("Ra E=351", data, 351-40, 351+40, [351, 10, 150000])]
chs += [getChannel("Ra E=609", data, 609-50, 609+50, [609, 4, 7000])]
chs += [getChannel("Ra E=768", data, 768-100, 768+100, [768, 4, 1000])]
chs += [getChannel("Ra E=934", data, 934-100, 934+100, [934, 5, 1200])]
chs += [getChannel("Ra E=1120", data, 1120-100, 1120+100, [1120, 5, 1000])]
chs += [getChannel("Ra E=1238", data, 1238-100, 1238+100, [1238, 6.1, 310])]
chs += [getChannel("Ra E=1377", data, 1377-100, 1377+100, [1377, 7.16413, 3500], [-0.0539055, -8.036607])]
chs += [getChannel("Ra E=1764", data, 1764-60, 1764+60, [1764, 7.8658, 7000], [0.09, -248])]
chs += [getChannel("Ra E=2204", data, 2204-50, 2204+50, [2204, 10, 100])]
chs += [getChannel("Ra E=2447", data, 2447-100, 2447+100, [2447, 11, 20])]
chs = np.array(chs)

abundances = np.array([3.59, 7.43, 19.3, 37.6, 46.1, 4.94, 3.03, 15.1, 5.79, 4.00, 15.4, 5.08, 1.57])/100
errAbund = np.array([0.06, 0.11, 0.2, 0.4, 0.5, 0.06, 0.04, 0.2, 0.08, 0.06, 0.2, 0.04, 0.2])/100
t1 = 776
t2 = 1.37463282e+11
time = (t2-t1)*10**(-8) #seconds
sourceactivitycurie = 4.8 #micro Ci
sourceactivitycurieErr = 0.05
#sourceativty in counts
sourceactivity = sourceactivitycurie*3.7*10**4
sourceactivityErr = sourceactivitycurieErr*3.7*10**4
radc = 2.3 #cm
radcErr = 0.3 #+-cm
detRad = 7.8/2 #cm
detRadErr = 0.5/2 #+-cm
rad = np.sqrt(radc**2+detRad**2)
radErr = 1/2*rad*(np.sqrt((2*radcErr/radc*radc**2)**2 + np.sqrt(2*detRadErr/detRad*detRad**2)**2)
                  /(radc**2+detRad**2))


expectations = abundances * 1/4 * detRad**2/rad**2 * sourceactivity
expectationserr = np.sqrt((errAbund/abundances)**2+((2*detRadErr/detRad)*detRad**2/detRad**2)**2+((2*radErr/rad)*rad**2/rad**2)**2+(sourceactivityErr/sourceactivity)**2)*1/4*expectations
expectationsUpper = (abundances+errAbund) * 1/4 * (detRad+detRadErr)**2/(rad-radErr)**2 * sourceactivity
expectationsLower = (abundances-errAbund) * 1/4 * (detRad-detRadErr)**2/(rad+radErr)**2 * sourceactivity
measured = chs[:, 0][:, 1] * chs[:, 0][:, 2] * np.sqrt(2*np.pi)/time
measuredErr = np.sqrt((chs[:,1][:,1]/chs[:,0][:,1])**2+(chs[:,1][:,2]/chs[:,0][:,2])**2) * np.sqrt(2*np.pi)/time*measured
#areaerror = sqrt(sigmaerr/sigma)**2+(amperr/amp)**2)*sqrt(2*pi)*areal
efficiencies = measured/expectations
efferr = np.sqrt((expectationserr/expectations)**2+(measuredErr/measured)**2)*efficiencies
energies = chs[:,0][:,0]
energyerr = chs[:,1][:,0]

plt.scatter(energies, efficiencies, label='mean efficiency')
plt.scatter(energies, measured/expectationsUpper, label='lower efficiency')
plt.scatter(energies, measured/expectationsLower, label='upper efficiency')
#plt.scatter(chs[:,0][:,0], expectations, label='expected')
plt.legend()
plt.show()

plt.errorbar(energies, efficiencies, yerr=efferr, xerr=energyerr, fmt=',', label='measured efficiencies')
#plt.show()


def negexp(x, a, b, c, d):
    return a*np.exp(-(x*b) + c) + d


def reciproc(x, a, b):
    return a*1/x+b


popt, pcov = curve_fit(negexp, energies, efficiencies, p0=[1, 1, 0, 0], sigma=100*efferr)
print('\n', 'exp fit')
print('a :', popt[0])
print('b :', popt[1])
print('c', popt[2])
print('d', popt[3])
perr = np.sqrt(np.diag(pcov))
print('usikkerheder:', perr)
chmin = np.sum(((efficiencies - negexp(energies, *popt)) / efferr) ** 2)
print('chi2:', chmin, ' ---> p:', ss.chi2.cdf(chmin, 4), '\n')
xhelp = np.linspace(chs[0][0][0]-10, 4000, 100)
plt.plot(xhelp, negexp(xhelp, *popt), label='exponential function fit')

popt1, pcov1 = curve_fit(reciproc, energies, efficiencies, p0=[1, 0], sigma=100*efferr)
print('\n', 'rciproc fit')
print('a :', popt1[0])
print('b :', popt1[1])
perr1 = np.sqrt(np.diag(pcov1))
print('usikkerheder:', perr1)
chmin1 = np.sum(((efficiencies - reciproc(energies, *popt1)) / efferr) ** 2)
print('chi2:', chmin1, ' ---> p:', ss.chi2.cdf(chmin1, 4), '\n')

plt.plot(xhelp, reciproc(xhelp, *popt1), label='reciprocal function fit')



def power(x, a, b,c):
    return a*x**b+c


popt2, pcov2 = curve_fit(power, energies, efficiencies, p0=[1, -1, 0], sigma=100*efferr)
print('\n', 'power fit')
print('a :', popt2[0])
print('b :', popt2[1])
print('c :', popt2[2])
perr2 = np.sqrt(np.diag(pcov2))
print('usikkerheder:', perr2)
chmin2 = np.sum(((efficiencies - power(energies, *popt2)) / efferr) ** 2)
print('chi2:', chmin2, ' ---> p:', ss.chi2.cdf(chmin1, 4), '\n')

plt.plot(xhelp, power(xhelp, *popt2), label='power function fit')
plt.title('fits for detector efficiency as function of energy')
plt.xlabel('Energy (keV)')
plt.ylabel('Effeciency')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()

# efficiency fit
#a : 0.2268636064383986
#b : 4.380389222781053e-05
#usikkerheder: [8.26845799e-03 7.70049304e-06]
#chi2: 24.68507571767809  ---> p: 0.9999417974035777
realtiveUncertainty = efferr/efficiencies
