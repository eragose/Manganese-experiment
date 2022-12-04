import numpy as np

#Defines the file to be split and reads it
file = 'C:/Users/Erik/OneDrive/Skrivebord/Eksperimentiel 3 ø1/Mg_1_maalning_dag2_ch000.txt'
#skiprows may have to be varied
dat = np.loadtxt(file, skiprows=4)

#define an interval and split the file into said interval
resolution = 10**(-8)
timeInterval = 10*60*resolution
indexes = [0]
for i in range(len(dat)):
    i = int(i)
    if dat[i][0] >= dat[indexes[-1][0]+timeInterval]:
        indexes += []
