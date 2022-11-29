import numpy as np

file = 'KaliCo_ch000.txt'

data = np.loadtxt(file, skiprows=5)

nr_files = 10
filesize = len(data)/10

for i in range(nr_files):
    lines = data[i*filesize:(i+1)*filesize]
    f = open("Mg_dag2_{}.txt".format(i), "w+")
    f.write(lines)
    f.close

