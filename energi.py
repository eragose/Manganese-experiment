import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# importing required modules
from zipfile import ZipFile


  
# specifying the zip file name
for i in range(5):
    file_name = f'Dag 2/Mg_data/Mg_1_maalning_dag2_ch000{i}.zip'
    
    # opening the zip file in READ mode
    with ZipFile(file_name, 'r') as zip:
        # printing all the contents of the zip file
        zip.printdir()
    
        # extracting all the files
        zip.extractall()


data = []
for i in range(5):
    data.append(np.loadtxt(f'Mg_1_maalning_dag2_ch000{i}.txt')[1])

def getCounts(i, lc: int = 20, hc: int = 6000):
    data = np.loadtxt(f"Mg_1_maalning_dag2_ch000{i}.txt")
    counts = data[:, 1]
    (x, y) = np.unique(counts, return_counts=True)
    lI = np.where(x >= lc)[0][0]
    hI = np.where(x >= hc)[0][0]
    x = x[lI:hI]
    y = y[lI:hI]
    plt.plot(x, y)
    plt.title('Mg')
    plt.show()
    return (x, y)

  

Mg1 = getCounts(0)
Mg2 = getCounts(1)
Mg3 = getCounts(2)
Mg4 = getCounts(3)
Mg5 = getCounts(4)


