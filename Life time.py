import numpy as np
import pandas as pd
import glob
import os
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import zipfile

folder = 'Dag 2\Mg_data'

#plt.rcParams['figure.format'] = 'svg'

names = glob.glob(os.path.join(folder,'*.zip'))
print(names)
archive = zipfile.ZipFile(names[0], 'r')
file = archive.read('Mg_1_maalning_dag2_ch0000.txt')
#new = file.decode('utf-8')
new = int.from_bytes(file,'little')
print(new)
#print(type(file[:100]))



