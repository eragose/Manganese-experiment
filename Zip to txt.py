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
        print('Extracting all the files now...')
        zip.extractall()
        print('Done!')