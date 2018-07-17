# Imports
import os, sys
lib_path = os.path.abspath("/home/jacob/Code/OpenPV/python/")
sys.path.append(lib_path)
import pvtools as pv
import numpy as np
from matplotlib import pyplot as plt


width  = 1
height = 1

# Load and scale data
dictionary = pv.readpvpfile("./V1ToInputError_W.pvp")['values'][0,0,]
dictionary -= np.min(dictionary)
dictionary /= np.max(dictionary)


print"Shape:", dictionary[0].shape
print"Values min/max:", dictionary.min(), dictionary.max()

for i in range(0,128):
    feature = dictionary[i]
    
    feature = feature.reshape(64,64)
    
    
    plt.imshow(feature,cmap='gray')
    plt.savefig("./figures/feature_" + str(i) + ".png") 


