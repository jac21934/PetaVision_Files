# Imports
import os, sys
lib_path = os.path.abspath("/projects/pcsri/PetaVision/OpenPV/python/")
sys.path.append(lib_path)
import pvtools as pv
import numpy as np
from matplotlib import pyplot as plt

if len(sys.argv) != 2:
    print 'Usage: ' + sys.argv[0] + ' <checkpoint directory>'
    sys.exit()

width  = 16
height = 8

# Load and scale data
dictionary = pv.readpvpfile(sys.argv[1] +  "/V1ToInputError_W.pvp")['values'][0,0,]
dictionary -= np.min(dictionary)
dictionary /= np.max(dictionary)

# Display
fig, ax = plt.subplots(width,height,figsize=(3,3))
for i in range(width):
    for j in range(height):
       ax[i,j].imshow(dictionary[height*i+j],vmin=0.0,vmax=1.0)
       ax[i,j].set_axis_off()
plt.show()
#plt.savefig("feature_kernel.png", transparent=True)

