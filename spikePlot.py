import sys
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np


matplotlib.rc('font', size=16)

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '*'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print'\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix),
    # Print New Line on Complete
    if iteration == total: 
        print()



def fit_exponential(x,m,c):
        return c * np.exp(-x/m)


DataPlot = False
Filename = "./kneeSpikeData_0.5"
numRuns = 4100

data = open(Filename, "r")
xs = [[]]
ys = [[]]
diff = [[]]
buff = 0
i = 0
j=0
xsbuff = []
ysbuff = []
diffbuff = []


for line in data:
    x, y,d,buff = line.split()
    xsbuff.append(float(x))
    ysbuff.append(float(y))
    diffbuff.append(float(d))
    i = i + 1
    if( i % 100 == 0):
        xs.append(xsbuff)
        ys.append(ysbuff)
        diff.append(diffbuff)
        xsbuff = []
        ysbuff = []
        diffbuff = []
        j = j + 1
        if( j == numRuns):
            break

    
data.close()
xs.pop(0)
ys.pop(0)
diff.pop(0)
exponent = []
fitStdDev = []
for i in range(numRuns):
    xs[i].pop(0)
    ys[i].pop(0)
    diff[i].pop(0)
    xs[i].pop(0)
    ys[i].pop(0)
    diff[i].pop(0)
    index = 0
    k = 0
    for d in diff[i]:
        if d <= 0:
            index = k
            break
        k = k + 1

    xs[i] = xs[i][:index]
    ys[i] = ys[i][:index]
    if xs[i] == [] or ys[i] == []:
        #print "Skipping run " + str(i+1)
        continue

    
    fitParams, Covariance = curve_fit(fit_exponential, xs[i], ys[i])
    expDat = []

    
    fitStdDev.append(np.sum(np.sqrt(np.diag(Covariance))))

    exponent.append(fitParams[0])


    

    if DataPlot == True:
        for x in xs[i]:
            expDat.append(fit_exponential(x, fitParams[0],fitParams[1]))

        line1 = plt.errorbar(xs[i], expDat,linestyle='solid')
        plot1, =  plt.plot(xs[i], ys[i], ".")
        plt.yscale("log")
        plotName = "./figures/SpikePlot" + str(i+1) + "0.5.png"
        plt.savefig(plotName)
        plt.close()
    printProgressBar(i + 1, numRuns, prefix = 'Progress:', suffix = 'Complete', length = 50)


    


    
plotExp, = plt.plot(exponent)
plt.savefig("./Exponents_0.5.png")
plt.close()

plotStd, = plt.plot(fitStdDev)
plt.savefig("./StdDev.png")


print np.min(exponent), exponent.index(np.min(exponent))
print np.max(exponent), exponent.index(np.max(exponent))
print np.mean(exponent)
print np.std(exponent)

    
