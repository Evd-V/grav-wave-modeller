import numpy as np
from matplotlib.pyplot import figure, show

import merger as mg
import boundaries as bo


def save_data(m1, m2, tS=-.05, tF=.05):
    """ Save data of merger """

        # Creating consistent file names
    dirName = "merger_data/"
    fName = dirName + "merger" + str(m1) + "_" + str(m2) + ".csv"

    M1, M2 = bo.geom_units(m1), bo.geom_units(m2)
    waveObject = mg.merger_wave(M1, M2)                     # Creating merger wave object
    timeRange = np.linspace(tS, tF, 250)                    # Default time range
    saveTime = np.copy(timeRange)

    t, hP, hC = waveObject.hMerger(timeRange)        # Wave

    reshapedArray = np.reshape(np.array([saveTime[1:], hP, hC]), 
                               (len(hP), 3))

    # np.savetxt(fName, reshapedArray)
    np.savetxt(fName, np.array([timeRange[1:], hP, hC]))

def read_data(m1, m2):
    """ Read data of merger """

        # Creating consistent file names
    dirName = "./merger_data"
    fName = dirName + "merger" + str(m1) + "_" + str(m2) + ".csv"

    data = np.loadtxt(fName, delimiter=" ")

    return data



def interp_value(m, mass="Primary"):
    """ Interpolate data for primary or secondary mass"""

    if mass == "Primary": listMass = np.array([15,20])      # Primary mass
    elif mass == "Secondary": listMass = np.array([15,20])  # Secondary mass

    secInd = (np.abs(listMass - m)).argmin()            # Index for secondary mass

        # Finding second closest index
    if listMass[secInd] < m: return secInd-1, secInd
    elif listMass[secInd] > m: return secInd, secInd+1
    else: return secInd                                 # No interpolation needed





def plot_data(m1, m2, saveFig=None):
    """ Plot data by reading file """

    data = read_data(m1, m2)

    # Plotting
    fig = figure(figsize=(14,7))
    ax = fig.add_subplot(111)

    ax.plot(data[0], data[1], 'b', label=r"$h+$")
    ax.plot(data[0], data[2], 'r--', label=r"$h_x$")

    ax.set_xlabel(r"Time ($M_\odot$)", fontsize=16)
    ax.set_ylabel("Amplitude", fontsize=16)
    ax.tick_params(axis='both', labelsize=16)

    ax.legend(fontsize=16)
    ax.grid()

    if saveFig: fig.savefig(saveFig)
    show()
    

save_data(20, 15)
plot_data(20, 15)