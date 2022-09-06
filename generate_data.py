import numpy as np
import os
import warnings
from matplotlib.pyplot import figure, show

import merger as mg
import boundaries as bo


def save_data(m1, m2, tS=-.05, tF=.05):
    """ Save data of merger """

        # Creating consistent file names
    dirName = "merger_data/"
    fName = dirName + "merger" + str(m1) + "_" + str(m2) + ".csv"

    M1, M2 = bo.geom_units(m1), bo.geom_units(m2)
    waveObject = mg.merger_wave(M1, M2)             # Creating merger wave object
    timeRange = np.linspace(tS, tF, 250)            # Default time range
    saveTime = np.copy(timeRange)

    t, hP, hC = waveObject.hMerger(timeRange)       # Wave

    np.savetxt(fName, np.array([*zip(*[timeRange[1:], hP, hC])]))


def read_data(m1, m2):
    """ Read data of merger """

        # Creating consistent file names
    dirName = "./merger_data/"
    fName = dirName + "merger" + str(m1) + "_" + str(m2) + ".csv"

    data = np.loadtxt(fName, delimiter=" ")

    return data


def mass_ranges():
    """ Determine which masses are in the database """

        # Path to data files 
    path1 = "//Users//Evan//Documents//Evan//Studie//BSc//"
    path2 = "Honours College//grav-wave-modeller//merger_data"
    
    fileList = os.listdir(path1 + path2)    # List of files in database

    primList, secList = [], []              # Lists to store data in

    for file in fileList:                   # Looping over file names

            # Cleaning string
        newName = (file.replace('merger', '')).replace('.csv', '')
        undInd = newName.rfind('_')         # Position of underscore

        primMass = int(newName[:undInd])    # Primary mass value
        secMass = int(newName[undInd+1:])   # Secondary mass value

        primList.append(primMass)
        secList.append(secMass)
    
    return np.unique(primList), np.unique(secList)
    
def closest_ind(m, mList, ind):
    """ Find two closest indices for m in mList """

    if mList[ind] > m: return [ind-1, ind]
    elif mList[ind] < m: return [ind, ind+1]
    else: return [ind]                          # No interpolation needed


def find_inds(m1, m2):
    """ Interpolate data for primary or secondary mass"""

    primMass, secMass = mass_ranges()           # Prim and sec masses in database

    primInd = (np.abs(primMass - m1)).argmin()  # Index for primary mass
    secInd = (np.abs(secMass - m2)).argmin()    # Index for secondary mass

        # Finding two closest indices for primary & secondary mass
    pr = closest_ind(m1, primMass, primInd)
    sec = closest_ind(m2, secMass, secInd)

        # Removing -1 indices and indices longer than list
    return np.setdiff1d(pr, [-1, len(primMass)]), np.setdiff1d(sec, [-1, len(secMass)])


def find_names(m1, m2):
    """ Interpolate data """

    if m1 < m2:
        raise ValueError("Primary mass must be bigger than secondary mass")

    primMass, secMass = mass_ranges()       # Prim and sec masses in database
    primInd, secInd = find_inds(m1, m2)     # Find indices of closest values

    if len(primInd) == 1:                   # Only 1 ind. found for primary
        if (m1 not in primMass):            # Interpolation is required

            string1 = "Only 1 suitable primary mass found,"
            string2 = "interpolation might be innaccurate"
            warnings.warn(string1 + string2)
    
    if len(secInd) == 1:                    # Only 1 ind. found for secondary
        if (m2 not in secMass):             # Interpolation is required

            string1 = "Only 1 suitable secondary mass found,"
            string2 = "interpolation might be innaccurate"
            warnings.warn(string1 + string2)


    fNames = []                         # List for file names

        # Retrieving file names
    for mP in primMass[primInd]:        # Looping over primary masses

        fName = "merger" + str(mP)      # Primary mass

        for mS in secMass[secInd]:      # Looping over secondary masses
            
            fSec = fName + "_" + str(mS) + ".csv"
            fNames.append(fSec)
    
    return fNames

def open_data(m1, m2):
    return [np.loadtxt(f, delimiter='') for f in find_names(m1, m2)]


def plot_data(m1, m2, saveFig=None):
    """ Plot data by reading file """

    data = read_data(m1, m2)

    # Plotting
    fig = figure(figsize=(14,7))
    ax = fig.add_subplot(111)

    ax.plot(data[:,0], data[:,1], 'b', label=r"$h+$")
    ax.plot(data[:,0], data[:,2], 'r--', label=r"$h_x$")

    ax.set_xlabel(r"Time ($M_\odot$)", fontsize=16)
    ax.set_ylabel("Amplitude", fontsize=16)
    ax.tick_params(axis='both', labelsize=16)

    ax.legend(fontsize=16)
    ax.grid()

    if saveFig: fig.savefig(saveFig)
    show()
    
# save_data(20, 5)
# plot_data(20, 10)
# print(find_names(20, 16))