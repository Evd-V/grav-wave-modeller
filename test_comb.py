import numpy as np
from matplotlib.pyplot import figure, show
import matplotlib

import general as ge
import boundaries as bo
import inspiral as ip
import merger as mg


def insp():
    R = 2.4e22                                                  # Distance in m
    M1 = bo.geom_units(20)
    M2 = bo.geom_units(20)
    
    mSec = M1.conv_kg_sec() + M2.conv_kg_sec()                  # Tot mass in s
    tIsco = 2 * mSec                                            # Time ISCO
    tStiff = 11.9257                                            # Stiff time
    tFinal = tStiff - tIsco                                     # Final time
    
    tStep = 0.001
    tLimits = (0, 11.925)
    
    t1Step, t2Step, t3Step = 0.01, 0.001, 0.00005
    t1Lim, t2Lim, t3Lim = (0, 11), (11, 11.7), (11.7, tFinal)
    
    stepList = [t1Step, t2Step, t3Step]
    limList = [t1Lim, t2Lim, t3Lim]
    
    
    multWave = ip.solve_inspiral(M1, M2, stepList, limList)
    tVals, xVals = multWave.execute()

    timeRange, hPlus, hCross = multWave.h_wave(R)

    difX = ge.diff_func(np.asarray(xVals[-1]), np.asarray(tVals[-1]))

    return timeRange, hPlus, hCross
    
def merger():
    
    M1 = bo.geom_units(20)
    M2 = bo.geom_units(20)
    
    # timeRange = np.linspace(-0.08725, 0.08725, 1000)          # Units M_sun
    timeRange = np.linspace(-0.05, 0.05, 500)
    testTime = np.copy(timeRange)
    
    mergeWave = mg.merger_wave(M1, M2)                     # Initializing wave
    unitLessTime, hP, hC = mergeWave.hMerger(timeRange) # Waves
    
    ampVals = mergeWave.A(unitLessTime)                 # Amplitude

    return testTime, hP, hC

def comb_waves():

    inspTime, inspHP, inspHC = insp()
    mergTime, mergHP, mergHC = merger()

    tau = 2.9e-3
    tS = 11.9273# + tau

    matplotlib.rcParams['font.family'] = ['Times']

    # Plotting
    fig = figure(figsize=(14,7))
    ax = fig.add_subplot(1,1,1)

    ax.plot(inspTime[1:], 0.57*inspHP/max(inspHP), label="Inspiral", color="navy")
    ax.plot(mergTime[1:]+tS, mergHP/max(mergHP), label="Merger", color="r", ls="--")

    ax.set_xlabel(r"$t$ (s)", fontsize=20)
    ax.set_ylabel("Strain (normalized)", fontsize=20)
    ax.tick_params(axis="both", labelsize=22)

    ax.set_xlim(11.85, 11.97)

    ax.legend(fontsize=20)
    ax.grid()

    fig.savefig("comb_wave.png")

    show()

comb_waves()