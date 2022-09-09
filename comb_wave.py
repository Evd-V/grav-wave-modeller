import numpy as np
from scipy.interpolate import interp1d
from matplotlib.pyplot import figure, show

import boundaries as bo
import general as ge
from inspiral import inspiral_wave, solve_inspiral
from merger import merger_wave


class comb_waves(object):
    """ Class for combining the inspiral and merger waves """
    
    def __init__(self, M1, M2, tSteps, tLims, R=2.4e22):
        """ Initialization """
        
        self.m1, self.m2 = M1.kg, M2.kg                     # Mass in kg
        self.d1, self.d2 = M1.dist, M2.dist                 # Mass in m
        self.t1, self.t2 = M1.time, M2.time                 # Mass in s
        self.M = self.m1 + self.m2                          # Tot mass in kg
        
        self.inspWave = solve_inspiral(M1, M2, tSteps, tLims)   # Inspiral
        self.mergWave = merger_wave(M1, M2)                     # Merger
        
        timeRange = np.linspace(-0.05, 0.05, 1000)              # Time range (s)

        self.tRange, self.hIP, self.hIC = self.inspWave.h_wave(R)       # Inspiral
        self.tMerge, self.RM, self.IM = self.mergWave.hMerger(timeRange)# Merger
    

    def end_insp(self):
        """ Find last points of inspiral wave """

        finT = self.tRange[-100:]
        finWave = self.hIP[-100:]

        difWave = ge.diff_func(finWave, finT)

        return finT, finWave, difWave
    
    def max_merge(self):
        """ Find maximum of merger """
        ind = ge.find_closest(self.RM, max(self.RM))
        return ind, self.RM[:ind]
    
    def interp_spiral(self, tVals, cut=100):
        """ Interpolate spiral """
        interpObj = interp1d(self.tRange[1:], self.hIP)
        return interpObj(tVals)
        # return interpObj(np.linspace(self.tRange[-cut], 
        #                  self.tRange[-1], len(self.max_merge()[1])))
    
    def interp_merge(self, tVals, cut=100):
        """ Interpolate merger """

        indMax = ge.find_closest(self.RM, max(self.RM)) # Index of max
        interpObj = interp1d(self.tMerge[1:]/1.476e3, self.RM)  # Interp. object

        return interpObj(tVals)

        # np.linspace(self.tMerge[indMax-cut], 
        #  self.tMerge[indMax], len(self.tMerge[:indMax]))

    def topt(self, cut=300):
        """ Find optimized time for overlap """
                           
        tInspTest = np.linspace(self.tRange[-cut], self.tRange[-1], 500)
        tMerge = tInspTest - .5*(tInspTest[0] + tInspTest[-1])
        print(tMerge)
        print(tMerge[0], tMerge[-1])
        secTM, realM, imgM = self.mergWave.hMerger(tMerge)

        mergeInd = ge.find_closest(realM, max(realM))
        N = mergeInd + cut

        tInsp = np.linspace(self.tRange[-cut], self.tRange[-1], N)
        
        inspData = np.zeros((N))                    # Data of inspiral
        mergerData = np.zeros((N))                  # Data of merger

        cutInsp = self.interp_spiral(tInsp)         # Cut inspiral
        cutMerge = realM[:mergeInd]                 # Cut merger

            # Assigning values
        inspData += cutInsp / max(cutInsp)
        mergerData[:mergeInd] += cutMerge / max(cutMerge)

        originalMerge = cutMerge / max(cutMerge)    # Original merger
        finalMerger = np.zeros((N))                 # Array for best fit

            # Differentiate
        diffInsp = np.zeros((N-1))                  # Diff data insp.
        diffMerge = np.zeros((N-1))                 # Diff data merger

        inspDif = ge.diff_func(inspData, tInsp)     # Differentiate inspiral
        mergeDif = ge.diff_func(cutMerge, self.tMerge[:mergeInd])

        diffInsp += inspDif / max(inspDif) 
        diffMerge[:mergeInd-1] += mergeDif / max(mergeDif)

        indices, differences = [], []               # Lists to collect data

            # Find optimal time for transition
        for i in range(cut-1):

            diff = abs(inspData - mergerData)       # Difference in waves
            diffD = abs(diffInsp - diffMerge)       # Difference in derivatives

                # Somewhat arbitrary condition, often satisfied multiple times
            if min(diff) < 0.001 and min(diffD) < 0.001:
                differences.append(np.sqrt(min(diff)**2 + min(diffD)**2))
                indices.append(i)

                # Move to next shifted time step
            mergerData[i+1:mergeInd+i+1] = mergerData[i:mergeInd+i]
            diffMerge[i+1:mergeInd+i+1] = diffMerge[i:mergeInd+i]

        cInd = ge.find_closest(np.asarray(differences), 0)      # Best fit
        index = indices[cInd]                                   # Best fit index
        finalMerger[index:mergeInd+index] = originalMerge       # Shift merger

        return inspData, mergerData, tInsp, index+cut-1, finalMerger


    def freq_insp(self):
        """ Frequency of inspiral wave """
        return self.inspWave.omega_orb() / (0.5*np.pi)
    
    def freq_merge(self, t):
        """ Frequency of merger wave """
        return self.mergWave.omega(t/(self.t1 + self.t2)) / (2 * np.pi)
    
    


def main():
    """ Main function that will be executed """
    
    R = 2.4e22
    M1 = bo.geom_units(20)                                  # Primary mass
    M2 = bo.geom_units(20)                                  # Secondary mass
    timeRange = np.linspace(-0.05, 0.05, 1000)              # Time range (s)
    
    # T = M1.time + M2.time
    
    t1Step, t2Step, t3Step = 0.01, 0.001, 0.00005
    t1Lim, t2Lim, t3Lim = (0, 11), (11, 11.7), (11.7, 11.924)
    
    stepList = [t1Step, t2Step, t3Step]
    limList = [t1Lim, t2Lim, t3Lim]
    
    combWave = comb_waves(M1, M2, stepList, limList)        # Wave object
    
    # inspWave = solve_inspiral(M1, M2, stepList, limList)
    # tRange, hPlus, hCross = inspWave.h_wave(R)

    # mergeWave = merger_wave(M1, M2)
    # tMerge, realMerge, imgMerge = mergeWave.hMerger(timeRange)

    cut = 350
    inspData, mergeData, tInsp, index, test = combWave.topt(cut=cut)

    tFInsp, hP = combWave.tRange, combWave.hIP
    tMerge, realM = combWave.tMerge, combWave.RM
    
    # Plotting
    fig = figure(figsize=(14,7))
    ax = fig.add_subplot(1,1,1)

    # ax.plot(tRange[1:], hPlus/max(hPlus), color="navy", label="Insp.")
    # ax.plot(tMerge[1:]/1.476e3+11.923999, realMerge/max(realMerge), color="r", label="Merge")

    ax.plot(tInsp, inspData, color="b", label="insp")
    ax.plot(tInsp, test, color="r", label="merge")
    # ax.plot(tFInsp[1:], hP/max(hP), color="b", label="Insp")
    # ax.plot(tMerge[1:]/1.476e3+tInsp[index], realM/max(realM), color="r", label="Merge")

    # ax.plot(tRange[2:-1], inspDiff/max(inspDiff), color="navy", label="Insp.")
    # ax.plot(tMerge[2:-1]/1.476e3+11.921, mergeDiff/max(mergeDiff), color="r", label="Merge")
    
    # ax.plot(redTime-11.924, fInsp, label="Inspiral", color="red")
    # ax.plot(timeRange+tau, fMerg, label="Merger", color="navy")
    
    ax.legend()
    ax.grid()
    
    ax.set_xlim(11.91, 11.93)
    
    ax.set_xlabel("time (s)")
    ax.set_ylabel("wave")
    
    fig.tight_layout()
#     fig.savefig("frequency.png")
    
    show()

if __name__ == "__main__":
    main()
