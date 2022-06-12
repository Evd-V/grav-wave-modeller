import numpy as np
from matplotlib.pyplot import figure, show

import boundaries as bo
from inspiral import solve_inspiral
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
        
        self.tRange, self.hIP, self.hIC = self.inspWave.h_wave(R)
    
    
    def freq_insp(self):
        """ Frequency of inspiral wave """
        return self.inspWave.omega_orb() / (0.5*np.pi)
    
    def freq_merge(self, t):
        """ Frequency of merger wave """
        return self.mergWave.omega(t/(self.t1 + self.t2)) / (2 * np.pi)
    
    


def main():
    """ Main function that will be executed """
    
    M1 = bo.geom_units(20)                                  # Primary mass
    M2 = bo.geom_units(20)                                  # Secondary mass
    timeRange = np.linspace(-0.05, 0.05, 1000)              # Time rane (s)
    
    T = M1.time + M2.time
    
    t1Step, t2Step, t3Step = 0.01, 0.001, 0.00005
    t1Lim, t2Lim, t3Lim = (0, 11), (11, 11.7), (11.7, 11.924)
    
    stepList = [t1Step, t2Step, t3Step]
    limList = [t1Lim, t2Lim, t3Lim]
    
    combWave = comb_waves(M1, M2, stepList, limList)        # Wave object
    
    fInsp = combWave.freq_insp()                            # f^inspiral
    fMerg = combWave.freq_merge(timeRange) / T              # f^merger
    
    redTime = combWave.tRange# / (combWave.t1 + combWave.t2)
    
        # Matching parameters
    tau = 2.84e-3                       # To overlap the frequency (s)
    phi0 = np.pi                        # To overlap h_+ inspiral & merger waves
    retTime = 0.5388e-3                 # Retarded time (s)
    deltaTime = 3 * retTime / 2         # Delta t to overlap peaks inspiral wave
    dTMerge = retTime                   # Shift for merger wave
    
    # Plotting
    fig = figure(figsize=(14,7))
    ax = fig.add_subplot(1,1,1)
    
    ax.plot(redTime-11.924, fInsp, label="Inspiral", color="red")
    ax.plot(timeRange+tau, fMerg, label="Merger", color="navy")
    
    ax.legend()
    ax.grid()
    
#     ax.set_xlim(-0.025, 0.025)
    
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    
    fig.tight_layout()
#     fig.savefig("frequency.png")
    
    show()

if __name__ == "__main__":
    main()
