import numpy as np
from scipy.constants import G
from scipy.integrate import solve_ivp, quad
from matplotlib.pyplot import figure, show
import matplotlib

import general as ge
import coefficients as cf
import boundaries as bo



class inspiral_wave(object):
    """ Class for inspiral gravitational waves """
    
    def __init__(self, m1, m2):
        """ Initializing """
        
        self.m1 = m1.kg                                 # Primary mass
        self.m2 = m2.kg                                 # Secondary mass
        self.M = self.tot_m()                           # Total mass
        
        self.d1, self.d2 = m1.dist, m2.dist             # Masses in meters
        self.t1, self.t2 = m1.time, m2.time             # Masses in seconds
        
        self.etaV = self.eta()                          # eta
    
    
    def tot_m(self):
        return self.m1 + self.m2
    
    def eta(self):
        return self.m1 * self.m2 / np.power(self.M, 2)
    
    
    def deriv_x(self, t, x):
        """ Differential equation dx/dt """
        
        etaV = self.etaV                                    # eta
        
        term1 = 64 * etaV * np.power(x, 5) / 5              # Term in front
        
        sum1 = 1                                            # First term in sum
        
        sum2 = cf.a1(etaV) * x                              # Second term in sum
        sum3 = cf.a1half(etaV) * np.power(x, 1.5)           # Third term in the sum
        sum4 = cf.a2(etaV) * np.power(x, 2)                 # Fourth term in sum
        sum5 = cf.a2half(etaV) * np.power(x, 2.5)           # Fifth term in sum
        sum6 = cf.a3(etaV, x) * np.power(x, 3)              # Sixth term in sum
        sum7 = cf.a3half(etaV, x) * np.power(x, 3.5)        # Seventh term in sum
        
        sum8 = cf.a4(etaV, x) * np.power(x, 4)              # Eigth term in sum
        sum9 = cf.a4half(etaV, x) * np.power(x, 4.5)        # Nineth term in the sum
        sum10 = cf.a5(etaV, x) * np.power(x, 5)             # Tenth term in sum
        sum11 = cf.a5half(etaV, x) * np.power(x, 5.5)       # Eleventh term in sum
        sum12 = cf.a6(etaV, x) * np.power(x, 6)             # Twelfth term in sum
        
        part1 = sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7  # First major part
        part2 = sum8 + sum9 + sum10 + sum11 + sum12             # Second major part

        totSum = part1 + part2                              # Total sum
        
        return term1 * totSum / (self.t1 + self.t2)
    
    
    def deriv_phi(self, t, phi, x):
        """ Differential equation dPhi/dt """
        return np.power(x, 1.5) / (self.t1 + self.t2)
    
    
    def r_x(self, x):
        """ r as a function of x"""
        
        M = self.d1 + self.d2                               # Total mass
        etaV = self.etaV                                    # eta
        
        part1 = 1 / x + cf.r1(etaV) + cf.r2(etaV) * x       # First part
        part2 = cf.r3(etaV) * np.power(x, 2)                # Second part
        
        return M * (part1 + part2)
    
    def amp(self, r, dr, dPhi, R):
        """ Function to calculate A (t) """
        
        M = self.d1 + self.d2                               # Total mass in m
        etaV = self.etaV                                    # eta
        
        const = -2 * etaV * M / R                           # M and R in meters
        
        A1 = const * (np.power(dr, 2) + np.power(r*dPhi, 2) + M/r)
        A2 = const * 2 * r * dr * dPhi
        
        return A1, A2
    
    def hplus(self, r, dr, phi, dPhi, R):
        """ Plus polarization """
        A1, A2 = self.amp(r, dr, dPhi, R)                   # Amplitudes
        return A1 * np.cos(2 * phi) + A2 * np.sin(2 * phi)
    
    
    def hcross(self, r, dr, phi, dPhi, R):
        """ Cross polarization """
        A1, A2 = self.amp(r, dr, dPhi, R)                   # Amplitudes
        return A1 * np.sin(2 * phi) - A2 * np.cos(2 * phi)
    
    
    def find_phi(self, tLim, phi0, xV):
        """ Find phi(t) """
        
        # tNow = tLim[0]                                      # Lower time boundary
        
        # tRange = np.arange(tLim[0], tLim[1]+h, h)           # Range of time values
        # yResults = [phi0]                                   # List with results

        tRange = np.linspace(*tLim, 500)
        print(xV)
        solved = solve_ivp(self.deriv_phi, [0, 11.94], [phi0], t_eval=tRange, 
                           args=(xV))

        # for ind in range(len(tRange)-1):
        #     yNew = ge.runga_kuta_solver(self.deriv_phi, h, tRange[ind], yResults[ind], 
        #                                 xV[ind])
        #     yResults.append(yNew)
        
        return solved.t, solved.y
        # return tRange, np.asarray(yResults)


class solve_inspiral(object):
    """ Class to find the inspiral wavefunction """
    
    def __init__(self, m1, m2, tSteps, tLims):
        """ Initialization """
        
            # Initializing the inspiral wave
        self.wave = inspiral_wave(m1, m2)               # Initializing the wave
        self.etaV = self.wave.etaV                      # eta
        
        self.m1, self.m2 = m1.kg, m2.kg
        self.M = self.wave.M
        
        self.d1, self.d2 = m1.dist, m2.dist             # Masses in meters
        self.t1, self.t2 = m1.time, m2.time             # Masses in seconds
        
            # x integration values
        self.x0 = self.x_start()                        # Starting x value
        self.x1 = self.x_end()                          # Ending x value
        
            # Initializing the timesteps
        self.steps = tSteps
        self.lims = tLims
        
            # time and x values
        self.tV, self.xV = self.execute()
        # self.tV, self.xV = self.test_execute([0, 11.94], 0.0337111)
        
        if self.determine_steps(): self.x2V = self.xV
        else: self.x2V = self.xV[-1]
        
    
    def x_start(self):
        return bo.x_low(self.m1, self.m2)
    
    def x_end(self):
        return bo.x_high(self.etaV)
    
    def determine_steps(self):
        """ Check if multiple steps are given """
        if type(self.steps) == float: return 1
        return 0
    
    
    def test_execute(self, tLim, x0):
        """ Test Euler method """

        tRange = np.linspace(*tLim, 500)
        solved = solve_ivp(self.wave.deriv_x, tLim, [x0], t_eval=tRange, 
                           method="Radau")
        
        return solved.t, solved.y


    def execute(self):
        """ Execute solver """

        if self.determine_steps():
            return ge.execute_solver(self.wave.deriv_x, self.steps, self.lims, 
                                     self.x0)
        
        x0 = self.x0
        tList, xList = [], []
        
        for ind in range(len(self.steps)):
            tV, xV = ge.execute_solver(self.wave.deriv_x, self.steps[ind], 
                                       self.lims[ind], x0)
            
            tList.append(tV)
            xList.append(xV)
            
            x0 = xV[-1]
        
        return tList, xList
    
    def find_r(self):
        """ Find function r(x) """
        return self.wave.r_x(self.x2V)
    
    def omega_orb(self):
        """ Find omega (x) """
        return np.power(self.x2V, 1.5) / (self.t1 + self.t2)
    
    def func_phi(self, phi0):
        """ Find function phi """

        tV, xV = self.test_execute([0, 11.94], 0.0337111)
        return self.wave.find_phi([0, 11.94], phi0, xV)

        if self.determine_steps():
            return self.wave.find_phi(self.xV, self.steps, self.lims, phi0)
        
        p0 = phi0
        tList, phiList = [], []
        
        for ind in range(len(self.steps)):
            tRange, phiRange = self.wave.find_phi(self.xV[ind], self.steps[ind], 
                                                  self.lims[ind], p0)
            
            tList.append(tRange)
            phiList.append(phiRange)
            
            p0 = phiRange[-1]
        
        return tList[-1], phiList[-1]
    

    def h_wave(self, R, phi0=15):
        """ Find h+ and hx wave polarizations """
        
        tRange, phiVals = self.func_phi(phi0)               # Phi
        dPhi = self.omega_orb()                             # dPhi/dt
        
        rVals = self.find_r()                               # t & r
        diffR = ge.diff_func(rVals, tRange)                 # dr/dt
        
        hP = self.wave.hplus(rVals[1:], diffR, phiVals[1:], dPhi[1:], R)   # h_+
        hX = self.wave.hcross(rVals[1:], diffR, phiVals[1:], dPhi[1:], R)  # h_x
        
        return tRange, hP, hX


def main():   
    
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
    
#     someWave = solve_inspiral(M1, M2, tStep, tLimits)
#     timeRange, hPlus, hCross = someWave.h_wave(R)
    
    multWave = solve_inspiral(M1, M2, stepList, limList)
    tVals, xVals = multWave.execute()

    # xS = bo.x_low(20, 20)
    xS = 0.0337111
    tTest, xTest = multWave.test_execute([0, 11.94], xS)
    # testWave = solve_inspiral(M1, M2, 0.001, [0, 11.94])

    timeRange, hPlus, hCross = multWave.h_wave(R)
    # timeRange, hPlus, hCross = testWave.h_wave(R)


    matplotlib.rcParams['font.family'] = ['Times']
    
    # Plotting
    fig = figure(figsize=(14,7))
    ax = fig.add_subplot(1,1,1)

    ax.plot(tTest, xTest[0], label="test")
    # ax.plot(timeRange[1:], hCross/max(hCross), label=r"$h_x$", ls="--", color="red")
    # ax.plot(timeRange[1:], hPlus/max(hCross), label=r"$h_+$", color="navy")
    
    # ax.set_xlim(11.7, 11.926)
    
    ax.set_xlabel(r"$t$ (s)", fontsize=20)
    ax.set_ylabel(r"strain (normalized)", fontsize=20)
    ax.tick_params(axis="both", labelsize=22)
    
    ax.grid()
    ax.legend(fontsize=20)
    
    fig.tight_layout()
    # fig.savefig("inspiral.png")
    
    show()

if __name__ == "__main__":
    main()
