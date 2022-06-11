import numpy as np
from scipy.integrate import quad
from matplotlib.pyplot import figure, show

import coefficients as cf
import inspiral as ip
import boundaries as bo


def omegaQNM(eta):
    """ Determine parameter omega_QNM """
    return 1 - 0.63 * np.power(1 - cf.sfin(eta), 0.3)


def f_hat(t, m1, m2):
    """ Determine the function f^hat"""
    
    eta = ip.eta(m1, m2)
    
    kappa = cf.kappa(eta)                       # Coefficient kappa
    b = cf.b(eta)                               # Coefficient b
    cCoef = cf.cCoef(eta)                       # Coefficient c
    
    part1 = cCoef * np.power(1 + 1/kappa, 1 + kappa) / 2
    part2 = 1 + np.exp(-2 * t / b) / kappa
    
    return part1 * (1 - np.power(part2, -kappa))


def omega(t, m1, m2):
    """ Find omega(t) """
    eta = ip.eta(m1, m2)
    return omegaQNM(eta) * (1 - f_hat(t, m1, m2))


def find_dif(y, t):
    """ Differentiate a function, y, w.r.t. time, t """
    
    yVals = np.diff(y) / np.diff(t)             # y values
#     tVals = (t[1:] + t[:-1]) / 2                # t values
    
    return yVals#, tVals


def phi_gIRS(tStart, tFin, m1, m2):
    """ Find the phase by integrating the function omega(t) """
    
    etaV = ip.eta(m1, m2)                                  # eta parameter
    phiVals = quad(omega, tStart, tFin, args=(m1, m2))    # Integrating
    
    return phiVals[0]


def A(t, A0, m1, m2):
    """ Amplitude as function of time """
    
    etaV = ip.eta(m1, m2)                                  # eta parameter
    alpha = cf.alpha(etaV)                               # alpha coefficient
    omegaT = omega(t, m1, m2)                              # omega(t)
    
    func = f_hat(t, m1, m2)                                # function f^hat
    difFunc = find_dif(func, t)                         # df^hat/dt
    
    denomP = np.power(func, 2) + np.power(func, 4)      # Part of denominator
    part1 = np.abs(difFunc) / (1 + alpha * denomP[1:])      # Part of equation
    
    return A0 * np.sqrt(part1) / omegaT[1:]


def hMerger(t, A0, m1, m2):
    """ Waveform strain """
    
    tS, tF = t[0], t[-1]                                # Start & end times
    
    aT = A(t, A0, m1, m2)                               # Amplitude
    
    phiVals = []
    for ind in range(len(t)):
        phigIRS = phi_gIRS(t[0], t[ind], m1, m2)               # Phase
        phiVals.append(phigIRS)
    
    phiRes = np.asarray(phiVals[:-1])
    
        # e^(-ix) = cos(x) - i sin(x)
    realMerg = aT * np.cos(phiRes)                     # Real part
    imgMerg = -aT * np.sin(phiRes)                     # Imaginary part
    
    return realMerg, imgMerg


def main():

    M1 = bo.geom_units(20)
    M2 = bo.geom_units(20)
    M = bo.geom_units(40)
    timeRange = np.linspace(-0.05, 0.05, 1000)
    
    unitLessTime = timeRange / M.mass_time()
    A0 = 1 / M.mass_time()
    
    aVals = A(unitLessTime, A0, M1.mass_time(), M2.mass_time())
    realH, imgH = hMerger(unitLessTime, A0, M1.mass_time(), M2.mass_time())
    
    # Plotting
    fig = figure(figsize=(14,7))
    ax = fig.add_subplot(1,1,1)
    
    ax.plot(unitLessTime[:-1], aVals/max(aVals), label="A(t)")
#     ax.plot(unitLessTime[1:], imgH/max(abs(imgH)), ls="--", label=r"$h_x$", 
#             color="red")
#     ax.plot(unitLessTime[1:], realH/max(abs(imgH)), label=r"$h_+$", color="navy")
    
    ax.set_xlabel(r"$t$ (M$_\odot$)", fontsize=15)
    ax.set_ylabel(r"$A$ (normalized)", fontsize=15)
    ax.tick_params(axis="both", labelsize=15)
#     ax.set_title(r"dashed = h$_x$, solid = h$_+$", fontsize=15)
    
    ax.set_xlim(-100, 100)
    
    ax.grid()
    ax.legend(fontsize=15)
    
    fig.tight_layout()
#     fig.savefig("amplitude.png")
    
    show()

main()