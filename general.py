import numpy as np


def find_closest(array, value):
    """ Find closest element to value in list """
    return (np.abs(array - value)).argmin()


def runga_kuta_solver(func, h, t0, y0, *args):
    """ Runga kuta solver for differential equations """
    
    k1 = func(t0, y0, *args)                            # First coefficient
    
    y1 = y0 + h * k1 / 2                                # New evaluation point
    k2 = func(t0 + h/2, y1, *args)                      # Second coefficient
    
    y2 = y0 + h * k2 / 2                                # New evaluation point
    k3 = func(t0 + h/2, y2, *args)                      # Third coefficient
    
    y3 = y0 + h * k3                                    # New evaluation point
    k4 = func(t0 + h, y3, *args)                        # Fourth coefficient
    
    yN = y0 + h * (k1 + 2*k2 + 2*k3 + k4) / 6           # Next y value
    
    return yN


def execute_solver(func, h, tLim, y0, *args):
    """ Execute the Runga kuta solver """
    
    tNow = tLim[0]                                      # Lower time boundary
    
    tRange = np.arange(tLim[0], tLim[1]+h, h)           # Range of time values
    yResults = [y0]                                     # List with results
    
    for ind in range(len(tRange)-1):
        yNew = runga_kuta_solver(func, h, tRange[ind], yResults[ind], *args)
        yResults.append(yNew)
    
    return tRange, np.asarray(yResults)

def adaptive_t(y, h, q=0.95, eps=0.0005):
    """ Solve ODE using adaptive time step """
    return q * eps * sum(h) / abs(sum(y))

def euler(func, h, t0, y0):
    """ Euler method """
    return y0 + h * func(t0, y0)

def adaptive_euler(func, hS, y, t, *args):
    """ Adaptive time step Euler method """
    h2 = adaptive_t(y, hS, *args)
    return euler(func, h2, t[1], y[1])

def execute_euler(func, tLim, *args):
    """ Execute adaptive Euler method """

    yV = [0.0337111, 0.03371804]
    hV = [0.01, 0.009]
    t = np.linspace(tLim[0], tLim[1], 1500)           # Test

    y2 = adaptive_euler(func, hV, yV, t)

    for n in range(len(t)):

        yN = adaptive_euler(func, hV[n:n+2], yV[n:n+2], t)
        hN = adaptive_t(yV[n:n+2], hV[n:n+2], *args)

        yV.append(yN)
        hV.append(hN)
    
    return t, yV


def diff_func(y, t):
    """ Differentiate a function """
    return (y[1:] - y[:-1]) / (t[1:] - t[:-1])
