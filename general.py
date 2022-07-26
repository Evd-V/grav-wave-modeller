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

def diff_func(y, t):
    """ Differentiate a function """
    return (y[1:] - y[:-1]) / (t[1:] - t[:-1])
