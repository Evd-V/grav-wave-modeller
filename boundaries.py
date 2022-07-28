import numpy as np
from scipy.constants import G, c
from astropy.constants import M_sun


def x_low(m1, m2, fLow=10):
    """ Find the lower bound for x """
    
    M = (m1 + m2)                                       # Total mass in kg
    brackets = G * M * np.pi * fLow / c**3              # Term in brackets
    
    return np.power(brackets, 2/3)

def x_high(eta):
    """ Upper boundary for x """
    return (1 + 7 * eta / 18) / 6

def t_low():
    return 0.1

def t_high(m1, m2):
    """ Upper boundary for time in integration """
    return c**3 / (np.power(6, 1.5) * G * (m1*2e30 + m2*2e30) * np.pi)

def t_final(m1, m2):
    """ Higher upper boundary, see paper """
    
    M = 4.923e-6                                        # Mass in units of sec
    tof = 2*M                                           # t_of
    tS = t_high(m1, m2)                                 # t_s
    
    return tS - tof


class geom_units(object):
    """ Units of solar mass """
    
    def __init__(self, mass):
        """ Initializing """
        
        self.mass = mass
        self.kg = self.mass_kg()
        self.dist = self.mass_dist()
        self.time = self.mass_time()
    
    def mass_kg(self):
        return self.mass * M_sun.value
    
    
    def conv_sec_m(self):
        """ convert seconds to meters """
        return c
    
    def conv_m_sec(self):
        """ Convert meters to seconds """
        return 1 / c
    
    def conv_kg_m(self):
        """ Convert kg to meters """
        return G * self.conv_m_sec()**2
    
    def conv_kg_sec(self):
        """ Convert kg to seconds """
        return self.conv_kg_m() * self.conv_m_sec()
    
    def mass_time(self):
        return self.kg * G * np.power(1/c, 3)
    
    def mass_dist(self):
        return self.kg * G * np.power(1/c, 2)
