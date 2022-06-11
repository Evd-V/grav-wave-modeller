import numpy as np



def a1(eta):
    """ Coefficient a1 """
    return -743 / 336 - 11 * eta / 4


def a1half(eta):
    """ Coefficient a3/2 """
    return 4 * np.pi


def a2(eta):
    """ Coefficient a2 """
    return 34103 / 18144 + 13661 * eta / 2016 + 59 * np.power(eta, 2) / 18


def a2half(eta):
    """ Coefficient a5/2 """
    return -4159 * np.pi / 672 - 189 * np.pi * eta / 8


def a3(eta, x):
    """ Coefficient a3 """
    
    gamma = 0.577216                                        # Euler's constant
    
    part1 = 16447322263 / 139708800  - 1712 * gamma / 105 
    part2 = 16 * np.pi * np.pi / 3 - 856 * np.log10(16 * x) / 105
    part3 = (- 56198689 / 217728 + 451 * np.pi * np.pi / 48) * eta
    part4 = 541 * np.power(eta, 2) / 896 - 5605 * np.power(eta, 3) / 2592
    
    return part1 + part2 + part3 + part4

def a3half(eta, x):
    """ Coefficient a7/2 """
    return -4415 / 4032 + 358675 * eta / 6048 + 91945 * np.power(eta,2) / 1512


def a4(eta, x):
    """ Coefficient a4 """
    
    part1 = 170.799 - 742.551 * eta + 370.173 * np.power(eta, 2)
    part2 = - 43.4703 * np.power(eta, 3) - 0.0249486 * np.power(eta, 4)
    part3 = (14.143 - 150.692 * eta) * np.log10(x)
    
    return part1 + part2 + part3


def a4half(eta, x):
    """ Coefficient a9/2 """
    
    part1 = 1047.25 - 2280.56 * eta + 923.756 * np.power(eta, 2)
    part2 = 22.7462 * np.power(eta, 3) - 102.446 * np.log10(x)
    
    return part1 + part2


def a5(eta, x):
    """ Coefficient a5 """
    
    part1 = 714.739 - 1936.48 * eta + 3058.95 * np.power(eta, 2)
    part2 = -514.288 * np.power(eta, 3) + 29.5523 * np.power(eta, 4) 
    part3 = -0.185941 * np.power(eta, 5)
    part4 = (-3.00846 + 1019.71 * eta + 1146.13 * np.power(eta, 2))
    
    return part1 + part2 + part3 + part4 * np.log10(x)


def a5half(eta, x):
    """ Coefficient a11/2 """
    
    part1 = 3622.99 - 11498.7 * eta + 12973.5 * np.power(eta, 2)
    part2 = -1623 * np.power(eta, 3) + 25.5499 * np.power(eta, 4)
    part3 = (83.1435 - 1893.65 * eta) * np.log10(x)
    
    return part1 + part2 + part3


def a6(eta, x):
    """ Coefficient a6 """
    
    part1 = 11583.1 - 45878.3 * eta + 33371.8 * np.power(eta, 2)
    part2 = -7650.04 * np.power(eta, 3) + 648.748 * np.power(eta, 4)
    part3 = -14.5589 * np.power(eta, 5) - 0.0925075 * np.power(eta, 6)
    
    part4 = -1155.61 + 7001.79 * eta 
    part5 = -2135.6 * np.power(eta, 2) - 2411.92 * np.power(eta, 3)
    part6 = (part4 + part5) * np.log10(x) + 33.2307 * np.log10(np.power(x, 2))
    
    return part1 + part2 + part3 + part6


def r1(eta):
    """ Coefficient r1PN """
    return 0.333333 * eta - 1


def r2(eta):
    """ Coefficient r2PN """
    return 4.75 * eta + 0.111111 * np.power(eta, 2)


def r3(eta):
    """ Coefficient r3PN """
    
    part1 = -7.51822 * eta - 3.08333 * np.power(eta, 2)
    part2 = 0.0246914 * np.power(eta, 3)
    return part1 + part2


def sfin(eta):
    """ Determine the coefficient s_fin """
    
    part1 = 2 * np.sqrt(3) * eta - 390 * np.power(eta, 2) / 79 
    part2 = 2379 * np.power(eta, 3) / 287 - 4621 * np.power(eta, 4) / 276
    
    return part1 + part2


def Q(eta):
    """ Determine coefficient Q(s_fin) """
    return 2 / np.power(1 - sfin(eta), 0.45)


def alpha(eta):
    """ Determine coefficient alpha(eta) """
    part1 = 16313 / 562 + 21345 * eta / 124
    return part1 / np.power(Q(eta), 2)


def b(eta):
    """ Determine coefficient b(eta) """
    return 16014 / 979 - 29132 * np.power(eta, 2) / 1343


def cCoef(eta):
    """ Determine coefficient c(eta) """
    part1 = 206 / 903 + 180 * np.sqrt(eta) / 1141
    return part1 + 424 * np.power(eta, 2) / (1205 * np.log10(eta))


def kappa(eta):
    """ Determine coefficient kappa(eta) """
    return 713 / 1056 - 23 * eta / 193


