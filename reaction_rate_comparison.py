import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const

kB = const.Boltzmann
NA = const.Avogadro
hbar = const.hbar
e = const.e

# ----------------------------
# Helpers
# ----------------------------

def reduced_mass(A1, A2):
    """
    Docstring for reduced_mass
    
    :param A1: Atomic mass of particle 1
    :param A2: Atomic mass of particle 2

    returns -> reduced mass in amu
    """
    return A1 * A2 / (A1 + A2)

def nonres_reaction_rate(T, Z1, Z2, A1, A2, S):
    """
    Docstring for nonres_reaction_rate
    
    :param omega_gamma: Description
    :param T: Temperature in GK
    :param Z1: Proton number of particle 1
    :param Z2: Proton number of particle 2
    :param A1: Atomic mass of particle 1
    :param A2: Atomic mass of particle 2
    :param S: S-factor

    Returns -> non-resonant reaction rate by NA (Avogadro's number); i.e. N_{A}<sigma nu>
    """

    mu = reduced_mass(A1, A2)
    C = 7.83e9 # constant
    C2 = -4.2487 # constant

    return C*((Z1*Z2)/(mu*T))**(1/3) * S * np.exp(C2*((Z1**2 * Z2**2 * mu)/(T))**(1/3))
    
def resonant_reaction_rate(omega_gamma, T, A1, A2, Er):
    """
    Docstring for resonant_reaction_rate
    
    :param omega_gamma: Description
    :param T: Temperature in GK
    :param A1: Atomic mass of particle 1
    :param A2: Atomic mass of particle 2
    :param Er: Resonant energy of center of mass

    Returns -> resonant reaction rate by NA (Avogadro's number); i.e. N_{A}<sigma nu>
    """

    C = 1.54e11 # constant
    C2 = 11.605 # constant

    mu = reduced_mass(A1, A2)

    return C*omega_gamma*(1/(mu*T))**(3/2) * np.exp(-(C2*Er/T))


def Breit_Wigner(T, A1, A2, cross_section, E):
    return 0

def cross_section(E, eta, S):
    return 0

def Sommerfield_parameter(Z1, Z2, A1, A2, E):
    
    mu = reduced_mass(A1, A2)
    v = np.sqrt((2*E)/mu)

    return (Z1*Z2*e**2)/(hbar*v)