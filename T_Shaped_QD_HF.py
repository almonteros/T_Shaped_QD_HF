# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 16:37:17 2022

@author: almon
"""

import matplotlib.pyplot as plt
from scipy.integrate import quad
import scipy.constants as con
import numpy as np
from os import path, makedirs
from time import perf_counter
from datetime import date
from numba import njit

plt.rcParams.update({'font.size': 32, 'axes.linewidth': 2,
                     'lines.linewidth': 2})

# Directory for saving data
directory = r"D:\OneDrive\Desktop\Python\test"

pi = con.pi
h = 2*pi  # Natural units #con.h
hbar = 1  # con.hbar
e = 1  # con.e
k = 1  # con.Boltzmann
T = 0.1  # Temperture in natural units
# Energies done in units of tunneling energy
U1 = 10.0  # Dot one confinement energy
U2 = 5.0  # Dot two confinement energy
e1 = -0.2  # Dot one energy level
e2 = 1.0  # Dot two energy level (to be looped over)
t = 1.0  # Tunneling energy
Delta = 0.5  # Lead-dot coupling energy
fermi = 0.0  # Fermi level, set to zero
kT = k*T
delta = 1e-3  # convergence factor
maxIt = 200  # maximum number of iterations before giving up
threshold = 1e-4  # allowed difference between values for the occupation numbers
numPoints = 200
e2s = np.linspace(-10, 10, numPoints)  # side dot energy values

save = False


def check_error(A, name='', threshold=1e-4):
    """
    Checks the relative error given the output of scipy.integrate.quad.

    Prints the name of the value, the value, and the relative error. No
    exception is raised.

    Parameters
    ----------
    A : tuple
        Output of scipy.integegrate.quad, any tuple containing the value and
        the absoulute error will work.
    name : string
           Name of the value we are checking.
    threshold : float, optional
        Allowed relative error. The default is 1e-4.

    Returns
    -------
    None.

    """

    ratio = A[1]/A[0]
    if ratio > threshold:
        print("high error")
        print(f"{name} = {A}: {ratio}")


def plot(x, y, xlabel, ylabel, title=""):
    """
    Plotting function.

    Parameters
    ----------
    x : list or array
        Independent variable.
    y : list or array
        Dependent variable.
    xlabel : string
        Label for the x-axis.
    ylabel : string
        Label for the y-axis.
    title : string, optional
        Title for the plot. The default is "".

    Returns
    -------
    None.

    """

    plt.figure(figsize=(12, 8))
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim(min(x), max(x))
    plt.tight_layout()


@njit
def FermiDirac(w):
    """
    Fermi-Dirac distribution.

    """

    return 1.0/(np.exp((hbar*w - fermi)/kT) + 1)
    # A runtime error, overflow will be reported when w is around 710.
    # In this case FermiDirac returns 0


@njit
def dfde(w):
    """
    Derivative of the Fermi-Dirac distribution.

    """

    return -(FermiDirac(w)**2 * np.exp((hbar*w - fermi)/kT)/kT)


@njit
def G11(w, n1, n2, e2):
    """
    Green's function for the main dot <<a_1:a_1^t>>.'

    Parameters
    ----------
    w : float or array
        Frequency.
    n1 : float
        Occupation number for the main dot.
    n2 : float
        Occupation number for the side dot.
    e2 : float
        Bias across the side dot.

    Returns
    -------
    answer : float or array
        The value of the Green's function.

    """

    w = hbar*w + hbar*1.0j*delta
    A = (w - e1)*(w - e1 - U1)/(w - e1 - U1*(1.0 - n1))
    B = - t**2*(w - e2 - U2*(1.0 - n2))/((w - e2)*(w - e2 - U2)) + 1.0j*Delta
    answer = (1.0/hbar)/(A + B)
    return answer


@njit
def G22(w, n1, n2, e2):
    """
    Green's function for the side dot <<a_2:a_2^t>>.'

    Parameters
    ----------
    w : float or array
        Frequency.
    n1 : float
        Occupation number for the main dot.
    n2 : float
        Occupation number for the side dot.
    e2 : float
        Bias across the side dot.

    Returns
    -------
    answer : float or array
        The value of the Green's function.

    """

    w = hbar*w + hbar*1.0j*delta
    A = (w - e2)*(w - e2 - U2)/(w - e2 - U2*(1.0 - n2))
    B = -t**2/((w - e1)*(w - e1 - U1)/(w - e1 - U1*(1.0 - n1)) + 1.0j*Delta)
    answer = (1.0/hbar)/(A + B)
    return answer


@njit
def l(w, n, n1, n2, e2):
    """
    Integrating function for calculating transport properties

    This function is often seen as L_n in the literature after it has been
    integrated, see Mahan chapter 3.

    Parameters
    ----------
    w : float or array
        Frequency.
    n :
        Order of the integrating function
    n1 : float
        Occupation number for the main dot.
    n2 : float
        Occupation number for the side dot.
    e2 : float
        Bias across the side dot.

    Returns
    -------
    l_n : float or array
        Value of integrating function.

    """
    Trans = np.imag(G11(w, n1, n2, e2))
    # The cosh term is the first derivative of the fermi distribution
    l_n = (w - fermi)**n * Trans/(np.cosh(hbar*w/kT) + 1)
    return l_n
    # np.cosh returns an overflow error, this ends with the function
    # returning zero which is close enough.


@njit
def n1Avg(w, n1, n2, e2, spin=1):
    """Average occupation for main dot."""

    return FermiDirac(w)*np.imag(G11(w, n1, n2, e2))


@njit
def n2Avg(w, n1, n2, e2, spin=1):
    """Averge occupation for side dot."""

    return FermiDirac(w)*np.imag(G22(w, n1, n2, e2))


ti = perf_counter()
n1s = np.zeros(numPoints)
n2s = np.zeros(numPoints)
n1n2 = np.zeros(numPoints)
L0s = np.zeros(numPoints)
L1s = np.zeros(numPoints)
L2s = np.zeros(numPoints)

# Looping over the side dot energy
n1_0 = 0.5
n2_0 = 0.5
for index, e2 in enumerate(e2s):
    # First calculate the occupation number given the previous value
    n1 = -1.0/pi*quad(n1Avg, -np.inf, np.inf, args=(n1_0, n2_0, e2), epsabs=0,
                      limit=100)[0]
    n2 = -1.0/pi*quad(n2Avg, -np.inf, np.inf, args=(n1_0, n2_0, e2), epsabs=0,
                      limit=100)[0]
    converged = False
    i = 0
    # Find the self-consistent solution for the Green's Function.
    while not converged and i <= maxIt:
        # Seed the occupation numbers with the previously calculated values
        # This speeds up the calculation but could cause a snowball effect
        # if there's something wrong.
        n1_0 = n1
        n2_0 = n2

        n1_i = quad(n1Avg, -np.inf, np.inf, args=(n1_0, n2_0, e2), epsabs=0,
                    limit=100)
        n2_i = quad(n2Avg, -np.inf, np.inf, args=(n1_0, n2_0, e2), epsabs=0,
                    limit=100)
        n1 = -1.0/pi*n1_i[0]
        n2 = -1.0/pi*n2_i[0]

        check_error(n1_i, "n1_i")
        check_error(n2_i, "n2_i")

        dn1 = abs(n1 - n1_0)
        dn2 = abs(n2 - n2_0)
        converged = (dn1 < threshold and dn2 < threshold)
        i += 1
    # Either the function converged for the given value or the max iteration
    # was reached
    if i >= maxIt:
        print(f"maxIt reached: {e2=}")
        print(f"dn1: {dn1} and dn2: {dn2}")

    n1s[index] = n1
    n2s[index] = n2

    # Calculate the integrals for the L_n functions
    L0_i = quad(l, -np.inf, np.inf, args=(0, n1, n2, e2), epsabs=0, limit=100)
    L1_i = quad(l, -np.inf, np.inf, args=(1, n1, n2, e2), epsabs=0,
                epsrel=0.0000001, limit=100)
    L2_i = quad(l, -np.inf, np.inf, args=(2, n1, n2, e2), epsabs=0, limit=100)

    check_error(L0_i)
    check_error(L1_i)
    check_error(L2_i)

    L0 = (1.0/h)*(-Delta*hbar**1)*L0_i[0]
    L1 = (1.0/h)*(-Delta*hbar**2)*L1_i[0]
    L2 = (1.0/h)*(-Delta*hbar**3)*L2_i[0]

    L0s[index] = L0
    L1s[index] = L1
    L2s[index] = L2

# Calculate the transport properties.
G = e**2 * L0s
kappa = (1.0/kT)*(L2s - L1s**2/L0s)
S = (-1.0/(e*kT)) * L1s/L0s
ZT = 1.0/((L0s*L2s/(L1s**2)) - 1)

tf = perf_counter()
print(f"Calculation time: {tf - ti}")

# Make graphs.
plot(e2s, n1s, r"$\epsilon_2$", r"$n_1$")
plot(e2s, n2s, r"$\epsilon_2$", r"$n_2$")
plot(e2s, n1s + n2s, r"$\epsilon_2$", r"$n_1 + n_2$")

plot(e2s, G, r"$\epsilon_2$", r"$G$")
plot(e2s, kappa, r"$\epsilon_2$", r"$\kappa$")
plot(e2s, S, r"$\epsilon_2$", r"$S$")
plot(e2s, ZT, r"$\epsilon_2$", r"$ZT$")

# Density of states calculation
w = np.linspace(-20, 20, numPoints)
index = numPoints//4
DOS = -1.0/pi*np.imag(G11(w, n1s[index], n2s[index], e2s[index]))
plot(w, DOS, r"$\omega$", r"$DOS$")

if save:
    # Setup FolderName and create.
    # Checks to see if the folder already exists, if so it creates a new one.
    n = 1
    folder = "/" + str(date.today()).replace('-', '') + "_" + str(n) + "/"
    newFolder = False
    while path.exists(directory + folder):
        folder = folder.replace("_"+str(n), "_" + str(n+1))
        n = n + 1
    newFolder = True
    makedirs(directory + folder)

    # Create a text file with all of the constants for this run
    if (k == 1 and e == 1 and hbar == 1):
        unit = "Units: Natural units"
    elif (e == con.e and k == con.k and hbar == con.hbar):
        unit = "Units: SI"
    else:
        unit = "Units: Unkwnown"
    with open(directory + folder + "Parameters.txt", 'w+') as paramFile:
        params = f"{unit}\n{U1=}\n{U2=}\n{e1=}\n{Delta=}\n{t=}\n{fermi=}\n"
        params += f"{T=}\n{delta=}\n{maxIt=}\n{threshold=}\n"
        paramFile.write(params)
    np.savez(directory + folder + "data",
             e2s=e2s, n1s=n1s, n2s=n2s, L0s=L0s, L1s=L1s, L2s=L2s)

# g11 = G11(w[None, :], n1s[:, None], n2s[:, None], e2s[:, None])
# g22 = G22(w[None, :], n1s[:, None], n2s[:, None], e2s[:, None])
# plt.figure()
# plt.pcolormesh(w, e2s, np.log10(-g11.imag))
# plt.colorbar()
# plt.title(r"$G_{11}(\omega, \espilon_2)$")
# plt.xlabel(r"$\omega$")
# plt.ylabel(r"$\epsilon_2$")

# plt.figure()
# plt.pcolormesh(w, e2s, np.log10(-g22.imag))
# plt.colorbar()
# plt.title(r"$G_{22}(\omega, \espilon_2)$")
# plt.xlabel(r"$\omega$")
# plt.ylabel(r"$\epsilon_2$")
