# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 20:16:49 2022

@author: almon
"""

import matplotlib.pyplot as plt
import scipy.constants as con
import numpy as np

folder = r"D:\OneDrive\Desktop\Python\test\20230207_2\\"


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


with open(folder + "Parameters.txt", 'r') as paramFile:
    params = {}
    units = paramFile.readline().split(": ")[1].removesuffix('\n')
    for line in paramFile:
        name_value = line.split("=")
        params[name_value[0]] = float(name_value[1])

with np.load(folder+"data.npz") as data:
    e2s = data['e2s']
    n1s = data['n1s']
    n2s = data['n2s']
    L0s = data['L0s']
    L1s = data['L1s']
    L2s = data['L2s']

pi = con.pi
if units == "Natural units":
    h = 2*pi  # Natural units #con.h
    hbar = 1  # con.hbar
    e = 1  # con.e
    k = 1  # con.Boltzmann
elif units == "SI":
    h = con.h
    hbar = con.hbar
    e = con.e
    k = con.Boltzman
else:
    raise Exception("Bad units")
T = params['T']  # Temperture in natural units
# Energies done in units of tunneling energy
U1 = params['U1']  # Dot one confinement energy
U2 = params['U2']  # Dot two confinement energy
e1 = params['e1']  # Dot one energy level
t = params['t']  # Tunneling energy
Delta = params['Delta']  # Lead-dot coupling energy
fermi = params['fermi']  # Fermi level, set to zero
kT = k*T
delta = params['delta']  # convergence factor
maxIt = params['maxIt']  # maximum number of iterations before giving up
# allowed difference between values for the occupation numbers
threshold = params['threshold']

numPoints = len(e2s)

G = e**2 * L0s
kappa = (1.0/kT)*(L2s - L1s**2/L0s)
S = (-1.0/(e*kT)) * L1s/L0s
ZT = 1.0/((L0s*L2s/(L1s**2)) - 1)

# Make graphs
plot(e2s, n1s, r"$\epsilon_2$", r"$n_1$")
plot(e2s, n2s, r"$\epsilon_2$", r"$n_2$")
plot(e2s, n1s + n2s, r"$\epsilon_2$", r"$n_1 + n_2$")

plot(e2s, G, r"$\epsilon_2$", r"$G$")
plot(e2s, kappa, r"$\epsilon_2$", r"$\kappa$")
plot(e2s, S, r"$\epsilon_2$", r"$S$")
plot(e2s, ZT, r"$\epsilon_2$", r"$ZT$")
