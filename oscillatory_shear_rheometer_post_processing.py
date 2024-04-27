import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd
import mpl_axes_aligner
import math
import glob
import os
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)
from scipy.signal import correlate, butter, filtfilt
from scipy.optimize import curve_fit, fsolve
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
from sympy.solvers import solve
from sympy import Symbol

omega = 100

data = pd.read_csv("data_lag.txt", sep=' ')

tau = np.array(data.stress)
tau = np.append(0.0, tau)

gamma = np.array(data.strain)
gamma = np.append(0.0, gamma)

t = np.array(data.t)
t = np.append(0.0, t)

dt=t[1]

tau_fft = fft(tau)
xf = fftfreq(len(t),dt)[:len(t)//2]

A = np.max(2./len(tau_fft)*np.abs(tau_fft))

def func(x,c):
    return A*np.sin(omega*x + c)

popt, pcov = curve_fit(func,t,tau)
tau_fit = func(t,popt[0])

fig,ax = plt.subplots(1,1,figsize=(16,12))

ax.plot(t, tau, 'k', label=r'$\tau$', linewidth=2)
ax.plot(t, tau_fit, 'r', label=r'$\tau_{fit}$', linewidth=2)
ax.set_xlabel('t [s]', fontsize=30)
ax.set_ylabel(r'$\tau$ [Pa]', fontsize=30)
ax.tick_params(axis='both',labelsize=30)
ax.grid()
ax.legend(fontsize=30, loc=4)
plt.tight_layout()
plt.savefig('stress_fit.png')
plt.close(fig)

fig,ax = plt.subplots(1,1,figsize=(16,12))

ax.plot(xf*2.*np.pi, 2./len(tau_fft)*np.abs(tau_fft)[:len(t)//2], 'k', label=r'FFT')
ax.set_xlabel('$\omega$ [rad/s]', fontsize=30)
ax.set_ylabel(r'$\sigma_{FFT}$ [Pa]', fontsize=30)
ax.set_xlim(0,2000)
ax.set_yscale('log')
ax.tick_params(axis='both',labelsize=30)
ax.grid()
ax.legend(fontsize=30)
plt.tight_layout()
plt.savefig('stress_fft.png')
plt.close(fig)

fig,ax1 = plt.subplots(1,1,figsize=(16,12))

ax1.plot(t, tau_fit, 'k', label=r'$\tau$', linewidth=2)
ax1.set_xlabel('t [s]', fontsize=30)
ax1.set_ylabel(r'$\tau$ [Pa]', fontsize=30)
ax1.tick_params(axis='both',labelsize=30)
ax1.grid()
ax1.legend(fontsize=30, loc=1)

ax2 = ax1.twinx()

ax2.plot(t, gamma, 'r', label=r'$\gamma$', linewidth=2)
ax2.set_ylabel(r'$\gamma$', fontsize=30)
ax2.legend(fontsize=30, loc=3)
ax2.tick_params(axis='both',labelsize=30)
plt.tight_layout()
plt.savefig('lag.png')

plt.close(fig)

