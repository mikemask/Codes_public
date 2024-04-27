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

data1 = pd.read_csv("data_c_def.txt", sep=' ')
data2 = pd.read_csv("data_m_def.txt", sep=' ')
data3 = pd.read_csv("data_f_def.txt", sep=' ')

pos_1 = np.array(data1.pos)
pos_2 = np.array(data2.pos)
pos_3 = np.array(data3.pos)

height = 1.01*0.946e-3*np.sqrt(2)*3+0.946e-3
area = height**2

print(height)

t = np.array(data1.t)
t = np.append(0.0, t)

eps = 0.01

eps_1 = (pos_1+4.73e-4-0.5*height)/height
eps_2 = (pos_2+3.07e-4-0.5*height)/height
eps_3 = (pos_3+2.27e-4-0.5*height)/height

eps_1 = np.append(0.0, eps_1)
eps_2 = np.append(0.0, eps_2)
eps_3 = np.append(0.0, eps_3)

cm = 1.e2
ck = 1.e2
km = 1.e4
kk = 1.e6

lambda_k = ck/kk

eps_an = 0.001/area*(1./km+1./kk*(1-np.exp(-t/lambda_k))+t/cm)

eps_an[0]=0

b, a = butter(1, 0.00004, btype='low', analog=False)

eps_fit1 = filtfilt(b, a, eps_1)
eps_fit2 = filtfilt(b, a, eps_2)
eps_fit3 = filtfilt(b, a, eps_3)

eps_fit1[0]=0
eps_fit2[0]=0
eps_fit3[0]=0

d = [0.946,0.614,0.454]

err_1 = np.abs((np.max(eps_fit1)-np.max(eps_an))/np.max(eps_an))*100
err_2 = np.abs((np.max(eps_fit2)-np.max(eps_an))/np.max(eps_an))*100
err_3 = np.abs((np.max(eps_fit3)-np.max(eps_an))/np.max(eps_an))*100

err = [err_1,err_2,err_3]

print(err_1,err_2,err_3)

def func(x,c):
    return A*np.sin(omega*x + c)

fig,ax1 = plt.subplots(1,1,figsize=(16,12))

ax1.plot(t, eps_1, 'k', label=r'$\varepsilon$', linewidth=2)
ax1.plot(t, eps_fit1, 'r', label=r'$\varepsilon_{fit}$', linewidth=2)
ax1.set_xlabel('t [s]', fontsize=20)
ax1.set_ylabel(r'$\varepsilon$', fontsize=30)
ax1.tick_params(axis='both',labelsize=30)
ax1.grid()
ax1.legend(fontsize=30, loc=4)

plt.savefig('fit_c_def.png')
plt.tight_layout()

plt.close(fig)

fig,ax1 = plt.subplots(1,1,figsize=(16,12))

ax1.plot(t, eps_2, 'k', label=r'$\varepsilon$', linewidth=2)
ax1.plot(t, eps_fit2, 'r', label=r'$\varepsilon_{fit}$', linewidth=2)
ax1.set_xlabel('t [s]', fontsize=20)
ax1.set_ylabel(r'$\varepsilon$', fontsize=30)
ax1.tick_params(axis='both',labelsize=30)
ax1.grid()
ax1.legend(fontsize=30, loc=4)

plt.savefig('fit_m_def.png')
plt.tight_layout()

plt.close(fig)

fig,ax1 = plt.subplots(1,1,figsize=(16,12))

ax1.plot(t, eps_3, 'k', label=r'$\varepsilon$', linewidth=2)
ax1.plot(t, eps_fit3, 'r', label=r'$\varepsilon_{fit}$', linewidth=2)
ax1.set_xlabel('t [s]', fontsize=20)
ax1.set_ylabel(r'$\varepsilon$', fontsize=30)
ax1.tick_params(axis='both',labelsize=30)
ax1.grid()
ax1.legend(fontsize=30, loc=4)

plt.savefig('fit_f_def.png')
plt.tight_layout()

plt.close(fig)

fig,ax = plt.subplots(1,1,figsize=(16,12))

ax.plot(t, eps_fit1, 'r^-', markevery=1200, label=r'$d=0.946\,mm$', markersize=6, linewidth=1.5,alpha=0.8)
ax.plot(t, eps_fit2, 'rs-', markevery=1200, label=r'$d=0.614\,mm$', markersize=6, linewidth=2,alpha=0.9)
ax.plot(t, eps_fit3, 'r*-', markevery=1200, label=r'$d=0.454\,mm$', markersize=8, linewidth=2.5,alpha=1)
ax.plot(t, eps_an, 'k', label='analytical', linewidth=4)
ax.set_xlabel('t [s]', fontsize=30)
ax.set_ylabel(r'$\varepsilon$', fontsize=30)
ax.tick_params(axis='both',labelsize=30)
ax.locator_params(nbins=3, axis='x')
ax.locator_params(nbins=5, axis='y')
ax.set_xlim(0.05,0.11)
ax.set_ylim(0.02,0.06)
ax.grid()
ax.legend(fontsize=30, loc=2)

ax2 = plt.axes([0,0,1,1])
ip = InsetPosition(ax, [0.55,0.1,0.4,0.3])
ax2.set_axes_locator(ip)

ax2.plot(d[0], err_1, 'r^', markersize=15, alpha=1)
ax2.plot(d[1], err_2, 'rs', markersize=15, alpha=1)
ax2.plot(d[2], err_3, 'r*', markersize=15, alpha=1)
ax2.plot(d, err, 'k--', label='error', linewidth=1.5, alpha=0.7)
ax2.set_xlabel('d [mm]',fontsize=25)
ax2.set_ylabel('$\chi$ [%]',fontsize=25)
ax2.tick_params(axis='both',labelsize=25)
ax2.legend(loc=4,fontsize=25)
ax2.locator_params(nbins=4, axis='y')
ax2.grid()


plt.tight_layout()
plt.savefig('fcc_conv_err_def.png')
plt.close(fig)

