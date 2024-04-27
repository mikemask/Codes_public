import numpy as np
import scipy as sp
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from scipy.optimize import minimize, fsolve, curve_fit, SR1, least_squares
from scipy.fft import fft, fftfreq
from timeit import default_timer as timer

#w = np.array([100,150,210,310,440,640])
#g_p_exp = np.array([2.02e4,2.2144e4,2.5777e4,2.8266e4,2.833e4,3.2e4])
#g_dp_exp = np.array([1.42e4,1.7962e4,2.2195e4,2.8266e4,3.3897e4,4.066e4])


# Single material calibration

#os.system('mkdir calibration/bosch_1')
#
data = pd.read_csv('calibration/bosch_1.csv')

gp_exp = np.array(data.gp)
gdp_exp = np.array(data.gdp)
w = np.array(data.omega)

x0 = np.array([1,1,1,1])
limits = (1.e-20,1.e20)
bnds = (limits,limits,limits,limits)

gp_fit = []
gdp_fit = []

gp_sim = []
gdp_sim = []

def fit(params):

    km, kk, cm, ck = params

    j_p = 1./km + kk/(kk**2+(w*ck)**2)
    j_dp = 1./(w*cm) + w*ck/(kk**2+(w*ck)**2)

    gp = j_p/(j_p**2+j_dp**2)
    gdp = j_dp/(j_p**2+j_dp**2)

    return sum((gp/gp_exp-1.)**2+(gdp/gdp_exp-1.)**2)

#res = sp.optimize.fmin_l_bfgs_b(fit, x0, fprime=None, bounds=bnds, approx_grad=True, epsilon=1.e-8, factr=1.e7, maxiter=1000)

#res = minimize(fit, x0, method='nelder-mead', options={'maxiter':10000, 'xatol':1.e-6, 'fatol':1.e-6, 'disp':True}, bounds=bnds)
res = minimize(fit, x0, method='nelder-mead', options={'maxiter':20000, 'xatol':1.e-8, 'fatol':1.e-5, 'disp':True, 'adaptive':True}, bounds=bnds) #good for bosch
#res = minimize(fit, x0, method='nelder-mead', options={'maxiter':10000, 'xatol':1.e-10, 'fatol':1.e-10, 'disp':True}, bounds=bnds)
#res = minimize(fit, x0, method='BGFS', jac='3-point', bounds=bnds, options={'maxiter':1000, 'eps':1.e-10, 'ftol':1e-9, 'gtol':1.e-9})
##res = minimize(fit, x0, method='CG', jac='3-point', options={'gtol':1e-8, 'eps':1e-6, 'maxiter':100000})

km = res.x[0]
kk = res.x[1]
cm = res.x[2]
c4 = res.x[3]

print(res)

j_p = 1./km + kk/(kk**2+(w*ck)**2)
j_dp = 1./(w*cm) + w*ck/(kk**2+(w*ck)**2)

gp_fit = j_p/(j_p**2+j_dp**2)
gdp_fit = j_dp/(j_p**2+j_dp**2)

max_gp_exp = np.max(gp_exp)
max_gp_fit = np.max(gp_fit)
max_gdp_exp = np.max(gdp_exp)
max_gdp_fit = np.max(gdp_fit)
max_plot = np.max([max_gp_exp,max_gdp_exp,max_gp_fit,max_gdp_fit])*10

min_gp_exp = np.min(gp_exp)
min_gp_fit = np.min(gp_fit)
min_gdp_exp = np.min(gdp_exp)
min_gdp_fit = np.min(gdp_fit)
min_plot = np.min([min_gp_exp,min_gdp_exp,min_gp_fit,min_gdp_fit])/10

fig,ax1 = plt.subplots(1,1,figsize=(14,10))

ax1.plot(w,gp_exp,'-bs',label="$G'$ exp", markersize=10)
ax1.plot(w,gp_fit,'-ro',label="$G'$ fit", markersize=10)
ax1.set_xlabel('$\omega$ [rad/s]', fontsize=20)
ax1.set_ylabel("$G'$ [Pa]", fontsize=20)
ax1.tick_params(axis='both', labelsize=20)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylim(min_plot,max_plot)
ax1.legend(fontsize=20, loc=4)

ax2 = ax1.twinx()

ax2.plot(w,gdp_exp,'-g^',label="$G''$ exp",markersize=10)
ax2.plot(w,gdp_fit,'-k*',label="$G''$ fit",markersize=10)
ax2.set_ylabel("$G''$ [Pa]", fontsize=20)
ax2.tick_params(axis='both', labelsize=20)
ax2.set_yscale('log')
ax2.set_ylim(min_plot,max_plot)
ax2.legend(fontsize=20, loc=2)

plt.savefig('calibration/bosch_1/fit.png')

temp = open('user_defined_parameters_fcc.txt', 'r')
list_of_lines = temp.readlines()
list_of_lines[7] = 'variable k1 equal '+str(k1)+'\n'
list_of_lines[8] = 'variable k2 equal '+str(k2)+'\n'
list_of_lines[9] = 'variable k3 equal '+str(k3)+'\n'
list_of_lines[10] = 'variable k4 equal '+str(k4)+'\n'

temp = open('user_defined_parameters_fcc.txt', 'w')
temp.writelines(list_of_lines)
temp.close()

os.system('cp user_defined_parameters_fcc.txt calibration/bosch_1')
#
w = w[w>10]

#for i in range(len(w)):
#
#    str_it = str(i)
#    zeroed_it = str_it.zfill(4)
#
#    temp = open('user_defined_parameters_fcc.txt', 'r')
#    list_of_lines = temp.readlines()
#    list_of_lines[3] = 'variable angVelConst equal '+str(w[i])+'\n'
#
#    temp = open('user_defined_parameters_fcc.txt', 'w')
#    temp.writelines(list_of_lines)
#    temp.close()
#
#    os.system('mpirun -np 4 ./max_model')
#    os.system('mv data.txt calibration/bosch_1/data_'+zeroed_it+'.txt')

dfs = []
filenames = []

filenames = sorted(glob.glob('calibration/bosch_1/data_*'))

for filename in filenames:

    dfs.append(pd.read_csv(filename, sep=' '))

gp_sim = np.zeros(len(w))
gdp_sim = np.zeros(len(w))

for i in range(len(w)):

    tau_om = np.array(dfs[i].stress)
    gamma_om = np.array(dfs[i].strain)
    t_om = np.array(dfs[i].t)

    tau_om = np.append(0.0, tau_om)
    gamma_om = np.append(0.0, gamma_om)
    t_om = np.append(0.0, t_om)

    dt = t_om[1]

    #t_trunc = 18.*np.pi/w[i]

    #len_old = len(t_om)

    #t_om = t_om[t_om >= t_trunc]

    #id_trunc = len_old - len(t_om)

    #tau_om = tau_om[id_trunc:]
    #gamma_om = gamma_om[id_trunc:]

    tau_fft = fft(tau_om)
    xf = fftfreq(len(t_om),dt)[:len(t_om)//2]
    #xf = fftfreq(len(t_om),dt)

    #A = np.max(2./len(t_om)*np.abs(tau_fft))
    A = np.max(2./len(t_om)*np.abs(tau_fft[:len(t_om)//2]))

    def func(x,c):
        return A*np.sin(w[i]*x + c)

    popt, pcov = curve_fit(func,t_om,tau_om)
    tau_fit = func(t_om,popt[0])

    fig,ax = plt.subplots(1,1,figsize=(14,10))

    ax.plot(t_om,tau_om,'b',label="response")
    ax.plot(t_om,tau_fit,'r',label="fit")
    ax.set_xlabel('t [s]', fontsize=20)
    ax.set_ylabel(r"$\tau$ [Pa]", fontsize=20)
    ax.tick_params(axis='both', labelsize=20)
    ax.legend(fontsize=20)

    plt.savefig('calibration/bosch_1/stress_fit'+str(i)+'.png')

    fig,ax = plt.subplots(1,1,figsize=(14,10))

    #ax.plot(xf,2.0/len(t_om)*np.abs(tau_fft),'b')
    ax.plot(xf,2.0/len(t_om)*np.abs(tau_fft[:len(t_om)//2]),'b')
    ax.set_xlabel('f [Hz]', fontsize=20)
    ax.set_ylabel(r"$\tau$ [Pa]", fontsize=20)
    ax.set_xlim(0,1000)

    plt.savefig('calibration/bosch_1/stress_fft'+str(i)+'.png')

    G_abs = A/np.max(gamma_om)

    phi = popt[0]

    delta = phi/(w[i])

    if (phi >= 0 and phi <= np.pi/2):

        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

    elif (phi > 0.5*np.pi and phi <= np.pi):

        cos_phi = -np.cos(phi)
        sin_phi = np.sin(phi)

    elif (phi > np.pi and phi <= 1.5*np.pi):

        cos_phi = -np.cos(phi)
        sin_phi = -np.sin(phi)

    else:

        cos_phi = np.cos(phi)
        sin_phi = -np.sin(phi)

    gp_sim[i] = G_abs*cos_phi
    gdp_sim[i] = G_abs*sin_phi


max_gp_sim = np.max(gp_sim)
max_gdp_sim = np.max(gdp_sim)
max_plot = np.max([max_gp_exp,max_gdp_exp,max_gp_sim,max_gdp_sim])*10

min_gp_sim = np.min(gp_sim)
min_gdp_sim = np.min(gdp_sim)
min_plot = np.min([min_gp_exp,min_gdp_exp,min_gp_sim,min_gdp_sim])/10

gp_exp = gp_exp[len(gp_exp)-len(w):]
gdp_exp = gdp_exp[len(gdp_exp)-len(w):]
#
rms_err_gp = 100./len(gp_sim)*np.sqrt(np.sum(gp_exp-gp_sim)**2)/(max_gp_exp-min_gp_exp)
rms_err_gdp = 100./len(gdp_sim)*np.sqrt(np.sum(gdp_exp-gdp_sim)**2)/(max_gdp_exp-min_gdp_exp)

fig,ax1 = plt.subplots(1,1,figsize=(14,10))

ax1.plot(w,gp_exp,'-bs',label="$G'$ exp", markersize=10)
ax1.plot(w,gp_sim,'-ro',label="$G'$ sim", markersize=10)
ax1.set_xlabel('$\omega$ [rad/s]', fontsize=20)
ax1.set_ylabel("$G'$ [Pa]", fontsize=20)
ax1.tick_params(axis='both', labelsize=20)
ax1.set_title("$RMSD_{G'} ="+str(rms_err_gp)+"$% $RMSD_{G''}="+str(rms_err_gdp)+"$%", fontsize=20)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylim(min_plot,max_plot)
ax1.legend(fontsize=20, loc=1)

ax2 = ax1.twinx()

ax2.plot(w,gdp_exp,'-g^',label="$G''$ exp",markersize=10)
ax2.plot(w,gdp_sim,'-k*',label="$G''$ sim",markersize=10)
ax2.set_ylabel("$G''$ [Pa]", fontsize=20)
ax2.tick_params(axis='both', labelsize=20)
ax2.set_yscale('log')
ax2.set_ylim(min_plot,max_plot)
ax2.legend(fontsize=20, loc=2)

plt.savefig('calibration/bosch_1/sim.png')


## Calibration for multi material

for m in range(1,9):
#
    #os.system('mkdir calibration/bosch_'+str(m))

    data = pd.read_csv('calibration/bosch_'+str(m)+'.csv')

    gp_exp = np.array(data.gp)
    gdp_exp = np.array(data.gdp)
    w = np.array(data.omega)

    x0 = np.array([1,1,1,1,1,1,1,1,1])
    limits = (1.e-20,1.e20)
    #bnds = ((1e-20,1e20),(1e-20,1e20),(1e-20,1e20),(1e-20,1e20),(1e-20,1e20),(1e-20,1e20))
    bnds = (limits,limits,limits,limits,limits,limits,limits,limits,limits)

    def fit(params):

        k1, k2, k3, k4, kk, c1, c2, c3, c4 = params

        gp1 = k1*(w*c1/k1)**2/(1.+(w*c1/k1)**2)
        gp2 = k2*(w*c2/k2)**2/(1.+(w*c2/k2)**2)
        gp3 = k3*(w*c3/k3)**2/(1.+(w*c3/k3)**2)
        gp4 = k4*(w*c4/k4)**2/(1.+(w*c4/k4)**2)

        gdp1 = w*c1/(1.+(w*c1/k1)**2)
        gdp2 = w*c2/(1.+(w*c2/k2)**2)
        gdp3 = w*c3/(1.+(w*c3/k3)**2)
        gdp4 = w*c4/(1.+(w*c4/k4)**2)

        gp = gp1 + gp2 + gp3 + gp4 + kk
        gdp = gdp1 + gdp2 + gdp3 + gdp4

        return sum((gp/gp_exp-1.)**2+(gdp/gdp_exp-1.)**2)

    res = minimize(fit, x0, method='nelder-mead', options={'maxiter':20000, 'xatol':1.e-8, 'fatol':1.e-6, 'disp':True, 'adaptive':True}, bounds=bnds)

    print(res)

    k1 = res.x[0]
    k2 = res.x[1]
    k3 = res.x[2]
    k4 = res.x[3]
    kk = res.x[4]
    c1 = res.x[5]
    c2 = res.x[6]
    c3 = res.x[7]
    c4 = res.x[8]

    gp1 = k1*(w*c1/k1)**2/(1.+(w*c1/k1)**2)
    gp2 = k2*(w*c2/k2)**2/(1.+(w*c2/k2)**2)
    gp3 = k3*(w*c3/k3)**2/(1.+(w*c3/k3)**2)
    gp4 = k4*(w*c4/k4)**2/(1.+(w*c4/k4)**2)

    gdp1 = w*c1/(1.+(w*c1/k1)**2)
    gdp2 = w*c2/(1.+(w*c2/k2)**2)
    gdp3 = w*c3/(1.+(w*c3/k3)**2)
    gdp4 = w*c4/(1.+(w*c4/k4)**2)

    gp_fit = gp1 + gp2 + gp3 + gp4 + kk
    gdp_fit = gdp1 + gdp2 + gdp3 + gdp4

    gp_fit_all.append(gp_fit)
    gdp_fit_all.append(gdp_fit)

    max_gp_exp = np.max(gp_exp)
    max_gp_fit = np.max(gp_fit)
    max_gdp_exp = np.max(gdp_exp)
    max_gdp_fit = np.max(gdp_fit)
    max_plot = np.max([max_gp_exp,max_gdp_exp,max_gp_fit,max_gdp_fit])*10

    min_gp_exp = np.min(gp_exp)
    min_gp_fit = np.min(gp_fit)
    min_gdp_exp = np.min(gdp_exp)
    min_gdp_fit = np.min(gdp_fit)
    min_plot = np.min([min_gp_exp,min_gdp_exp,min_gp_fit,min_gdp_fit])/10

    rms_err_gp_fit = 100./len(gp_fit)*np.sqrt(np.sum(gp_exp-gp_fit)**2)/(max_gp_exp-min_gp_exp)
    rms_err_gdp_fit = 100./len(gdp_fit)*np.sqrt(np.sum(gdp_exp-gdp_fit)**2)/(max_gdp_exp-min_gdp_exp)

    err_fit.append([format(rms_err_gp_fit,".4f"),format(rms_err_gdp_fit,".4f")])

    #fig,ax1 = plt.subplots(1,1,figsize=(14,10))

    #ax1.plot(w,gp_exp,'-bs',label="$G'$ exp", markersize=10)
    #ax1.plot(w,gp_fit,'-ro',label="$G'$ fit", markersize=10)
    #ax1.set_xlabel('$\omega$ [rad/s]', fontsize=20)
    #ax1.set_ylabel("$G'$ [Pa]", fontsize=20)
    #ax1.tick_params(axis='both', labelsize=20)
    #ax1.set_title("$RMSD_{G'} ="+str(err_fit[m-1][0])+"$ $RMSD_{G''}="+str(err_fit[m-1][1])+"$", fontsize=20)
    #ax1.set_xscale('log')
    #ax1.set_yscale('log')
    #ax1.set_ylim(min_plot,max_plot)
    #ax1.legend(fontsize=20, loc=4)

    #ax2 = ax1.twinx()

    #ax2.plot(w,gdp_exp,'-g^',label="$G''$ exp",markersize=10)
    #ax2.plot(w,gdp_fit,'-k*',label="$G''$ fit",markersize=10)
    #ax2.set_ylabel("$G''$ [Pa]", fontsize=20)
    #ax2.tick_params(axis='both', labelsize=20)
    #ax2.set_yscale('log')
    #ax2.set_ylim(min_plot,max_plot)
    #ax2.legend(fontsize=20, loc=2)

    #plt.savefig('calibration/bosch_fit_errors/fit_'+str(m)+'.png')

    #temp = open('user_defined_parameters_fcc.txt', 'r')
    #list_of_lines = temp.readlines()
    #list_of_lines[7] = 'variable k1 equal '+str(k1)+'\n'
    #list_of_lines[8] = 'variable k2 equal '+str(k2)+'\n'
    #list_of_lines[9] = 'variable k3 equal '+str(k3)+'\n'
    #list_of_lines[10] = 'variable k4 equal '+str(k4)+'\n'
    #list_of_lines[11] = 'variable kk equal '+str(kk)+'\n'
    #list_of_lines[12] = 'variable c1 equal '+str(c1)+'\n'
    #list_of_lines[13] = 'variable c2 equal '+str(c2)+'\n'
    #list_of_lines[14] = 'variable c3 equal '+str(c3)+'\n'
    #list_of_lines[15] = 'variable c4 equal '+str(c4)+'\n'

    #temp = open('user_defined_parameters_fcc.txt', 'w')
    #temp.writelines(list_of_lines)
    #temp.close()

    #os.system('cp user_defined_parameters_fcc.txt calibration/bosch_'+str(m))

    #w = w[w > 100]

    #for i in range(len(w)):

    #    str_it = str(i)
    #    zeroed_it = str_it.zfill(4)

    #    temp = open('user_defined_parameters_fcc.txt', 'r')
    #    list_of_lines = temp.readlines()
    #    list_of_lines[3] = 'variable angVelConst equal '+str(w[i])+'\n'

    #    temp = open('user_defined_parameters_fcc.txt', 'w')
    #    temp.writelines(list_of_lines)
    #    temp.close()

    #    os.system('mpirun -np 4 ./max_model')
    #    os.system('mv data.txt calibration/bosch_'+str(m)+'/data_'+zeroed_it+'.txt')

    #dfs = []
    #filenames = []

    #filenames = sorted(glob.glob('calibration/bosch_'+str(m)+'/data_*'))

    #for filename in filenames:

    #    dfs.append(pd.read_csv(filename, sep=' '))

    #gp_sim = np.zeros(len(w))
    #gdp_sim = np.zeros(len(w))

    #for i in range(len(w)):

    #    tau_om = np.array(dfs[i].stress)
    #    gamma_om = np.array(dfs[i].strain)
    #    t_om = np.array(dfs[i].t)

    #    dt = t_om[0]
    #    tau_fft = fft(tau_om)
    #    xf = fftfreq(len(t_om),dt)[:len(t_om)//2]
    #    A = np.max(2./len(t_om)*np.abs(tau_fft[:len(t_om)//2]))

    #    def func(x,c):
    #        return A*np.sin(w[i]*x + c)

    #    popt, pcov = curve_fit(func,t_om,tau_om)
    #    tau_fit = func(t_om,popt[0])

    #    fig,ax = plt.subplots(1,1,figsize=(14,10))

    #    ax.plot(xf,2.0/len(t_om)*np.abs(tau_fft[:len(t_om)//2]),'b')
    #    ax.set_xlabel('f [Hz]', fontsize=20)
    #    ax.set_ylabel(r"$\tau$ [Pa]", fontsize=20)
    #    ax.tick_params(axis='both', labelsize=20)
    #    ax.set_xlim(0,1000)

    #    plt.savefig('calibration/bosch_'+str(m)+'/stress_fft'+str(i)+'.png')
    #    plt.close(fig)

    #    fig,ax = plt.subplots(1,1,figsize=(14,10))

    #    ax.plot(t_om,tau_om,'b',label="response")
    #    ax.plot(t_om,tau_fit,'r',label="fit")
    #    ax.set_xlabel('t [s]', fontsize=20)
    #    ax.set_ylabel(r"$\tau$ [Pa]", fontsize=20)
    #    ax.tick_params(axis='both', labelsize=20)
    #    ax.legend(fontsize=20)

    #    plt.savefig('calibration/bosch_'+str(m)+'/stress_fit'+str(i)+'.png')

    #    G_abs = A/np.max(gamma_om)

    #    phi = popt[0]

    #    delta = phi/(w[i])

    #    if (phi >= 0 and phi <= np.pi/2):

    #        cos_phi = np.cos(phi)
    #        sin_phi = np.sin(phi)

    #    elif (phi > 0.5*np.pi and phi <= np.pi):

    #        cos_phi = -np.cos(phi)
    #        sin_phi = np.sin(phi)

    #    elif (phi > np.pi and phi <= 1.5*np.pi):

    #        cos_phi = -np.cos(phi)
    #        sin_phi = -np.sin(phi)

    #    else:

    #        cos_phi = np.cos(phi)
    #        sin_phi = -np.sin(phi)

    #    gp_sim[i] = G_abs*cos_phi
    #    gdp_sim[i] = G_abs*sin_phi

    #gp_exp = gp_exp[len(gp_exp)-len(w):]
    #gdp_exp = gdp_exp[len(gdp_exp)-len(w):]

    #max_gp_exp = np.max(gp_exp)
    #max_gdp_exp = np.max(gdp_exp)
    #min_gp_exp = np.min(gp_exp)
    #min_gdp_exp = np.min(gdp_exp)

    #max_gp_sim = np.max(gp_sim)
    #max_gdp_sim = np.max(gdp_sim)
    #min_gp_sim = np.min(gp_sim)
    #min_gdp_sim = np.min(gdp_sim)

    #max_plot = np.max([max_gp_exp,max_gdp_exp,max_gp_sim,max_gdp_sim])*10
    #min_plot = np.min([min_gp_exp,min_gdp_exp,min_gp_sim,min_gdp_sim])/10

    #rms_err_gp_sim = 100./len(gp_sim)*np.sqrt(np.sum(gp_exp-gp_sim)**2)/(max_gp_exp-min_gp_exp)
    #rms_err_gdp_sim = 100./len(gdp_sim)*np.sqrt(np.sum(gdp_exp-gdp_sim)**2)/(max_gdp_exp-min_gdp_exp)

    #err_sim.append([format(rms_err_gp_sim,".4f"),format(rms_err_gdp_sim,".4f")])

    #fig,ax1 = plt.subplots(1,1,figsize=(14,10))

    #ax1.plot(w,gp_exp,'-bs',label="$G'$ exp", markersize=10)
    #ax1.plot(w,gp_sim,'-ro',label="$G'$ sim", markersize=10)
    #ax1.set_xlabel('$\omega$ [rad/s]', fontsize=20)
    #ax1.set_ylabel("$G'$ [Pa]", fontsize=20)
    #ax1.tick_params(axis='both', labelsize=20)
    #ax1.set_title("$RMSD_{G'} ="+str(err_sim[m-1][0])+"$ $RMSD_{G''}="+str(err_sim[m-1][1])+"$", fontsize=20)
    #ax1.set_xscale('log')
    #ax1.set_yscale('log')
    #ax1.set_ylim(min_plot,max_plot)
    #ax1.legend(fontsize=20, loc=4)

    #ax2 = ax1.twinx()

    #ax2.plot(w,gdp_exp,'-g^',label="$G''$ exp",markersize=10)
    #ax2.plot(w,gdp_sim,'-k*',label="$G''$ sim",markersize=10)
    #ax2.set_ylabel("$G''$ [Pa]", fontsize=20)
    #ax2.tick_params(axis='both', labelsize=20)
    #ax2.set_yscale('log')
    #ax2.set_ylim(min_plot,max_plot)
    #ax2.legend(fontsize=20, loc=2)

    #plt.savefig('calibration/bosch_'+str(m)+'/sim.png')

fig,ax = plt.subplots(1,1,figsize=(15,11))

p01, = ax.plot(1,1e7, 'k-o', markersize=7)
p02, = ax.plot(1,1e7, 'k-^', markersize = 7)
p1, = ax.plot(w,gp_fit_all[0], 'b-o',  markersize=7,linewidth=2)
p2, = ax.plot(w,gdp_fit_all[0], 'b-^', markersize=7,linewidth=2)
p3, = ax.plot(w,gp_fit_all[1], 'r-o', markersize=7,linewidth=2)
p4, = ax.plot(w,gdp_fit_all[1], 'r-^', markersize=7,linewidth=2)
p5, = ax.plot(w,gp_fit_all[2], 'g-o', markersize=7,linewidth=2)
p6, = ax.plot(w,gdp_fit_all[2], 'g-^', markersize=7,linewidth=2)
p7, = ax.plot(w,gp_fit_all[3], 'y-o', markersize=7,linewidth=2)
p8, = ax.plot(w,gdp_fit_all[3], 'y-^', markersize=7,linewidth=2)
p9, = ax.plot(w,gp_fit_all[4], 'c-o', markersize=7,linewidth=2)
p10, = ax.plot(w,gdp_fit_all[4], 'c-^', markersize=7,linewidth=2)
#p11, = ax.plot(w,gp_fit_all[5], 'm-o', markersize=7,linewidth=2)
#p12, = ax.plot(w,gdp_fit_all[5], 'm-^', markersize=7,linewidth=2)
#p13, = ax.plot(w,gp_fit_all[6], color='gray', linestyle='solid', marker='o', markersize=7,linewidth=2)
#p14, = ax.plot(w,gdp_fit_all[6], color='gray', linestyle='solid', marker='^', markersize=7,linewidth=2)
ax.set_xlabel('$\omega$ [rad/s]', fontsize=30)
ax.set_ylabel("$G', G''$ [Pa]", fontsize=30)
ax.tick_params(axis='both', labelsize=30)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(1e-1,1.e6)
ax.grid()
ax.legend([(p01),(p02),(p1,p2),(p3,p4),(p5,p6),(p7,p8),(p9,p10)],["G'"," G''",'material 1','material 2','material 3','material 4','material 5'], handler_map={tuple: HandlerTuple(ndivide=None)},fontsize=30)
plt.savefig('calibration/bosch_all_fit.png')
#
