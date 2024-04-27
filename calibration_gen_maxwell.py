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


## Calibration for multi-material

def fitting(mat,gp_exp,gdp_exp,w):

    x0 = np.array([1,1,1,1,1,1,1,1,1])
    limits = (1.e-20,1.e20)
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

    res = minimize(fit, x0, method='nelder-mead', options={'maxiter':100000, 'xatol':1.e-8, 'fatol':1.e-8, 'disp':True, 'adaptive':True}, bounds=bnds)

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

    rms_err_gp_fit = 100.*np.sqrt(np.sum((gp_exp-gp_fit)**2)/len(w))/(max_gp_exp-min_gp_exp)
    rms_err_gdp_fit = 100.*np.sqrt(np.sum((gdp_exp-gdp_fit)**2)/len(w))/(max_gdp_exp-min_gdp_exp)

    err_fit = [format(rms_err_gp_fit,".4f"),format(rms_err_gdp_fit,".4f")]

    fig,ax1 = plt.subplots(1,1,figsize=(14,10))

    ax1.plot(w,gp_exp,'-bs',label="$G'$ exp", markersize=10)
    ax1.plot(w,gp_fit,'-ro',label="$G'$ fit", markersize=10)
    ax1.set_xlabel('$\omega$ [rad/s]', fontsize=20)
    ax1.set_ylabel("$G'$ [Pa]", fontsize=20)
    ax1.tick_params(axis='both', labelsize=20)
    ax1.set_title("$RMSD_{G'} ="+str(err_fit[0])+"$%  $RMSD_{G''}="+str(err_fit[1])+"$%", fontsize=20)
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

    plt.savefig('calibration/bosch_'+str(mat)+'/fit.png')

    plt.close(fig)

    temp = open('user_defined_parameters_fcc.txt', 'r')
    #temp = open('user_defined_parameters_hcp.txt', 'r')
    list_of_lines = temp.readlines()
    list_of_lines[7] = 'variable k1 equal '+str(k1)+'\n'
    list_of_lines[8] = 'variable k2 equal '+str(k2)+'\n'
    list_of_lines[9] = 'variable k3 equal '+str(k3)+'\n'
    list_of_lines[10] = 'variable k4 equal '+str(k4)+'\n'
    list_of_lines[11] = 'variable kk equal '+str(kk)+'\n'
    list_of_lines[12] = 'variable c1 equal '+str(c1)+'\n'
    list_of_lines[13] = 'variable c2 equal '+str(c2)+'\n'
    list_of_lines[14] = 'variable c3 equal '+str(c3)+'\n'
    list_of_lines[15] = 'variable c4 equal '+str(c4)+'\n'
    list_of_lines[16] = 'variable ns_ratio equal '+str(1.)+'\n'

    temp = open('user_defined_parameters_fcc.txt', 'w')
    #temp = open('user_defined_parameters_hcp.txt', 'w')
    temp.writelines(list_of_lines)
    temp.close()

    os.system('cp user_defined_parameters_fcc.txt calibration/bosch_'+str(mat))
    #os.system('cp user_defined_parameters_hcp.txt calibration/bosch_'+str(mat))

gp_exp_all = []
gdp_exp_all = []
gp_sim_all = []
gdp_sim_all = []

for m in range(1,6):

    #os.system('mkdir calibration/bosch_'+str(m))

    data = pd.read_csv('calibration/bosch_'+str(m)+'/bosch_'+str(m)+'.csv')

    gp_exp = np.array(data.gp)
    gdp_exp = np.array(data.gdp)
    w = np.array(data.omega)

    #fitting(m,gp_exp,gdp_exp,w)

    w = w[w > 10]

    #gp_exp = gp_exp[:len(w)]
    #gdp_exp = gdp_exp[:len(w)]

    gp_exp = gp_exp[len(gp_exp)-len(w):]
    gdp_exp = gdp_exp[len(gdp_exp)-len(w):]

    gp_exp_all.append(gp_exp)
    gdp_exp_all.append(gdp_exp)

    max_gp_exp = np.max(gp_exp)
    max_gdp_exp = np.max(gdp_exp)
    min_gp_exp = np.min(gp_exp)
    min_gdp_exp = np.min(gdp_exp)

    def magic(psi):

        temp = open('user_defined_parameters_fcc.txt', 'r')
        #temp = open('user_defined_parameters_hcp.txt', 'r')
        list_of_lines = temp.readlines()
        list_of_lines[16] = 'variable ns_ratio equal '+str(psi)+'\n'

        temp = open('user_defined_parameters_fcc.txt', 'w')
        #temp = open('user_defined_parameters_hcp.txt', 'w')
        temp.writelines(list_of_lines)
        temp.close()

        #os.system('cp user_defined_parameters_hcp.txt calibration/bosch_'+str(m))
        os.system('cp user_defined_parameters_fcc.txt calibration/bosch_'+str(m))

        for i in range(len(w)):

            omega = w[i]

            str_it = str(i)
            zeroed_it = str_it.zfill(4)

            temp = open('user_defined_parameters_fcc.txt', 'r')
            #temp = open('user_defined_parameters_hcp.txt', 'r')
            list_of_lines = temp.readlines()
            list_of_lines[3] = 'variable angVelConst equal '+str(omega)+'\n'
            temp = open('user_defined_parameters_fcc.txt', 'w')
            #temp = open('user_defined_parameters_hcp.txt', 'w')
            temp.writelines(list_of_lines)
            temp.close()

            os.system('mpirun -np 4 ./max_model')
            os.system('mv data.txt calibration/bosch_'+str(m)+'/ratio_cal/data_'+zeroed_it+'.txt')

        dfs = []
        filenames = []

        filenames = sorted(glob.glob('calibration/bosch_'+str(m)+'/ratio_cal/data_*'))

        for filename in filenames:

            dfs.append(pd.read_csv(filename, sep=' '))

        gp_sim = np.zeros(len(w))
        gdp_sim = np.zeros(len(w))

        for i in range(len(w)):

            omega = w[i]

            tau_om = np.array(dfs[i].stress)
            gamma_om = np.array(dfs[i].strain)
            t_om = np.array(dfs[i].t)

            tau_om = np.append(0.0, tau_om)
            gamma_om = np.append(0.0, gamma_om)
            t_om = np.append(0.0, t_om)

            dt = t_om[1]

            tau_fft = fft(tau_om)
            xf = fftfreq(len(t_om),dt)[:len(t_om)//2]
            A = np.max(2./len(t_om)*np.abs(tau_fft[:len(t_om)//2]))

            def func(x,c):
                return A*np.sin(omega*x + c)

            popt, pcov = curve_fit(func,t_om,tau_om)
            tau_fit = func(t_om,popt[0])

            fig,ax = plt.subplots(1,1,figsize=(14,10))

            ax.plot(t_om,tau_om,'b',label="response")
            ax.plot(t_om,tau_fit,'r',label="fit")
            ax.set_xlabel('t [s]', fontsize=20)
            ax.set_ylabel(r"$\tau$ [Pa]", fontsize=20)
            ax.tick_params(axis='both', labelsize=20)
            ax.legend(fontsize=20)

            plt.savefig('calibration/bosch_'+str(m)+'/ratio_cal/stress_fit'+str(i)+'.png')
            plt.close(fig)

            fig,ax = plt.subplots(1,1,figsize=(14,10))

            ax.plot(xf,2.0/len(t_om)*np.abs(tau_fft[:len(t_om)//2]),'b')
            ax.set_xlabel('f [Hz]', fontsize=20)
            ax.set_ylabel(r"$\tau$ [Pa]", fontsize=20)
            ax.tick_params(axis='both', labelsize=20)
            ax.set_xlim(0,1000)

            plt.savefig('calibration/bosch_'+str(m)+'/ratio_cal/stress_fft'+str(i)+'.png')
            plt.close(fig)

            G_abs = A/np.max(gamma_om)

            phi = popt[0]

            delta = phi/(omega)

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

        rms_err_gp_sim = 100.*np.sqrt(np.sum((gp_exp-gp_sim)**2)/len(w))/(max_gp_exp-min_gp_exp)
        rms_err_gdp_sim = 100.*np.sqrt(np.sum((gdp_exp-gdp_sim)**2)/len(w))/(max_gdp_exp-min_gdp_exp)

        err = [rms_err_gp_sim,rms_err_gdp_sim]

        return np.max(err)

    #if (m==1):

    #    if (magic(2.13) > 10):

    #        res_magic = sp.optimize.minimize_scalar(magic, method='bounded', options={'maxiter':100, 'xatol':1.e-3}, bounds=(1,10))
    #else:

    #    if (magic(1) > 10):

    #        res_magic = sp.optimize.minimize_scalar(magic, method='bounded', options={'maxiter':100, 'xatol':1.e-3}, bounds=(1,10))

    #os.system('cp calibration/bosch_'+str(m)+'/user_defined_parameters_fcc.txt .')

    #for i in range(len(w)):

    #    omega = w[i]

    #    str_it = str(i)
    #    zeroed_it = str_it.zfill(4)

    #    temp = open('user_defined_parameters_fcc.txt', 'r')
    #    #temp = open('user_defined_parameters_hcp.txt', 'r')
    #    list_of_lines = temp.readlines()
    #    list_of_lines[3] = 'variable angVelConst equal '+str(omega)+'\n'
    #    temp = open('user_defined_parameters_fcc.txt', 'w')
    #    #temp = open('user_defined_parameters_hcp.txt', 'w')
    #    temp.writelines(list_of_lines)
    #    temp.close()

    #    os.system('mpirun -np 4 ./max_model')
    #    os.system('mv data.txt calibration/bosch_'+str(m)+'/data_'+zeroed_it+'.txt')

    dfs = []
    filenames = []

    filenames = sorted(glob.glob('calibration/bosch_'+str(m)+'/data_*'))

    for filename in filenames:

        dfs.append(pd.read_csv(filename, sep=' '))

    gp_sim = np.zeros(len(w))
    gdp_sim = np.zeros(len(w))

    for i in range(len(w)):

        omega = w[i]

        tau_om = np.array(dfs[i].stress)
        gamma_om = np.array(dfs[i].strain)
        t_om = np.array(dfs[i].t)

        tau_om = np.append(0.0, tau_om)
        gamma_om = np.append(0.0, gamma_om)
        t_om = np.append(0.0, t_om)

        dt = t_om[1]

        tau_fft = fft(tau_om)
        xf = fftfreq(len(t_om),dt)[:len(t_om)//2]
        A = np.max(2./len(t_om)*np.abs(tau_fft[:len(t_om)//2]))

        def func(x,c):
            return A*np.sin(omega*x + c)

        popt, pcov = curve_fit(func,t_om,tau_om)
        tau_fit = func(t_om,popt[0])

        G_abs = A/np.max(gamma_om)

        phi = popt[0]

        delta = phi/(omega)

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

    gp_sim_all.append(gp_sim)
    gdp_sim_all.append(gdp_sim)

    max_gp_sim = np.max(gp_sim)
    max_gdp_sim = np.max(gdp_sim)
    max_plot = np.max([max_gp_exp,max_gdp_exp,max_gp_sim,max_gdp_sim])*10

    min_gp_sim = np.min(gp_sim)
    min_gdp_sim = np.min(gdp_sim)
    min_plot = np.min([min_gp_exp,min_gdp_exp,min_gp_sim,min_gdp_sim])/10

    gp_avg = 1./len(gp_exp)*np.sum(gp_exp)
    gdp_avg = 1./len(gdp_exp)*np.sum(gdp_exp)

    print (max_gp_exp,min_gp_exp,max_gp_sim,min_gp_sim)

    rms_err_gp_sim_fin = 100.*np.sqrt(np.sum((gp_exp-gp_sim)**2)/len(w))/(max_gp_exp-min_gp_exp)
    rms_err_gdp_sim_fin = 100.*np.sqrt(np.sum((gdp_exp-gdp_sim)**2)/len(w))/(max_gdp_exp-min_gdp_exp)

    err_fin = [format(rms_err_gp_sim_fin,'.4f'),format(rms_err_gdp_sim_fin,'.4f')]

fig,ax = plt.subplots(2,2,sharex=True,figsize=(18,12))

ax[0,0].plot(w,gp_exp_all[0],'k-o',label="$G'$ exp", linewidth=2, markersize=8)
ax[0,0].plot(w,gp_sim_all[0],'k-.',label="$G'$ sim", linewidth=4, alpha=0.7)
ax[0,0].plot(w,gdp_exp_all[0],'r-o',label="$G''$ exp", linewidth=2, markersize=8)
ax[0,0].plot(w,gdp_sim_all[0],'r-.', label="$G''$ sim", linewidth=4, alpha=0.7)
ax[0,0].tick_params(axis='both', labelsize=30)
ax[0,0].set_title("Material 1", fontsize=30)
ax[0,0].set_xscale('log')
ax[0,0].set_yscale('log')
ax[0,0].set_ylim(1.e3,1.e5)
ax[0,0].legend(fontsize=20, loc=4)
ax[0,0].grid()

ax[0,1].plot(w,gp_exp_all[1],'k-o',label="$G'$ exp", linewidth=2, markersize=8)
ax[0,1].plot(w,gp_sim_all[1],'k-.',label="$G'$ sim", linewidth=4, alpha=0.7)
ax[0,1].plot(w,gdp_exp_all[1],'r-o',label="$G''$ exp", linewidth=2, markersize=8)
ax[0,1].plot(w,gdp_sim_all[1],'r-.', label="$G''$ sim", linewidth=4, alpha=0.7)
ax[0,1].tick_params(axis='both', labelsize=30)
ax[0,1].set_title("Material 2", fontsize=30)
ax[0,1].set_xscale('log')
ax[0,1].set_yscale('log')
ax[0,1].set_ylim(1.e2,1.e5)
ax[0,1].legend(fontsize=20, loc=4)
ax[0,1].grid()

ax[1,0].plot(w,gp_exp_all[2],'k-o',label="$G'$ exp", linewidth=2, markersize=8)
ax[1,0].plot(w,gp_sim_all[2],'k-.',label="$G'$ sim", linewidth=4, alpha=0.7)
ax[1,0].plot(w,gdp_exp_all[2],'r-o',label="$G''$ exp", linewidth=2, markersize=8)
ax[1,0].plot(w,gdp_sim_all[2],'r-.', label="$G''$ sim", linewidth=4, alpha=0.7)
ax[1,0].tick_params(axis='both', labelsize=30)
ax[1,0].set_title("Material 3", fontsize=30)
ax[1,0].set_xscale('log')
ax[1,0].set_yscale('log')
ax[1,0].set_ylim(1.e1,1.e5)
ax[1,0].legend(fontsize=20, loc=4)
ax[1,0].grid()

ax[1,1].plot(w,gp_exp_all[4],'k-o',label="$G'$ exp", linewidth=2, markersize=8)
ax[1,1].plot(w,gp_sim_all[4],'k-.',label="$G'$ sim", linewidth=4, alpha=0.7)
ax[1,1].plot(w,gdp_exp_all[4],'r-o',label="$G''$ exp", linewidth=2, markersize=8)
ax[1,1].plot(w,gdp_sim_all[4],'r-.', label="$G''$ sim", linewidth=4, alpha=0.7)
ax[1,1].tick_params(axis='both', labelsize=30)
ax[1,1].set_title("Material 4", fontsize=30)
ax[1,1].set_xscale('log')
ax[1,1].set_yscale('log')
ax[1,1].set_ylim(1.e3,1.e6)
ax[1,1].legend(fontsize=20, loc=4)
ax[1,1].grid()

fig.supxlabel('$\omega$ [rad/s]', fontsize=30)
fig.supylabel("$G', G''$ [Pa]", fontsize=30)

plt.tight_layout()

plt.savefig('calibration/sim_all_2x2.png')

