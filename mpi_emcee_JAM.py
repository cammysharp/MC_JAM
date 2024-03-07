#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 13:56:07 2024

@author: 48105686
"""


import numpy
import emcee
import pickle
from jampy.jam_axi_proj import jam_axi_proj
import sys
from schwimmbad import MPIPool
from mgefit.mge_fit_1d import mge_fit_1d
from pafit.fit_kinematic_pa import fit_kinematic_pa

import time
import h5py

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

from os import path

pathname='/fred/oz059/cammy/MC_JAM/NGC2974/'

def dark_halo_gNFW_MGE(gamma,lg_rbreak,lg_rho_s):
    start=time.time()
    rbreak,rho_s=10**lg_rbreak,10**lg_rho_s
    rbreak /= pc
    
    n = 300     # Number of values to sample the gNFW profile for the MGE fit
    r = np.geomspace(1, rbreak*10, n)
    
    
    
    rho=(rho_s)/(((r/rbreak)**gamma)*((1+(r/rbreak)**2)**((3-gamma)/2.)))
    
    #rho = (r/rbreak)**gamma * (0.5 + 0.5*r/rbreak)**(-gamma - 3)  # rho=1 at r=rbreak
    
    
    m = mge_fit_1d(r, rho, ngauss=20, quiet=1, plot=0)
    #plt.pause(1)
    surf_dm, sigma_dm = m.sol
    
    
    
    qobs_dm= np.ones_like(surf_dm)
    
    #surf_pot_dm=surf_dm/(sigma_dm*np.sqrt(2*np.pi))
    
    end = time.time()
    serial_time = end - start
    #print("dark mge took {0:.1f} seconds".format(serial_time))
    return surf_dm,sigma_dm, qobs_dm

def total_mass_mge(gamma, lg_rbreak,lg_rho_s, alpha,inc):
    
    start=time.time()
    
    surf_dm,sigma_dm,qobs_dm= dark_halo_gNFW_MGE(gamma,lg_rbreak,lg_rho_s)
    #print(surf_lum,'wjuehfuheu')
    scaled_surf_mass = surf_mass*alpha
    
    if any(np.isnan(surf_lum))==1 or any(np.isnan(surf_dm))==1:
        print('shit',surf_lum,surf_dm)
    
    surf_pot = np.append(scaled_surf_mass,surf_dm)
    sigma_pot= np.append(sigma_lum,sigma_dm)
    qobs_pot= np.append(qobs_lum,qobs_dm)
    
    
    #resets Surf lum as it is mulitplied by alpha and then does not get
    #Set again before running the next MC run
    #surf_lum=get_lum_mge()[0]
    end = time.time()
    serial_time = end - start
    #print("total mass mge took {0:.1f} seconds".format(serial_time))
    
    return surf_pot,sigma_pot,qobs_pot

def JAM_model(pars):
    
    start=time.time()
    #defines the parameters used
    alpha,lg_rbreak,lg_rho_s,gamma,lg_mbh = pars
    
    #print(pars,'check')
    #print(rms,'check')
    inc=60
    mbh=10**lg_mbh
    #Need to fit the dark matter density fit mge 1d fit to add the surface 
    #density to the mass mge of the luminous part. The dark matter density is 
    #paramerterised by lg_rbreak, lg_rho_s and gamma.
    
    #total mass mge appends the dark mge's to the mass mge's to get a 1D array
    #of all the mge's
    

    
    #surf_lum *= alpha
    
    surf_pot,sigma_pot,qobs_pot = total_mass_mge(gamma,lg_rbreak,lg_rho_s, alpha,inc)
    #Note: In the example this is where surf pot is scaled by M/L including the dark matter mge's,
    #This is not fully explained but is said to keep the fraction of the dark matter the same, this
    #also scales Mbh
    
    
    #print(surf_lum)
    #Runs the actual JAM model to get the model RMS out (other moments can be selected)
    jam = jam_axi_proj(surf_lum, sigma_lum, qobs_lum, surf_pot, sigma_pot, qobs_pot,
                       inc, mbh, distance, xbin, ybin, plot=0, pixsize=0, quiet=1,
                       align='cyl', data=rms, errors=erms, ml=None,goodbins=goodbins,interp=False)
    
    
    
    Vrms_model=jam.model
    
    if any(np.isnan(Vrms_model))==1:
        print('fuck',surf_lum,Vrms_model,erms,pars)
    
    end = time.time()
    serial_time = end - start
    #print("JAM_model took {0:.1f} seconds".format(serial_time))
    return Vrms_model

def lnlike(pars):
    start=time.time()
    
    Vrms_model=JAM_model(pars)
    
    lnlike= -0.5*sum(((rms-Vrms_model)/erms)**2)
    
    if np.isnan(lnlike)==1:
        print('fuck',surf_lum,Vrms_model,pars)
    end = time.time()
    serial_time = end - start
    #print("lnlike took {0:.1f} seconds".format(serial_time))
    
    return lnlike

def lnprior(pars):
    start=time.time()
    check=np.zeros(len(pars))
    #alpha,lg_rbreak,lg_rho_s,gamma,lg_mbh = pars
    
    for i in range(len(pars)):
        if pars[i]>=bounds[0][i] and pars[i]<=bounds[1][i]:
            check[i]=0
        else:
            check[i]=1
    
    if check.any()==1:
        lp=-np.inf
        #print('shitting bastard')
    else:
        lp=0.0
    
    end = time.time()
    serial_time = end - start
    #print("lnprior took {0:.1f} seconds".format(serial_time))
    return lp


def lnprob(pars):
    #print(surf_lum,'check lnlike')
    start = time.time()
    lp= lnprior(pars)
    if lp != 0.0:
        #print('fuuuuckkk')
        return -np.inf
    else:
        return lp + lnlike(pars)
    end = time.time()
    serial_time = end - start
    #print("lnprob took {0:.1f} seconds".format(serial_time))


def get_fitz():
    hdul = fits.open(pathname+"PXF_bin_MS_NGC2974_r5_idl.fits.gz")
    hdul.info()
    #print(hdul[1].header)
    data=hdul[1].data
    xbin,ybin, V, sigma,flux,Verr, serr = hdul[1].data['XS'],hdul[1].data['YS'],hdul[1].data['VPXF'],hdul[1].data['SPXF'],hdul[1].data['FLUX'],hdul[1].data['EVPXF'],hdul[1].data['ESPXF']
    #print(xbin,ybin,sigma,flux,Verr,V,serr)
    Vhel,distance=1887.,20.89
    rms=np.sqrt((V-Vhel)**2+sigma**2)
    erms=(1./rms)*np.sqrt((V*Verr)**2+(sigma*serr)**2)
    
    print('fits done')
    return xbin,ybin,rms,erms,distance,V

def get_mass_lum():
    surf_lum, sigma_lum, qobs_lum, surf_mass=np.load(pathname+"mgeparameters_NGC2974.npy")
   
    return surf_lum, sigma_lum, qobs_lum, surf_mass

def jam_lnprob(pars):
    """
    Args:
    pars: the starting point for the emcee process

    Outputs:
    chi2: chi-squared value for the given inputs

    """

    alpha,lg_rbreak,lg_rho_s,gamma,lg_mbh = pars

    check_alpha = bounds[0][0] < alpha < bounds[1][0]
    check_lg_rbreak =bounds[0][1] < lg_rbreak < bounds[1][1]
    check_lg_rho_s=bounds[0][2] < lg_rho_s < bounds[1][2]
    check_gamma= bounds[0][3] < gamma < bounds[1][3]
    check_lg_mbh= bounds[0][4] < lg_mbh < bounds[1][4]

    if check_alpha and check_lg_rbreak and check_lg_rho_s and check_gamma and check_lg_mbh:
        start=time.time()
        #defines the parameters us
        
        #print(pars,'check')
        #print(rms,'check')
        inc=60
        mbh=10**lg_mbh
        #Need to fit the dark matter density fit mge 1d fit to add the surface 
        #density to the mass mge of the luminous part. The dark matter density is 
        #paramerterised by lg_rbreak, lg_rho_s and gamma.
        
        #total mass mge appends the dark mge's to the mass mge's to get a 1D array
        #of all the mge's
        

        
        #surf_lum *= alpha
        
        surf_pot,sigma_pot,qobs_pot = total_mass_mge(gamma,lg_rbreak,lg_rho_s, alpha,inc)
        #Note: In the example this is where surf pot is scaled by M/L including the dark matter mge's,
        #This is not fully explained but is said to keep the fraction of the dark matter the same, this
        #also scales Mbh
        
        
        #print(surf_lum)
        #Runs the actual JAM model to get the model RMS out (other moments can be selected)
        jam = jam_axi_proj(surf_lum, sigma_lum, qobs_lum, surf_pot, sigma_pot, qobs_pot,
                           inc, mbh, distance, xbin, ybin, plot=0, pixsize=0, quiet=1,
                           align='cyl', data=rms, errors=erms, ml=None,goodbins=goodbins,interp=False)
        
        
        
        Vrms_model=jam.model
        
        if any(np.isnan(Vrms_model))==1:
            print('fuck',surf_lum,Vrms_model,erms,pars)
        
        end = time.time()
        serial_time = end - start
        #print("JAM_model took {0:.1f} seconds".format(serial_time))
        return Vrms_model

    else:
        chi2 = -numpy.inf
        print('yes')

    return chi2



def transform(x,y,angle):
    theta=np.radians(angle)
    xm=x*np.cos(theta)-y*np.sin(theta)
    ym=x*np.sin(theta)+y*np.cos(theta)
    return xm,ym

print('start')

xbin,ybin,rms,erms,distance,V=get_fitz()
#print('xbin and that:',xbin,ybin,rms,erms,distance,V)

surf_lum,sigma_lum,qobs_lum,surf_mass=get_mass_lum()
#print('surf lum and that:',surf_lum,sigma_lum,qobs_lum,surf_mass)


pixsize = 0.8 # spaxel size in arcsec (before Voronoi binning)
sigmapsf = 2.6/2.355      # sigma PSF in arcsec (=FWHM/2.355)
normpsf = 1
pc = distance*np.pi/0.648




#labels={'alpha','rbreak','density','gamma','Mbh'}




labels=[r"$alpha$",r"$rbreak$",r"$density$",r"$\gamma$",r"$Mbh$"]

vel_corr=V-np.median(V)

angBest, angErr, vSyst = fit_kinematic_pa(xbin, ybin, vel_corr, debug=False, plot=False,quiet=1)

xbin,ybin = transform(xbin,ybin,angBest)
goodbins = np.isfinite(xbin)

print(angBest)
loc=0
print('finding 0')

for i in range(0,len(xbin)):
    if xbin[i]==0 and ybin[i]==0:
        print('0 at',i)
    else:
        #print(i)
        loc = np.append(loc,i)

print(loc)
xbin=xbin[loc]
ybin=ybin[loc]
rms=rms[loc]
erms=erms[loc]


kwargs = {'surf_lum': surf_lum, 'sigma_lum': sigma_lum, 'qobs_lum': qobs_lum,
              'distance': distance, 'xbin': xbin, 'ybin': ybin, 'sigmapsf': sigmapsf,
              'normpsf': normpsf, 'rms': rms, 'erms': erms, 'pixsize': pixsize}

args=[surf_mass,surf_lum, sigma_lum, qobs_lum, distance,
              xbin, ybin, sigmapsf, normpsf, 
              rms, erms, pixsize,goodbins]



nsteps = 1000

nwalkers = 30



alpha0 = 1.8     # 
lg_rbreak0 = 4.7   # 
lg_rho_s = -2.25  # 
gamma0 = 0.1      #
lg_mbh0= 9.5     #


initial = np.asarray([alpha0, lg_rbreak0, lg_rho_s, gamma0, lg_mbh0])



bounds = np.asarray([[1., 3., -5., 0.,6.], [2., 5., -1,1.1, 11.]])

timer_start=time.time()
jam_lnprob(initial)
timer_end=time.time()

print('one ln_jam model production takes:',timer_end-timer_start,'s')




timer_start=time.time()
lnprob(initial)
timer_end=time.time()

print('one full jam model production takes:',timer_end-timer_start,'s')
ndim = len(initial)

p0 = [initial + 0.2*numpy.random.randn(ndim) for i in range(nwalkers)]
    
filename=pathname+'mc_JAM_out_NGC2974.h5'
          
with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool,backend=backend)
    sampler.run_mcmc(p0, nsteps, progress=True)

samples = sampler.get_chain()



samples = sampler.get_chain(flat=True)

plt.hist(samples[:, 0], 100, color="k", histtype="step")
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$p(\theta_1)$")
plt.gca().set_yticks([]);

plt.savefig('test.png')
plt.clf()



#hf= h5py.File(/fred/oz059/cammy/MC_JAM/NGC2974/+'test_out.h5','w')
#hf.create_dataset('sampler',data=sampler)

