#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 11:41:09 2024

@author: 48105686
"""


import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from os import path

from plotbin.display_pixels import display_pixels

import mgefit
from mgefit.find_galaxy import find_galaxy
from mgefit.mge_fit_1d import mge_fit_1d
from mgefit.sectors_photometry import sectors_photometry
from mgefit.mge_fit_sectors import mge_fit_sectors
from mgefit.mge_print_contours import mge_print_contours
from mgefit.mge_fit_sectors_twist import mge_fit_sectors_twist
from mgefit.sectors_photometry_twist import sectors_photometry_twist
from mgefit.mge_print_contours_twist import mge_print_contours_twist
from mgefit.mge_fit_sectors_regularized import mge_fit_sectors_regularized

import scipy.optimize as opt



def dist_circle(xc, yc, s):
    """
    Returns an array in which the value of each element is its distance from
    a specified center. Useful for masking inside a circular aperture.

    The (xc, yc) coordinates are the ones one can read on the figure axes
    e.g. when plotting the result of my find_galaxy() procedure.

    """
    x, y = np.ogrid[-yc:s[0] - yc, -xc:s[1] - xc]   # note yc before xc
    rad = np.sqrt(x**2 + y**2)

    return rad


hdu = fits.open('/Users/48105686/Documents/MC_JAM/data/NGC2974/cutout_rings.v3.skycell.1268.000.stk.r.unconv.fits')
hdu.info()
print(hdu[0].header)
pixelsize=0.25 #arcsecs
data=hdu[0].data
ra=hdu[0].header['CRVAL1']+(np.arange(hdu[0].header['NAXIS1'])-hdu[0].header['CRPIX1'])*hdu[0].header['CDELT1']
dec=hdu[0].header['CRVAL2']+(np.arange(hdu[0].header['NAXIS1'])-hdu[0].header['CRPIX2'])*hdu[0].header['CDELT2']
RA,DEC=np.meshgrid(ra,dec)

img=2.5*data
img[np.isnan(img)]=0
print(-2.5*np.log10(2.59437E-7/3631))
skylev = 8*(-2.5*np.log10(2.59437E-7/3631))   # counts/pixel
img -= skylev   # subtract sky
scale = 0.25  # arcsec/pixel
minlevel = 100.5  # counts/pixel
ngauss = 12

exptime=572
gain=1.05718


r = dist_circle(490, 255, img.shape)  # distance matrix from (216, 542)
mask1 = r > 60 
mask1=~mask1
r = dist_circle(520, 470, img.shape)  # distance matrix from (216, 542)
mask2 = r > 35
mask2=~mask2
r = dist_circle(120, 305, img.shape)  # distance matrix from (216, 542)
mask3 = r > 35
mask3=~mask3
r = dist_circle(115, 525, img.shape)  # distance matrix from (216, 542)
mask4 = r > 35 
mask4=~mask4
mask=~(mask1+mask2+mask3+mask4)
print(np.all(mask))


    # Here we use an accurate four gaussians MGE PSF for
    # the HST/WFPC2/F814W filter, taken from Table 3 of
    # Cappellari et al. (2002, ApJ, 578, 787)

#sigmapsf = [0.494, 1.44, 4.71, 13.4]      # In PC1 pixels
#normpsf = [0.294, 0.559, 0.0813, 0.0657]  # total(normpsf)=1

    # Here we use FIND_GALAXY directly inside the procedure. Usually you may want
    # to experiment with different values of the FRACTION keyword, before adopting
    # given values of Eps, Ang, Xc, Yc.
plt.clf()
f = find_galaxy(img, fraction=0.04, plot=1)
plt.pause(1)  # Allow plot to appear on the screen

    # Perform galaxy photometry
plt.clf()
s = sectors_photometry(img, f.eps, f.theta, f.xpeak, f.ypeak,
                           minlevel=minlevel, plot=1,mask=mask)
plt.pause(1)  # Allow plot to appear on the screen

    # Do the actual MGE fit
    # *********************** IMPORTANT ***********************************
    # For the final publication-quality MGE fit one should include the line
    # "from mge_fit_sectors_regularized import mge_fit_sectors_regularized"
    # at the top of this file, rename mge_fit_sectors() into
    # mge_fit_sectors_regularized() and re-run the procedure.
    # See the documentation of mge_fit_sectors_regularized for details.
    # *********************************************************************
plt.clf()
m = mge_fit_sectors_regularized(s.radius, s.angle, s.counts, f.eps,
                        ngauss=ngauss,
                        scale=scale, plot=1, bulge_disk=0, linear=0)
plt.pause(1)  # Allow plot to appear on the screen

    # Show contour plots of the results
plt.clf()
plt.subplot(121)
mge_print_contours(img, f.theta, f.xpeak, f.ypeak, m.sol, scale=scale,
                       binning=9,
                       minlevel=minlevel,mask=mask)

    # Extract the central part of the image to plot at high resolution.
    # The MGE is centered to fractional pixel accuracy to ease visual comparson.

n = 50
img = img[f.xpeak-n:f.xpeak+n, f.ypeak-n:f.ypeak+n]
xc, yc = n - f.xpeak + f.xmed, n - f.ypeak + f.ymed
plt.subplot(122)
mge_print_contours(img, f.theta, xc, yc, m.sol, scale=scale)
plt.pause(1)  # Allow plot to appear on the screen

zp=25
Ar=0.124
solmagr=4.67

totalcounts=m.sol[0]
sigmapixels=m.sol[1]
qobs=m.sol[2]

C0=totalcounts/(2.*np.pi*(sigmapixels**2)*qobs)

mewr=zp-2.5*np.log(C0/exptime*(scale)**2)-Ar

I=((64800/np.pi)**2)*10**(0.4*(solmagr-mewr))

sigma=scale*sigmapixels

for i in range(len(I)):
    print(I[i],sigma[i],qobs[i])
    
hdu=fits.open('/Users/48105686/Documents/MC_JAM/data/NGC2974/Poci2017_ml_maps/sfh_compr_NGC2974_E3D.fits')
hdu.info()
hdu[1].header
x1,y1=hdu[1].data['XBIN'],hdu[1].data['YBIN']
ML=hdu[1].data['MLSTARR']

mask=np.where((ML>0) & (np.sqrt(x1**2+y1**2)>0))


plt.scatter(np.sqrt(x1[mask]**2+y1[mask]**2),ML[mask])
plt.show()
x1,y1,ML=x1[mask],y1[mask],ML[mask]/1.7



#print(np.sqrt(x1**2+y1**2))

plt.scatter(np.sqrt(x1**2+y1**2),ML)

I=yangL=np.asarray([4276.01,7782.37,2853.55,3171.34,220.000,970.160,252.150])
sigma=yangsigma=np.asarray([0.54153,0.88097,1.44526,3.81993,6.64704,10.7437,28.4453])
qobs=yangqobs=np.asarray([0.83144,0.82501 ,0.94271 ,0.67267 ,0.99990 ,0.55375,0.61238])
yangmass=np.asarray([16208.47,26366.23,13148.71,11329.50,1966.17, 2890.09,778.71])

print(yangmass/yangL)

def gamma(xy,*params):
    x,y=xy
    g=np.zeros_like(x)
    bottom=np.zeros_like(x)
    
    for i in range(I.size):
        amp1=I[i]/(2*np.pi*(sigma[i]**2)*qobs[i])
        bottom+=amp1*np.exp((-0.5/(sigma[i]**2))*((x**2)+(y/qobs[i])**2))
    
    
    for i in range(I.size):
        amp0=params[i]*I[i]/(2*np.pi*(sigma[i]**2)*qobs[i])
        g=g+amp0*np.exp((-0.5/(sigma[i]**2))*((x**2)+(y/qobs[i])**2))/bottom
    g=np.reshape(g,-1)
    return g

guess=[1,1,1,1,1,1,1]

b=[[0,0,0,0,0,0,0],[10,10,10,10,10,10,10]]

print(len(guess),len(b[0]),len(b[1]))



def yanggamma(xy,*params):
    x,y=xy
    g=np.zeros_like(x)
    bottom=np.zeros_like(x)
    
    for i in range(yangL.size):
        amp1=yangL[i]/(2*np.pi*(yangsigma[i]**2)*yangqobs[i])
        bottom+=amp1*np.exp((-0.5/(yangsigma[i]**2))*((x**2)+(y/yangqobs[i])**2))
    
    
    for i in range(yangL.size):
        amp0=params[i]*yangL[i]/(2*np.pi*(yangsigma[i]**2)*yangqobs[i])
        g=g+amp0*np.exp((-0.5/(yangsigma[i]**2))*((x**2)+(y/yangqobs[i])**2))/bottom
    g=np.reshape(g,-1)
    return g


#xdata=np.linspace()
ML=np.reshape(ML,-1)
popt, pcov = opt.curve_fit(gamma, (x1, y1), ML,p0=guess,bounds=b)
print(popt)

xsmooth,ysmooth=np.linspace(min(np.unique(x1)),max(np.unique(x1)),500),np.linspace(min(np.unique(x1)),max(np.unique(x1)),500)
plt.plot(np.sqrt(xsmooth**2+ysmooth**2),gamma((xsmooth,ysmooth),*popt),'r')
plt.plot(np.sqrt(xsmooth**2+ysmooth**2),yanggamma((xsmooth,ysmooth),*(yangmass/yangL)),'k')
plt.scatter(np.sqrt(x1**2+y1**2),ML)


for i in range(I.size):
    print(I[i],sigma[i],qobs[i],popt[i]*I[i])
    
    
np.savetxt('mgeparameters_2974.txt', [I,sigma,qobs,popt*I])


