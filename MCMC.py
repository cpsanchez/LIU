#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####### create disk model and do MCMC #######

###### IMPORTS ######

from hciplot import plot_frames, plot_cubes
from matplotlib.pyplot import *
from matplotlib import pyplot as plt
import numpy as np
from packaging import version
from astropy.io import fits
from astropy.convolution import convolve
from scipy.optimize import minimize
import emcee
import os

import vip_hci as vip
vvip = vip.__version__
print("VIP version: ", vvip)
if version.parse(vvip) < version.parse("1.0.0"):
    msg = "Please upgrade your version of VIP"
    msg+= "It should be 1.0.0 or above to run this notebook."
    raise ValueError(msg)
elif version.parse(vvip) <= version.parse("1.0.3"):
    from vip_hci.conf import time_ini, timing
    from vip_hci.medsub import median_sub
    from vip_hci.metrics import cube_inject_fakedisk, ScatteredLightDisk
else:
    from vip_hci.config import time_ini, timing
    from vip_hci.fm import cube_inject_fakedisk, ScatteredLightDisk
    from vip_hci.psfsub import median_sub
# common to all versions:
from vip_hci.var import create_synth_psf


##### FUNCTIONS #####

def log_likelihood(theta, data, zerr, SNR, background):
    itilt, a, ksi0, alpha_in, alpha_out, g1, g2, weight1,e,w = theta
    
    pixel_scale=0.01225 # pixel scale in arcsec/px
    dstar= 80 # distance to the star in pc
    nx = 200 # number of pixels of your image in X
    ny = 200 # number of pixels of your image in Y
    gamma = 2.
    beta = 1
    
    a = np.abs(a)
    ksi0 = np.abs(ksi0)
    weight1 = np.abs(weight1)
    
    if weight1 > 1.:
        weight1 = 0.999
    elif weight1 < 0:
        weight1 = 0.001

    fake_disk = ScatteredLightDisk(nx=nx, ny=ny, distance=dstar,
                                itilt=itilt, omega=w, pxInArcsec=pixel_scale, pa=0,
                                density_dico={'name':'2PowerLaws','ain':alpha_in,'aout':alpha_out,
                                              'a':a,'e':e,'ksi0':ksi0,'gamma':gamma,'beta':beta},
                                spf_dico={'name':'DoubleHG', 'g':[g1,g2], 'weight':weight1, 'polar':False},
                                flux_max=SNR*np.nanmax(background)/div)
    model_map = fake_disk.compute_scattered_light()
    model_convolved = convolve(model_map, PSF)

    sigma2 = zerr**2
    
    return -0.5 * np.nansum((data - model_convolved) ** 2 / sigma2 )

def log_prior(theta):

    inc = theta[0]
    a = theta[1]
    ksi0 = theta[2]
    ain = theta[3]
    aout = theta[4]
    g1 = theta[5]
    g2 = theta[6]
    alpha = theta[7]
    e = theta[8]
    w = theta[9]
    
    if (inc < 0.0 or inc > 87.6):
        return -np.inf

    if (a < 20 or a > 130):  
        return -np.inf
        
    if (ksi0/a < 0.005 or ksi0/a > 0.2):  
        return -np.inf

    if (ain < 1 or aout > 30):
        return -np.inf

    if (aout < -30 or aout > -1):
        return -np.inf

    if (g1 < 0.0005 or g1 > 0.9999):
        return -np.inf

    if (g2 < -0.9999 or g2 > -0.0005):
        return -np.inf

    if (alpha < 0.0005 or alpha > 0.9999):
        return -np.inf

    if (e < 0.0005 or e > 0.9999):
        return -np.inf

    if (w < -5 or w > 95.):
        return -np.inf
        
    # otherwise ...

    return 0.0

def log_probability(theta, data, zerr, SNR, background):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    print('here')
    return lp + log_likelihood(theta, data, zerr, SNR, background)



##### NOISE + PSF #####

filename = "FITS_files/disk_LIU_background.fits"
with fits.open(filename) as hdulist: 
    header = hdulist[0].header
    background = hdulist[0].data

filename = "FITS_files/disk_LIU_noisemap.fits"
with fits.open(filename) as hdulist: 
    header = hdulist[0].header
    noisemap = hdulist[0].data

filename = "FITS_files/disk_LIU_PSF_convolution.fits"
with fits.open(filename) as hdulist: 
    header = hdulist[0].header
    PSF = hdulist[0].data
PSF = PSF[:-1,:-1]


##### DISK PARAMETERS #####

pixel_scale=0.01225 # pixel scale in arcsec/px
dstar= 80 # distance to the star in pc
nx = 200 # number of pixels of your image in X
ny = 200 # number of pixels of your image in Y

rm = 15.1*pixel_scale*dstar/2
itilt = 30. # inclination of your disk in degrees
a = 8*rm # semimajoraxis of the disk in au
ksi0 = 0.03*a # rerence scale height at the semi-major axis of the disk*a
gamma = 2. # exponant of the vertical exponential decay
alpha_in = 12
alpha_out = -12
beta = 1

g1=0.677
g2=-0.042
weight1=0.742

SNR = 10.
div = 2.

reset = 1 # 1 resets backend, 0 continues from last entry of backend

e = 0.05
w = 0

##### DISK MODEL #####

fake_disk1 = ScatteredLightDisk(nx=nx, ny=ny, distance=dstar,
                                itilt=itilt, omega=w, pxInArcsec=pixel_scale, pa=0,
                                density_dico={'name':'2PowerLaws','ain':alpha_in,'aout':alpha_out,
                                              'a':a,'e':e,'ksi0':ksi0,'gamma':gamma,'beta':beta},
                                spf_dico={'name':'DoubleHG', 'g':[g1,g2], 'weight':weight1, 'polar':False},
                                flux_max=SNR*np.nanmax(background)/div)

fake_disk1_map = fake_disk1.compute_scattered_light()
fake_disk1_conv = convolve(fake_disk1_map,PSF)
fake_disk1_conv_noise = fake_disk1_conv + background



##### ML 1st GUESS #####

np.random.seed(42)
nll = lambda *args: -log_likelihood(*args)
initial = np.array([itilt, a, ksi0, alpha_in, alpha_out, g1, g2, weight1,e,w]) + 0.1 * np.random.randn(10)
soln = minimize(nll, initial, args=(fake_disk1_conv_noise, noisemap, SNR, background))
itilt_ml, a_ml, ksi0_ml, alpha_in_ml, alpha_out_ml, g1_ml, g2_ml, weight1_ml,e_ml,w_ml = soln.x

print("Maximum likelihood estimates:")
print("Inclination = {0:.3f}".format(itilt_ml))
print("Semimajor axis = {0:.3f}".format(np.abs(a_ml)))
print("Reference scale height = {0:.3f}".format(np.abs(ksi0_ml)))
print("alpha_in = {0:.3f}".format(alpha_in_ml))
print("alpha_out = {0:.3f}".format(alpha_out_ml))
print("g1 = {0:.3f}".format(g1_ml))
print("g2 = {0:.3f}".format(g2_ml))
print("Weight = {0:.3f}".format(np.abs(weight1_ml)))
print("e = {0:.3f}".format(e_ml))
print("w = {0:.3f}".format(w_ml))



##### BACKEND #####

pos = soln.x + 1e-4 * np.random.randn(46, 10)
nwalkers, ndim = pos.shape

filename_backend = os.path.join("results_MCMC_sphere/disk_LIU_backend_file_mcmc_i"+str(int(itilt))+"_a"+str(int(a/rm))+"_ksi0"+str(int(ksi0/a*100))+"_SNR"+str(int(SNR))+"_e"+str(int(e*100))+"_w"+str(w)+".h5")
backend_ini = emcee.backends.HDFBackend(filename_backend)
if reset:
    backend_ini.reset(nwalkers, ndim)



##### MCMC #####

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(fake_disk1_conv_noise, noisemap, SNR, background), backend=backend_ini
)

sampler.run_mcmc(pos, 1000)
print('Finished')