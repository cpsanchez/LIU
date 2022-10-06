#!/usr/bin/env python3
# -*- coding: utf-8 -*-

############   IMPORTS   ############
from hciplot import plot_frames, plot_cubes
from matplotlib.pyplot import *
from matplotlib import pyplot as plt
import numpy as np
from packaging import version
from astropy.io import fits
from astropy.convolution import convolve
from scipy.optimize import minimize
import emcee
from IPython.display import display, Math
import corner

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

def log_likelihood(theta, data, zerr):
    itilt, a, ksi0, alpha_in, alpha_out, g1, g2, weight1 = theta
    
    pixel_scale=0.01225 # pixel scale in arcsec/px
    dstar= 320 # distance to the star in pc
    nx = 50 # number of pixels of your image in X
    ny = 50 # number of pixels of your image in Y
    gamma = 2.
    beta = 1
    
    a = np.abs(a)
    ksi0 = np.abs(ksi0)
    weight1 = np.abs(weight1)
    
    if weight1 > 1:
        weight1 = 1

    fake_disk = ScatteredLightDisk(nx=nx, ny=ny, distance=dstar,
                                itilt=itilt, omega=0, pxInArcsec=pixel_scale, pa=0,
                                density_dico={'name':'2PowerLaws','ain':alpha_in,'aout':alpha_out,
                                              'a':a,'e':0.0,'ksi0':ksi0,'gamma':gamma,'beta':beta},
                                spf_dico={'name':'DoubleHG', 'g':[g1,g2], 'weight':weight1, 'polar':False},
                                flux_max=1.)
    model_map = fake_disk.compute_scattered_light()
    # convolve here too
    sigma2 = zerr**2
    
    return -0.5 * np.sum((data - model_map) ** 2 / sigma2 )
    
def log_likelihood1(theta, data, zerr):
    itilt, a, ksi0, alpha_in, alpha_out, g1, g2, weight1 = theta
    
    pixel_scale=0.01225 # pixel scale in arcsec/px
    dstar= 320 # distance to the star in pc
    nx = 50 # number of pixels of your image in X
    ny = 50 # number of pixels of your image in Y
    gamma = 2.
    beta = 1

    fake_disk = ScatteredLightDisk(nx=nx, ny=ny, distance=dstar,
                                itilt=itilt, omega=0, pxInArcsec=pixel_scale, pa=0,
                                density_dico={'name':'2PowerLaws','ain':alpha_in,'aout':alpha_out,
                                              'a':a,'e':0.0,'ksi0':ksi0,'gamma':gamma,'beta':beta},
                                spf_dico={'name':'DoubleHG', 'g':[g1,g2], 'weight':weight1, 'polar':False},
                                flux_max=1.)
    model_map = fake_disk.compute_scattered_light()
    # convolve here too
    sigma2 = zerr**2
    
    return -0.5 * np.sum((data - model_map) ** 2 / sigma2 )
    
def log_prior(theta):
    itilt, a, ksi0, alpha_in, alpha_out, g1, g2, weight1 = theta
    if 0 < itilt < 90.0 and 0.0 < a < 100.0 and 0.0 < ksi0 < 100.0 and -20.0 < alpha_in < 20.0 and -20.0 < alpha_out < 20.0 and 0.0 < g1 < 1.0 and -1.0 < g2 < 0.0 and 0.0 < weight1 < 1.0:
        return 0.0
    return -np.inf

def log_probability(theta, data, zerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood1(theta, data, zerr)

if __name__ == '__main__':

    ############   DISK   ############
    pixel_scale=0.01225 # pixel scale in arcsec/px
    dstar= 320 # distance to the star in pc
    nx = 50 # number of pixels of your image in X
    ny = 50 # number of pixels of your image in Y
    
    itilt = 45. # inclination of your disk in degrees
    a = 70. # semimajoraxis of the disk in au
    ksi0 = 3. # reference scale height at the semi-major axis of the disk
    gamma = 2. # exponant of the vertical exponential decay
    alpha_in = 12
    alpha_out = -12
    beta = 1
    
    # Double HG phase function scattering
    g1=0.7
    g2=-0.2
    weight1=0.665
    
    fake_disk1 = ScatteredLightDisk(nx=nx, ny=ny, distance=dstar,
                                itilt=itilt, omega=0, pxInArcsec=pixel_scale, pa=0,
                                density_dico={'name':'2PowerLaws','ain':alpha_in,'aout':alpha_out,
                                              'a':a,'e':0.0,'ksi0':ksi0,'gamma':gamma,'beta':beta},
                                spf_dico={'name':'DoubleHG', 'g':[g1,g2], 'weight':weight1, 'polar':False},
                                flux_max=1.)
                               
    # Plot phase function 
    fake_disk1.phase_function.plot_phase_function()
    
    fake_disk1_map = fake_disk1.compute_scattered_light()

    fake_disk1.print_info()
    
    filename = "../Sphere_hr4796_PSF_convolution.fits"
    fits.info(filename)
    with fits.open(filename) as hdulist: 
        header = hdulist[0].header
        PSF = hdulist[0].data
    
    # Plot PSF
    plot_frames(PSF, size_factor=4)
    fake_disk1_conv = convolve(fake_disk1_map,PSF)
    
    # Noise
    dim1 = fake_disk1_conv.shape[0]
    dim2 = fake_disk1_conv.shape[1]

    mean = 0 
    var = 0.01
    sigma = np.sqrt(var)
    err = 0.1 + 0.3 * np.random.normal(loc=mean,scale=sigma,size=(dim1,dim2))
    fake_disk1_conv_noise = fake_disk1_conv + err
    
    # Plot model
    plot_frames(fake_disk1_conv_noise, grid=False, size_factor=6)
    
    ############   MAX LIKELIHOOD   ############
    np.random.seed(42)
    nll = lambda *args: -log_likelihood(*args)
    initial = np.array([itilt, a, ksi0, alpha_in, alpha_out, g1, g2, weight1]) + 0.1 * np.random.randn(8)
    soln = minimize(nll, initial, args=(fake_disk1_conv_noise, err))
    itilt_ml, a_ml, ksi0_ml, alpha_in_ml, alpha_out_ml, g1_ml, g2_ml, weight1_ml = soln.x
    
    print("\n\nMaximum likelihood estimates:")
    print("Inclination = {0:.3f}".format(itilt_ml))
    print("Semimajor axis = {0:.3f}".format(np.abs(a_ml)))
    print("Reference scale height = {0:.3f}".format(np.abs(ksi0_ml)))
    print("alpha_in = {0:.3f}".format(alpha_in_ml))
    print("alpha_out = {0:.3f}".format(alpha_out_ml))
    print("g1 = {0:.3f}".format(g1_ml))
    print("g2 = {0:.3f}".format(g2_ml))
    print("Weight = {0:.3f}\n\n".format(np.abs(weight1_ml)))
    
    ############   MCMC   ############
    pos = soln.x + 1e-4 * np.random.randn(32, 8)
    pos[:,2] = np.abs(pos[:,2])
    nwalkers, ndim = pos.shape
    
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(fake_disk1_conv_noise, err)
    )
    sampler.run_mcmc(pos, 5000, progress=True);
    
    fig, axes = plt.subplots(8, figsize=(10, 8), sharex=True)
    samples = sampler.get_chain()
    labels = ["itilt", "a", "ksi0", "alpha_in", "alpha_out", "g1", "g2", "weight1"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
        
    axes[-1].set_xlabel("step number");
    
    flat_samples = sampler.get_chain(discard=500, thin=15, flat=True)
    
    labels = ["itilt", "a", "ksi0", "alpha_{in}", "alpha_{out}", "g_1", "g_2", "weight1"]
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        display(Math(txt))
        
    fig = corner.corner(
        flat_samples, labels=labels, truths=[itilt,a,ksi0,alpha_in,alpha_out,g1,g2,weight1]
    );

