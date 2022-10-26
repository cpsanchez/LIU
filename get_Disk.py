#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####### Generate disk model using VIP and save it as .fits #######

import numpy as np
from vip_hci.fm import ScatteredLightDisk
from astropy.io import fits
from astropy.convolution import convolve
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def get_Noise(emp_sph,PSF):

   # measure photon noise:
   # for HIP 57013  : http://simbad.u-strasbg.fr/simbad/sim-id?Ident=HIP+57313
   #  H = 5.078 : http://astroweb.case.edu/ssm/ASTR620/mags.html#flux
   # so flux in Jy is
   # f_Jy = 1080 * 10^(-0.4*H)=2.75e-10 = 10.1 Jy
   
   # https://www.eso.org/sci/facilities/paranal/instruments/sphere/inst/filters.html
   
   # dlambda/lambda = 0.032 in H3
   # VLT mirror is pi*(4)^2 = 50 m2
   
   # f_phot= 10.1 Jy * 1.51e7 * 0.032 * 50 = 2.4 10^8 photons sec^-1
   
   # We take a loss of flux of ~0.3 du to going though sphere (optics trghouput etc)
   # f_phot_pup= 10.1 Jy * 1.51e7 * 0.032 * 50 = 1 10^8 photons sec^-1
   
   f_phot_pup = 1e8 # photons/s
   
   # We multiply image by 60s
   PSF_photons = PSF/np.sum(PSF)*f_phot_pup*60 # photons
   emp_sph_photons = emp_sph/np.sum(PSF)*f_phot_pup*60 # photons

   # Ensure same noise
   np.random.seed(53)
   noise = (np.random.poisson(emp_sph_photons) - emp_sph_photons)/np.max(PSF_photons)
   noise = np.where(noise == 0,1e-12,noise)
   plt.figure()
   plt.imshow(noise)
   plt.colorbar(label='Contrast')
   plt.savefig('Images/noise_photons_contrast.jpg')
   plt.show()

   hdu = fits.PrimaryHDU(noise)
   hdu.writeto('FITS_files/disk_LIU_noisemap.fits',overwrite=True)
   
   return noise
   
def getDisk(theta, noise, SNR):

   # Image parameters
   pixel_scale = 0.01225 # pixel scale in arcsec/px
   dstar = 80 # distance to the star in pc
   nx = 200 # number of pixels of your image in X
   ny = 200 # number of pixels of your image in Y

   # Geometrical and Scattering (DHG) parameters
   itilt, a, ksi0, alpha_in, alpha_out, g1, g2, alpha = theta
   
   if alpha > 1.0:
   	alpha = 0.999
   elif alpha < 0.0:
        alpha = 0.001

   # Disk model
   model = ScatteredLightDisk(nx=nx,
                               ny=ny,
                               distance=dstar,
                               itilt=itilt,
                               omega=0.,
                               pxInArcsec=pixel_scale,
                               pa=0.,
                               density_dico={
                                   'name': '2PowerLaws',
                                   'ain': alpha_in,
                                   'aout': alpha_out,
                                   'a': a,
                                   'e': 0.,
                                   'ksi0': ksi0,
                                   'gamma': 2.,
                                   'beta': 1.
                               },
                               spf_dico={
                                   'name': 'DoubleHG',
                                   'g': [g1, g2],
                                   'weight': alpha,
                                   'polar': False
                               },
                               flux_max=SNR*np.max(noise))

   return model.compute_scattered_light()

def logl(theta, data, noise, PSF, SNR):
    """ measure the Chisquare (log of the likelyhood) of the parameter set.
        create disk
        convolve by the PSF (psf is global)
        do the forward modeling (diskFM obj is global)
        nan out when it is out of the zone (zone mask is global)
        subctract from data and divide by noise (data and noise are global)

    Args:
        theta: list of parameters of the MCMC

    Returns:
        Chisquare
    """

    model = getDisk(theta, noise, SNR)
    
    # convolve takes inputs for kernel (PSF) with odd sizes in both axis
    if PSF.shape[0]%2 == 0 and PSF.shape[1]%2 == 0:
       model_convolved = convolve(model_map, PSF[:-1,:-1])
    elif PSF.shape[0]%2 == 0:
       model_convolved = convolve(model_map, PSF[:-1,:])
    elif PSF.shape[1]%2 == 0:
       model_convolved = convolve(model_map, PSF[:,:-1])
    else:
       model_convolved = convolve(model_map, PSF)

    res = (data - model_convolved) / noise

    Logliklyhodd = np.nansum(-0.5 * (res * res))

    return Logliklyhodd
     
     
# PSF

hdulist_PSF = fits.open('FITS_files/ird_convert_recenter_dc2022-IRD_SCIENCE_PSF_MASTER_CUBE-median_unsat.fits')
PSF = hdulist_PSF[0].data[-1,-1,:,:]
hdulist_PSF.close()

hdu = fits.PrimaryHDU(PSF)
hdu.writeto('FITS_files/disk_LIU_PSF_convolution.fits',overwrite=True)

# Noise

hdulist_emp_sph = fits.open('FITS_files/ird_convert_recenter_dc2022-IRD_SCIENCE_REDUCED_MASTER_CUBE-center_im.fits')
emp_sph = hdulist_emp_sph[0].data[0,52,412:1024-412,412:1024-412]
hdulist_emp_sph.close()

noise = get_Noise(emp_sph,PSF)

# Image parameters
pixel_scale = 0.01225 # pixel scale in arcsec/px
dstar = 80 # distance to the star in pc
nx = 200 # number of pixels of your image in X
ny = 200 # number of pixels of your image in Y

# Geometrical parameters
itilt = 45. # inclination of your disk in degrees
a = 70. # semimajoraxis of the disk in au
ksi0 = 3. # reference scale height at the semi-major axis of the disk
alpha_in = 12.
alpha_out = -12.

# Scattering: Double Henyey-Greenstein phase function
g1 = 0.7
g2 = -0.2
alpha = 0.665

SNR = 2.

# Disk model
model = ScatteredLightDisk(nx=nx,
                               ny=ny,
                               distance=dstar,
                               itilt=itilt,
                               omega=0.,
                               pxInArcsec=pixel_scale,
                               pa=0.,
                               density_dico={
                                   'name': '2PowerLaws',
                                   'ain': alpha_in,
                                   'aout': alpha_out,
                                   'a': a,
                                   'e': 0.,
                                   'ksi0': ksi0,
                                   'gamma': 2.,
                                   'beta': 1.
                               },
                               spf_dico={
                                   'name': 'DoubleHG',
                                   'g': [g1, g2],
                                   'weight': alpha,
                                   'polar': False
                               },
                               flux_max=SNR*np.max(noise))
                               
model.phase_function.plot_phase_function()

model_map = model.compute_scattered_light()

plt.figure()
plt.imshow(model_map)
plt.colorbar()
plt.savefig('Images/model_vip_SNR_'+str(int(SNR))+'.jpg')
plt.show()

# convolve takes inputs for kernel (PSF) with odd sizes in both axis
if PSF.shape[0]%2 == 0 and PSF.shape[1]%2 == 0:
   model_convolved = convolve(model_map, PSF[:-1,:-1])
elif PSF.shape[0]%2 == 0:
   model_convolved = convolve(model_map, PSF[:-1,:])
elif PSF.shape[1]%2 == 0:
   model_convolved = convolve(model_map, PSF[:,:-1])
else:
    model_convolved = convolve(model_map, PSF)

plt.figure()
plt.imshow(model_convolved)
plt.colorbar()
plt.title('Convolved model (no noise)')
plt.show()

final_model = model_convolved + noise

plt.figure()
plt.imshow(final_model)
plt.colorbar()
plt.title('Final model')
plt.savefig('Images/final_model_SNR_'+str(int(SNR))+'.jpg')
plt.show()

hdu = fits.PrimaryHDU(final_model)
hdu.writeto('FITS_files/disk_LIU_model_SNR_'+str(int(SNR))+'.fits',overwrite=True)

############   MAX LIKELIHOOD   ############

np.random.seed(42)
nll = lambda *args: -logl(*args)
initial = np.array([itilt, a, ksi0, alpha_in, alpha_out, g1, g2, alpha]) + 0.1 * np.random.randn(8)
soln = minimize(nll, initial, args=(final_model, noise, PSF, SNR))
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

labels = ['inc','a','H','a_in','a_out','g1','g2','weight']
save =  np.column_stack((labels, soln.x))

head = 'True values: inc='+str(itilt)+', a='+str(a)+', H='+str(ksi0)+', a_in='+str(alpha_in)+', a_out='+str(alpha_out)+', g1='+str(g1)+', g2='+str(g2)+', weight='+str(alpha)
np.savetxt('ML_IC/ML_IC_SNR_'+str(int(SNR))+'.txt',save,delimiter=" ", fmt="%s",header=head)





