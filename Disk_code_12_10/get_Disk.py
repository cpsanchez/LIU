#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####### Generate disk model using VIP and save it as .fits #######

import numpy as np
from vip_hci.fm import ScatteredLightDisk
from astropy.io import fits
#from astropy.convolution import convolve
from scipy.ndimage import convolve
import matplotlib.pyplot as plt

def get_Noise(model_convolved):

   dim1 = model_convolved.shape[0]
   dim2 = model_convolved.shape[1]
   
   mean = 0 
   var = 0.01
   sigma = np.sqrt(var)
   err = 0.5 + 2 * np.random.normal(loc=mean,scale=sigma,size=(dim1,dim2))

   plt.figure()
   plt.imshow(err)
   plt.title('Size of Noise Map: %i x %i'%(dim1,dim2))
   plt.show()

   hdu = fits.PrimaryHDU(err)
   hdu.writeto('disk_LIU_noisemap.fits',overwrite=True)
   
   return err
   
     

# Image parameters
pixel_scale = 0.01225 # pixel scale in arcsec/px
dstar = 80 # distance to the star in pc
nx = 200 # number of pixels of your image in X
ny = 200 # number of pixels of your image in Y

# Geometrical parameters
itilt = 45. # inclination of your disk in degrees
a = 70. # semimajoraxis of the disk in au
ksi0 = 3. # reference scale height at the semi-major axis of the disk
alpha_in = 12
alpha_out = -12

# Scattering: Double Henyey-Greenstein phase function
g1=0.7
g2=-0.2
alpha=0.665

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
                               })
                               
model.phase_function.plot_phase_function()

model_map = model.compute_scattered_light()

hdulist_PSF = fits.open('disk_LIU_PSF_convolution.fits')
PSF = hdulist_PSF[0].data[0,0,:,:]
hdulist_PSF.close()

model_convolved = convolve(model_map, PSF)

noise = get_Noise(model_convolved)

final_model = model_convolved + noise

plt.figure()
plt.imshow(final_model)
plt.title('Final model')
plt.show()

hdu = fits.PrimaryHDU(final_model)
hdu.writeto('disk_LIU_model.fits',overwrite=True)
