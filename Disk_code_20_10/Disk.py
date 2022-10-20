#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##### IMPORTS #####

from packaging import version
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

from astropy.convolution import convolve
import numpy as np
from scipy.optimize import minimize
import emcee
from astropy.io import fits
import socket
import os
import sys
import yaml
import distutils.dir_util

##### FUNCTIONS #####

def getDisk(theta, noise, SNR):

   # Image parameters
   pixel_scale = 0.01225 # pixel scale in arcsec/px
   dstar = 80 # distance to the star in pc
   nx = 200 # number of pixels of your image in X
   ny = 200 # number of pixels of your image in Y

   # Geometrical and Scattering (DHG) parameters
   itilt, a, ksi0, alpha_in, alpha_out, g1, g2, alpha = theta

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

    model_map = getDisk(theta, noise, SNR)
    
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
    
def logp(theta):
    """ measure the log of the priors of the parameter set.

    Args:
        theta: list of parameters of the MCMC

    Returns:
        log of priors
    """

    inc = theta[0]
    a = theta[1]
    ksi0 = theta[2]
    ain = theta[3]
    aout = theta[4]
    g1 = theta[5]
    g2 = theta[6]
    alpha = theta[7]
    
    if (inc < 0 or inc > 90):
        return -np.inf

    if (a < 20 or a > 130):  
        return -np.inf
        
    if (ksi0 < 0.1 or ksi0 > 10):  
        return -np.inf

    if (ain < 1 or aout > 30):
        return -np.inf

    if (aout < -30 or aout > -1):
        return -np.inf

    if (g1 < 0.0005 or g1 > 0.9999):
        return -np.inf

    if (g2 < -0.9999 or g2 > -0.0005):
        return -np.inf

    if (alpha < 0.01 or alpha > 0.9999):
        return -np.inf
        
    # otherwise ...

    return 0.0
 
def log_probability(theta, data, zerr, PSF, SNR):
    lp = logp(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + logl(theta, data, zerr, PSF, SNR)   
    
def initialize_walkers_backend(params_mcmc_yaml, DATADIR):
    """ initialize the MCMC by preparing the initial position of the
        walkers and the backend file

    Args:
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file

    Returns:
        if new_backend ==1 then [intial position of the walkers, a clean BACKEND]
        if new_backend ==0 then [None, the loaded BACKEND]
    """

    # if new_backend = 0, reset the backend, if not restart the chains.
    # Be careful if you change the parameters or walkers #, you have to put new_backend = 1
    new_backend = params_mcmc_yaml['NEW_BACKEND']

    nwalkers = params_mcmc_yaml['NWALKERS']
    n_dim_mcmc = params_mcmc_yaml['N_DIM_MCMC']

    file_prefix = params_mcmc_yaml['FILE_PREFIX']

    mcmcresultdir = os.path.join(DATADIR, 'results_MCMC')

    inc_init = params_mcmc_yaml['inc_init']
    a_init = params_mcmc_yaml['a_init']
    ksi0_init = params_mcmc_yaml['ksi0_init']
    ain_init = params_mcmc_yaml['ain_init']
    aout_init = params_mcmc_yaml['aout_init']
    g1_init = params_mcmc_yaml['g1_init']
    g2_init = params_mcmc_yaml['g2_init']
    alpha_init = params_mcmc_yaml['alpha_init']
    
    SNR = params_mcmc_yaml['SNR']

    theta_init = (inc_init, a_init, ksi0_init,
                  ain_init, aout_init, 
                  g1_init, g2_init, alpha_init)

    # Set up the backend
    # Don't forget to clear it in case the file already exists
    filename_backend = os.path.join(mcmcresultdir,
                                    file_prefix + "_backend_file_mcmc_SNR_"+str(int(SNR))+".h5")
    backend_ini = emcee.backends.HDFBackend(filename_backend)

    #############################################################
    # Initialize the walkers. The best technique seems to be
    # to start in a small ball around the a priori preferred position.
    # Dont worry, the walkers quickly branch out and explore the
    # rest of the space.
    if new_backend == 1 or new_backend == 0:
        init_ball0 = np.random.uniform(theta_init[0] * 0.999,
                                       theta_init[0] * 1.001,
                                       size=(nwalkers))
        init_ball1 = np.random.uniform(theta_init[1] * 0.99,
                                       theta_init[1] * 1.01,
                                       size=(nwalkers))
        init_ball2 = np.random.uniform(theta_init[2] * 0.99,
                                       theta_init[2] * 1.01,
                                       size=(nwalkers))
        init_ball3 = np.random.uniform(theta_init[3] * 0.99,
                                       theta_init[3] * 1.01,
                                       size=(nwalkers))
        init_ball4 = np.random.uniform(theta_init[4] * 0.99,
                                       theta_init[4] * 1.01,
                                       size=(nwalkers))
        init_ball5 = np.random.uniform(theta_init[5] * 0.99,
                                       theta_init[5] * 1.01,
                                       size=(nwalkers))
        init_ball6 = np.random.uniform(theta_init[6] * 0.99,
                                       theta_init[6] * 1.01,
                                       size=(nwalkers))
        init_ball7 = np.random.uniform(theta_init[7] * 0.99,
                                       theta_init[7] * 1.01,
                                       size=(nwalkers))

        p0 = np.dstack((init_ball0, init_ball1, init_ball2, init_ball3,
                        init_ball4, init_ball5, init_ball6, init_ball7))

        backend_ini.reset(nwalkers, n_dim_mcmc)
        return p0[0], backend_ini

    return None, backend_ini
    

   
if __name__ == '__main__':

    if len(sys.argv) == 1:
        str_yalm = 'Disk_LIU_MCMC.yaml'
    else:
        str_yalm = sys.argv[1]

    # test on which machine I am
    if socket.gethostname() == 'e-m2irt-7':
        basedir = '/home/localuser/Documents/LIU'
        progress = True  # if on my local machine, showing the MCMC progress bar
    else:
        #basedir = '/home/jmazoyer/data_python/tycho/'
        basedir = 'ASK_JOHAN'
        progress = False
        
    # open the parameter file
    yaml_path_file = os.path.join(os.getcwd(), 'initialization_files',
                                  str_yalm)
    with open(yaml_path_file, 'r') as yaml_file:
        params_mcmc_yaml = yaml.safe_load(yaml_file)

    DATADIR = os.path.join(basedir, params_mcmc_yaml['BAND_DIR'])
    FITSDIR = DATADIR + '/FITS_files'
    FILE_PREFIX = params_mcmc_yaml['FILE_PREFIX']

    mcmcresultdir = os.path.join(DATADIR, 'results_MCMC')
    distutils.dir_util.mkpath(mcmcresultdir)
    
    SNR = params_mcmc_yaml['SNR']
    
    dataset = fits.getdata(
        os.path.join(FITSDIR, FILE_PREFIX + '_model_SNR_'+str(int(SNR))+'.fits'))
            
    # load DISTANCE_STAR & PIXSCALE_INS & DIMENSION and make them global
    DISTANCE_STAR = params_mcmc_yaml['DISTANCE_STAR']
    PIXSCALE_INS = params_mcmc_yaml['PIXSCALE_INS']
    DIMENSION = dataset.shape[0]
    
    # load PSF and make it global
    PSF_full = fits.getdata(
        os.path.join(FITSDIR, FILE_PREFIX + '_PSF_convolution.fits'))
    
    if PSF_full.shape[0]%2 == 0 and PSF_full.shape[1]%2 == 0:
       PSF = PSF_full[:-1,:-1]
    elif PSF_full.shape[0]%2 == 0:
       PSF = PSF_full[:-1,:]
    elif PSF_full.shape[1]%2 == 0:
       PSF = PSF_full[:,:-1]
    else:
       PSF = PSF_full
       
    # load noise and make it global
    NOISE = fits.getdata(os.path.join(FITSDIR, FILE_PREFIX + '_noisemap.fits'))

    # load reduced_data and make it a global variable
    REDUCED_DATA = fits.getdata(
        os.path.join(FITSDIR, FILE_PREFIX + '_model_SNR_'+str(int(SNR))+'.fits'))

    ############   MCMC   ############
    
    init_walkers, BACKEND = initialize_walkers_backend(params_mcmc_yaml, DATADIR)
    
    nwalkers = params_mcmc_yaml['NWALKERS']
    n_dim_mcmc = params_mcmc_yaml['N_DIM_MCMC']

    sampler = emcee.EnsembleSampler(
        nwalkers, n_dim_mcmc, log_probability, args=(REDUCED_DATA, NOISE, PSF, SNR), backend=BACKEND
    )
    sampler.run_mcmc(init_walkers, params_mcmc_yaml['N_ITER_MCMC'], progress=progress);

