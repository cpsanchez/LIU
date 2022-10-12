# pylint: disable=C0103
####### This is the MCMC fitting code for fitting a disk #######
import os

import sys

import socket

import distutils.dir_util
import warnings

from multiprocessing import cpu_count
from multiprocessing import Pool

import contextlib

from datetime import datetime

import math as mt
import numpy as np

import scipy.optimize as op

import astropy.io.fits as fits
from astropy.convolution import convolve

import yaml

from emcee import EnsembleSampler
from emcee import backends

# import make_gpi_psf_for_disks as gpidiskpsf

from vip_hci.fm import ScatteredLightDisk
import astro_unit_conversion as convert

os.environ["OMP_NUM_THREADS"] = "1"


#######################################################
def call_gen_disk(theta):
    """ call the disk model from a set of parameters. 2g SPF
        use DIMENSION, PIXSCALE_INS and DISTANCE_STAR

    Args:
        theta: list of parameters of the MCMC

    Returns:
        a 2d model
    """
    
    inc = theta[0]
    a = theta[1]
    ksi0 = theta[2]
    ain = theta[3]
    aout = theta[4]
    g1 = theta[5]
    g2 = theta[6]
    alpha = theta[7]

    #generate the model

    model = ScatteredLightDisk(nx=DIMENSION,
                               ny=DIMENSION,
                               distance=DISTANCE_STAR,
                               itilt=inc,
                               omega=0.,
                               pxInArcsec=PIXSCALE_INS,
                               pa=0.,
                               density_dico={
                                   'name': '2PowerLaws',
                                   'ain': ain,
                                   'aout': aout,
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
    return model.compute_scattered_light()


########################################################
def logl(theta):
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

    model = call_gen_disk(theta)

    modelconvolved = convolve(model, PSF, boundary='wrap')

    # reduced data have already been naned outside of the minimization
    # zone, so we don't need to do it also for model_fm
    res = (REDUCED_DATA - modelconvolved) / NOISE

    Logliklyhodd = np.nansum(-0.5 * (res * res))

    return Logliklyhodd


########################################################
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

    if (a < 20 or a > 130):  #Can't be bigger than 200 AU
        return -np.inf
        
    if (ksi0 < 0.1 or ksi0 > 10):  #The aspect ratio
        return -np.inf

    if (ain < 1 or aout > 30):
        return -np.inf

    if (aout < -30 or aout > -1):
        return -np.inf

    if (g1 < 0.05 or g1 > 0.9999):
        return -np.inf

    if (g2 < -0.9999 or g2 > -0.05):
        return -np.inf

    if (alpha < 0.01 or alpha > 0.9999):
        return -np.inf
        
    # otherwise ...

    return 0.0


########################################################
def lnpb(theta):
    """ sum the logs of the priors (return of the logp function)
        and of the likelyhood (return of the logl function)


    Args:
        theta: list of parameters of the MCMC

    Returns:
        log of priors + log of likelyhood
    """
    from datetime import datetime
    starttime=datetime.now()
    lp = logp(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = logl(theta)
    print("Running time model + FM: ", datetime.now()-starttime)

    return lp + ll

def initialize_walkers_backend(params_mcmc_yaml):
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

    theta_init = from_param_to_theta_init(params_mcmc_yaml)

    # Set up the backend
    # Don't forget to clear it in case the file already exists
    filename_backend = os.path.join(mcmcresultdir,
                                    file_prefix + "_backend_file_mcmc.h5")
    backend_ini = backends.HDFBackend(filename_backend)

    #############################################################
    # Initialize the walkers. The best technique seems to be
    # to start in a small ball around the a priori preferred position.
    # Dont worry, the walkers quickly branch out and explore the
    # rest of the space.
    if new_backend == 1:
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


########################################################
def from_param_to_theta_init(params_mcmc_yaml):
    """ create a initial set of MCMCparameter from the initial parmeters
        store in the init yaml file
    Args:
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file

    Returns:
        initial set of MCMC parameter
    """
    
    inc_init = params_mcmc_yaml['inc_init']
    a_init = params_mcmc_yaml['a_init']
    ksi0_init = params_mcmc_yaml['ksi0_init']
    ain_init = params_mcmc_yaml['ain_init']
    aout_init = params_mcmc_yaml['aout_init']
    g1_init = params_mcmc_yaml['g1_init']
    g2_init = params_mcmc_yaml['g2_init']
    alpha_init = params_mcmc_yaml['alpha_init']

    theta_init = (inc_init, a_init, ksi0_init,
                  ain_init, aout_init, 
                  g1_init, g2_init, alpha_init)

    for parami in theta_init:
        if parami == 0:
            raise ValueError("""Do not initialize one of your parameters 
            at exactly 0.0, it messes up the small ball at MCMC initialization"""
                             )

    return theta_init


if __name__ == '__main__':

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # warnings.filterwarnings("ignore", category=UserWarning)
    # warnings.simplefilter('ignore', category=AstropyWarning)

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
        params_mcmc_yaml = yaml.load(yaml_file)

    DATADIR = os.path.join(basedir, params_mcmc_yaml['BAND_DIR'])
    FILE_PREFIX = params_mcmc_yaml['FILE_PREFIX']

    mcmcresultdir = os.path.join(DATADIR, 'results_MCMC')
    distutils.dir_util.mkpath(mcmcresultdir)

    # initialize the things necessary to do a
    # dataset = here put your initialdataset
    dataset = fits.getdata(
        os.path.join(DATADIR, FILE_PREFIX + '_model.fits'))[
            0]  

    # load DISTANCE_STAR & PIXSCALE_INS & DIMENSION and make them global
    DISTANCE_STAR = params_mcmc_yaml['DISTANCE_STAR']
    PIXSCALE_INS = params_mcmc_yaml['PIXSCALE_INS']
    DIMENSION = dataset.input.shape[1]

    # load PSF and make it global
    PSF = fits.getdata(
        os.path.join(DATADIR, FILE_PREFIX + '_PSF_convolution.fits'))

    # load wheremask2generatedisk and make it global
    # WHEREMASK2GENERATEDISK = (fits.getdata(
    #     os.path.join(klipdir, FILE_PREFIX + '_mask2generatedisk.fits')) == 0)

    # load noise and make it global
    NOISE = fits.getdata(os.path.join(DATADIR, FILE_PREFIX + '_noisemap.fits'))

    # load reduced_data and make it a global variable
    REDUCED_DATA = fits.getdata(
        os.path.join(DATADIR, FILE_PREFIX + '_model.fits'))[
            0]  

    print("")
    print("")
    # Make a final test by printing the likelyhood of the iniatial model
    startTime = datetime.now()
    lnpb_model = lnpb(from_param_to_theta_init(params_mcmc_yaml))

    print("Time for a single model : ", datetime.now() - startTime)
    print('Parameter Starting point:',
          from_param_to_theta_init(params_mcmc_yaml))
    print("Test likelyhood on initial model :", lnpb_model)

    exploration_algo = "MCMC"

    if exploration_algo == "MCMC":
        print("Initialization MCMC")
        # initialize the walkers if necessary. initialize/load the backend
        # make them global
        init_walkers, BACKEND = initialize_walkers_backend(params_mcmc_yaml)

        # load the Parameters necessary to launch the MCMC
        NWALKERS = params_mcmc_yaml['NWALKERS']  #Number of walkers
        N_ITER_MCMC = params_mcmc_yaml['N_ITER_MCMC']  #Number of iteration
        N_DIM_MCMC = params_mcmc_yaml['N_DIM_MCMC']  #Number of MCMC dimension

        # last chance to remove some global variable to be as light as possible
        # in the MCMC
        del params_mcmc_yaml

        #Let's start the MCMC
        startTime = datetime.now()
        print("Start MCMC")
        with contextlib.closing(Pool()) as pool:

            # Set up the Sampler. I purposefully passed the variables (KL modes,
            # reduced data, masks) in global variables to save time as advised in
            # https://emcee.readthedocs.io/en/latest/tutorials/parallel/
            sampler = EnsembleSampler(NWALKERS,
                                      N_DIM_MCMC,
                                      lnpb,
                                      pool=pool,
                                      backend=BACKEND)

            sampler.run_mcmc(init_walkers, N_ITER_MCMC, progress=progress)

        print("\n time for {0} iterations with {1} walkers and {2} cpus: {3}".
              format(N_ITER_MCMC, NWALKERS, cpu_count(),
                     datetime.now() - startTime))
