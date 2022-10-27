#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####### Do plots from MCMC backend file #######

##### IMPORTS #####

import socket
import os
import sys
import yaml
import emcee
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import corner
from astropy.io import fits
from datetime import datetime
import matplotlib.lines as mlines

##### FUNCTIONS #####

def make_chain_plot(params_mcmc_yaml,basedir):
    """ make_chain_plot reading the .h5 file from emcee

    Args:
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file

    Returns:
        None
    """

    thin = params_mcmc_yaml['THIN']
    burnin = params_mcmc_yaml['BURNIN']
    quality_plot = params_mcmc_yaml['QUALITY_PLOT']
    labels = params_mcmc_yaml['LABELS']
    names = params_mcmc_yaml['NAMES']
    SNR = params_mcmc_yaml['SNR']


    mcmcresultdir = os.path.join(basedir, 'results_MCMC_sphere')
    file_prefix = params_mcmc_yaml['FILE_PREFIX']

    name_h5 = file_prefix + '_backend_file_mcmc_SNR_' + str(int(SNR))
    print(os.path.join(mcmcresultdir, name_h5 + '.h5'))
    reader = emcee.backends.HDFBackend(os.path.join(mcmcresultdir, name_h5 + '.h5'))
    chain = reader.get_chain(discard=0, thin=thin)
    log_prob_samples_flat = reader.get_log_prob(discard=burnin,
                                                flat=True,
                                                thin=thin)
    # print(log_prob_samples_flat)
    tau = reader.get_autocorr_time(tol=0)
    if burnin > reader.iteration - 1:
        raise ValueError(
            "the burnin cannot be larger than the # of iterations")
    print("")
    print("")
    print(name_h5)
    print("# of iteration in the backend chain initially: {0}".format(
        reader.iteration))
    print("Max Tau times 50: {0}".format(50 * np.max(tau)))
    print("")

    print("Maximum Likelyhood: {0}".format(np.nanmax(log_prob_samples_flat)))

    print("burn-in: {0}".format(burnin))
    print("thin: {0}".format(thin))
    print("chain shape: {0}".format(chain.shape))

    n_dim_mcmc = chain.shape[2]
    nwalkers = chain.shape[1]

    fig, axarr = plt.subplots(n_dim_mcmc,
                            sharex=True,
                            figsize=(6.4 * quality_plot, 4.8 * quality_plot))

    for i in range(n_dim_mcmc):
        axarr[i].set_ylabel(labels[names[i]], fontsize=5 * quality_plot)
        axarr[i].tick_params(axis='y', labelsize=4 * quality_plot)

        for j in range(nwalkers):
            axarr[i].plot(chain[:, j, i], linewidth=quality_plot)

        axarr[i].axvline(x=burnin, color='black', linewidth=1.5 * quality_plot)

    axarr[n_dim_mcmc - 1].tick_params(axis='x', labelsize=6 * quality_plot)
    axarr[n_dim_mcmc - 1].set_xlabel('Iterations', fontsize=10 * quality_plot)

    #plt.savefig(os.path.join(mcmcresultdir, name_h5 + '_chains.jpg'))
    plt.close()
    
def make_corner_plot(params_mcmc_yaml,basedir):
    """ make corner plot reading the .h5 file from emcee

    Args:
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file


    Returns:
        None
    """

    thin = params_mcmc_yaml['THIN']
    burnin = params_mcmc_yaml['BURNIN']
    labels = params_mcmc_yaml['LABELS']
    names = params_mcmc_yaml['NAMES']
    sigma = params_mcmc_yaml['sigma']
    nwalkers = params_mcmc_yaml['NWALKERS']

    n_dim_mcmc = params_mcmc_yaml['N_DIM_MCMC']

    file_prefix = params_mcmc_yaml['FILE_PREFIX']
    
    SNR = params_mcmc_yaml['SNR']

    name_h5 = file_prefix + '_backend_file_mcmc_SNR_' + str(int(SNR))

    band_name = params_mcmc_yaml['BAND_NAME']
    mcmcresultdir = os.path.join(basedir, 'results_MCMC_sphere')
    reader = emcee.backends.HDFBackend(os.path.join(mcmcresultdir, name_h5 + '.h5'))
    chain_flat = reader.get_chain(discard=burnin, thin=thin, flat=True)

    rcParams['axes.labelsize'] = 19
    rcParams['axes.titlesize'] = 14

    rcParams['xtick.labelsize'] = 13
    rcParams['ytick.labelsize'] = 13

    ### cumulative percentiles
    ### value at 50% is the center of the Normal law
    ### value at 50% - value at 15.9% is -1 sigma
    ### value at 84.1%% - value at 50% is 1 sigma
    if sigma == 1:
        quants = (0.159, 0.5, 0.841)
    if sigma == 2:
        quants = (0.023, 0.5, 0.977)
    if sigma == 3:
        quants = (0.001, 0.5, 0.999)

    #### Check truths = bests parameters

    labels_hash = [labels[names[i]] for i in range(n_dim_mcmc)]
    fig = corner.corner(chain_flat,
                        labels=labels_hash,
                        quantiles=quants,
                        show_titles=True,
                        plot_datapoints=True,
                        verbose=False,
                        truths=(45.,70.,3.,12.,-12.,0.7,-0.2,0.665))

    if file_prefix == 'disk_LIU_not':
        initial_values = [
            45.05, 69.986, 3.065, 12.152, -12.023, 0.677, -0.042, 0.742
        ]

        green_line = mlines.Line2D([], [],
                                   color='green',
                                   label='True injected values')
        plt.legend(handles=[green_line],
                   loc='upper right',
                   bbox_to_anchor=(-1, 10),
                   fontsize=30)

        # log_prob_samples_flat = reader.get_log_prob(discard=burnin,
        #                                             flat=True,
        #                                             thin=thin)
        # wheremin = np.where(
        #     log_prob_samples_flat == np.max(log_prob_samples_flat))
        # wheremin0 = np.array(wheremin).flatten()[0]

        # red_line = mlines.Line2D([], [],
        #                         color='red',
        #                         label='Maximum likelyhood values')
        # plt.legend(handles=[green_line, red_line],
        #         loc='upper right',
        #         bbox_to_anchor=(-1, 10),
        #         fontsize=30)

        # Extract the axes
        axes = np.array(fig.axes).reshape((n_dim_mcmc, n_dim_mcmc))

        # Loop over the diagonal
        for i in range(n_dim_mcmc):
            ax = axes[i, i]
            ax.axvline(initial_values[i], color="g")
            # ax.axvline(samples[wheremin0, i], color="r")

        # Loop over the histograms
        for yi in range(n_dim_mcmc):
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.axvline(initial_values[xi], color="g")
                ax.axhline(initial_values[yi], color="g")

                # ax.axvline(samples[wheremin0, xi], color="r")
                # ax.axhline(samples[wheremin0, yi], color="r")

    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(wspace=0)

    fig.gca().annotate(
                       "{0:,} iterations (with {1:,} burn-in)".format(
                           reader.iteration, burnin),
                       xy=(0.55, 0.99),
                       xycoords="figure fraction",
                       xytext=(-20, -10),
                       textcoords="offset points",
                       ha="center",
                       va="top",
                       fontsize=44)

    fig.gca().annotate("{0:,} walkers: {1:,} models".format(
        nwalkers, reader.iteration * nwalkers),
                       xy=(0.55, 0.95),
                       xycoords="figure fraction",
                       xytext=(-20, -10),
                       textcoords="offset points",
                       ha="center",
                       va="top",
                       fontsize=44)

    #plt.savefig(os.path.join(mcmcresultdir, name_h5 + '_pdfs.pdf'))
    plt.close()

def create_header(params_mcmc_yaml):
    """ measure all the important parameters and exctract their error bars
        and print them and save them in a hdr file

    Args:
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file


    Returns:
        header for all the fits
    """

    thin = params_mcmc_yaml['THIN']
    burnin = params_mcmc_yaml['BURNIN']

    comments = params_mcmc_yaml['COMMENTS']
    names = params_mcmc_yaml['NAMES']

    distance_star = params_mcmc_yaml['DISTANCE_STAR']
    PIXSCALE_INS = params_mcmc_yaml['PIXSCALE_INS']

    sigma = params_mcmc_yaml['sigma']
    n_dim_mcmc = params_mcmc_yaml['N_DIM_MCMC']
    nwalkers = params_mcmc_yaml['NWALKERS']

    file_prefix = params_mcmc_yaml['FILE_PREFIX']
    SNR = params_mcmc_yaml['SNR']
    name_h5 = file_prefix + '_backend_file_mcmc_SNR_' +str(SNR)

    reader = emcee.backends.HDFBackend(os.path.join(mcmcresultdir, name_h5 + '.h5'))
    chain_flat = reader.get_chain(discard=burnin, thin=thin, flat=True)
    log_prob_samples_flat = reader.get_log_prob(discard=burnin,
                                                flat=True,
                                                thin=thin)

    samples_dict = dict()
    comments_dict = comments
    MLval_mcmc_val_mcmc_err_dict = dict()

    for i, key in enumerate(names[:n_dim_mcmc]):
        samples_dict[key] = chain_flat[:, i]

    for i, key in enumerate(names[n_dim_mcmc:]):
        samples_dict[key] = chain_flat[:, i] * 0.


        # dAlpha, dDelta = offset_2_RA_dec(dx_here, dy_here, inc_here, pa_here,
        #                                  distance_star)

        # samples_dict['RA'][modeli] = dAlpha
        # samples_dict['Decl'][modeli] = dDelta

        # semimajoraxis = convert.au_to_mas(r1_here, distance_star)
        # ecc = np.sin(np.radians(inc_here))
        # semiminoraxis = semimajoraxis*np.sqrt(1- ecc**2)

        # samples_dict['Smaj'][modeli] = semimajoraxis
        # samples_dict['ecc'][modeli] = ecc
        # samples_dict['Smin'][modeli] = semiminoraxis

        # true_a, true_ecc, argperi, inc, longnode = kowalsky(
        #     semimajoraxis, ecc, pa_here, dAlpha, dDelta)

        # samples_dict['Rkowa'][modeli] = true_a
        # samples_dict['ekowa'][modeli] = true_ecc
        # samples_dict['ikowa'][modeli] = inc
        # samples_dict['Omega'][modeli] = longnode
        # samples_dict['Argpe'][modeAli] = argperi

    wheremin = np.where(log_prob_samples_flat == np.max(log_prob_samples_flat))
    wheremin0 = np.array(wheremin).flatten()[0]

    if sigma == 1:
        quants = [15.9, 50., 84.1]
    if sigma == 2:
        quants = [2.3, 50., 97.77]
    if sigma == 3:
        quants = [0.1, 50., 99.9]

    for key in samples_dict.keys():
        MLval_mcmc_val_mcmc_err_dict[key] = np.zeros(4)

        percent = np.percentile(samples_dict[key], quants)

        MLval_mcmc_val_mcmc_err_dict[key][0] = samples_dict[key][wheremin0]
        MLval_mcmc_val_mcmc_err_dict[key][1] = percent[1]
        MLval_mcmc_val_mcmc_err_dict[key][2] = percent[0] - percent[1]
        MLval_mcmc_val_mcmc_err_dict[key][3] = percent[2] - percent[1]

    # MLval_mcmc_val_mcmc_err_dict['RAp'] = convert.mas_to_pix(
    #     MLval_mcmc_val_mcmc_err_dict['RA'], PIXSCALE_INS)
    # MLval_mcmc_val_mcmc_err_dict['Declp'] = convert.mas_to_pix(
    #     MLval_mcmc_val_mcmc_err_dict['Decl'], PIXSCALE_INS)

    # MLval_mcmc_val_mcmc_err_dict['R2mas'] = convert.au_to_mas(
    #     MLval_mcmc_val_mcmc_err_dict['R2'], distance_star)

    # print(" ")
    # for key in MLval_mcmc_val_mcmc_err_dict.keys():
    #     print(key +
    #           '_ML: {0:.3f}, MCMC {1:.3f}, -/+1sig: {2:.3f}/+{3:.3f}'.format(
    #               MLval_mcmc_val_mcmc_err_dict[key][0],
    #               MLval_mcmc_val_mcmc_err_dict[key][1],
    #               MLval_mcmc_val_mcmc_err_dict[key][2],
    #               MLval_mcmc_val_mcmc_err_dict[key][3]) + comments_dict[key])
    # print(" ")

    print(" ")
    just_these_params = ['inc','a','ksi0','ain','aout','g1', 'g2', 'Alph']
    for key in just_these_params:
        print(key + ' MCMC {0:.3f}, -/+1sig: {1:.3f}/+{2:.3f}'.format(
            MLval_mcmc_val_mcmc_err_dict[key][1],
            MLval_mcmc_val_mcmc_err_dict[key][2],
            MLval_mcmc_val_mcmc_err_dict[key][3]))
    print(" ")

    hdr = fits.Header()
    hdr['COMMENT'] = 'Best model of the MCMC reduction'
    hdr['COMMENT'] = 'PARAM_ML are the parameters producing the best LH'
    hdr['COMMENT'] = 'PARAM_MM are the parameters at the 50% percentile in the MCMC'
    hdr['COMMENT'] = 'PARAM_M and PARAM_P are the -/+ sigma error bars (16%, 84%)'
    hdr['KL_FILE'] = name_h5
    hdr['FITSDATE'] = str(datetime.now())
    hdr['BURNIN'] = burnin
    hdr['THIN'] = thin

    hdr['TOT_ITER'] = reader.iteration

    hdr['n_walker'] = nwalkers
    hdr['n_param'] = n_dim_mcmc

    hdr['MAX_LH'] = (np.max(log_prob_samples_flat),
                     'Max likelyhood, obtained for the ML parameters')

    for key in samples_dict.keys():
        hdr[key + '_ML'] = (MLval_mcmc_val_mcmc_err_dict[key][0],
                            comments_dict[key])
        hdr[key + '_MC'] = MLval_mcmc_val_mcmc_err_dict[key][1]
        hdr[key + '_M'] = MLval_mcmc_val_mcmc_err_dict[key][2]
        hdr[key + '_P'] = MLval_mcmc_val_mcmc_err_dict[key][3]

    return hdr
    
      
    
if __name__ == '__main__':

    if len(sys.argv) == 1:
        str_yalm = 'Disk_LIU_MCMC.yaml'
    else:
        str_yalm = sys.argv[1]

    # test on which machine I am
    if socket.gethostname() == 'e-m2irt-7':
        basedir = '/home/localuser/Documents/LIU/Disk_code'
        progress = True  # if on my local machine, showing the MCMC progress bar
    else:
        #basedir = '/home/jmazoyer/data_python/tycho/'
        basedir = '/obs/cpuertosanchez/LIU'
        progress = False
        
    # open the parameter file
    yaml_path_file = os.path.join(os.getcwd(), 'initialization_files',
                                  str_yalm)
    with open(yaml_path_file, 'r') as yaml_file:
        params_mcmc_yaml = yaml.safe_load(yaml_file)

    FILE_PREFIX = params_mcmc_yaml['FILE_PREFIX']
    mcmcresultdir = os.path.join(basedir, 'results_MCMC_sphere')
    
    # Plot the chain values
    make_chain_plot(params_mcmc_yaml,basedir)
    
    # Plot the PDFs
    make_corner_plot(params_mcmc_yaml,basedir)

    # measure the best likelyhood model and excract MCMC errors
    #hdr = create_header(params_mcmc_yaml)


