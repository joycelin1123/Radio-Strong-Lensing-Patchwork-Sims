import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gc
import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import h5py


from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.Util.param_util import phi_q2_ellipticity

# use cosmology used in MillenniumTNG --> Planck 2016
from astropy.cosmology import Planck15

# open file and convert to pandas dataframe
def open_halo_file(filename):    
    df = pd.read_hdf(filename+'.h5', key = 'df', mode = 'r+')
    return df

# save to file
def close_halo_file(filename, df):
    df.to_hdf(filename+'.h5', key = 'df', mode = 'w')
    del df
    gc.collect()

# open file and create pandas dataframe
# code taken from: https://stackoverflow.com/questions/71388502/read-hdf5-file-created-with-h5py-using-pandas 
def convert_hdf5_file(filename):
    dictionary = {}
    with h5py.File(filename+'.hdf5', "r") as f:
        for key in f.keys():
            ds_arr = f[key][()]   # returns as a numpy array
            dictionary[key] = ds_arr # appends the array in the dict under the key

    # df holding data
    df = pd.DataFrame.from_dict(dictionary)

    return df

# calculate the Einstein radius  
def get_einstein_rings(z_lens_dat, vdisp_dat):

    # to get the largest einstein ring, lens is halfway between source and observer
    z_source_dat = 2 * z_lens_dat

    # create cosmo class based on the redshifts and the Planck 2016 parameters
    lc = LensCosmo(z_lens_dat, z_source_dat, cosmo=Planck15)

    # calculate the Einstein radius in arcseconds assuming a SIS profile
    # SIS profile more accurately describes galaxies but not groups or clusters
    theta_E_dat = lc.sis_sigma_v2theta_E(vdisp_dat)

    return theta_E_dat

# add einstein rings to the dataframe
def add_einstein_rings(df):

    # get z_lens, vdisp
    z_lens_dat = np.array(df['z_cos'])
    vdisp_dat = np.array(df['vdisp'])

    radius_E = get_einstein_rings(z_lens_dat, vdisp_dat)

    # add in distributions to the df
    df['theta_E'] = radius_E

    return df

# calculate list of ellipticities given axis ratio (q) quantities
def get_ellipticities(qmean, qsigma, N, skew, qmin, qmax):

    # literal adaption from:
    # http://stackoverflow.com/questions/4643285/how-to-generate-random-numbers-that-follow-skew-normal-distribution-in-matlab  
    # original at:
    # http://www.ozgrid.com/forum/showthread.php?t=108175
    def rand_skew_norm(fAlpha, fLocation, fScale, fmin, fmax):
        
        sigma = fAlpha / np.sqrt(1.0 + fAlpha**2) 

        afRN = np.random.randn(2)
        u0 = afRN[0]
        v = afRN[1]
        u1 = sigma*u0 + np.sqrt(1.0 -sigma**2) * v 

        if u0 >= 0: val = u1*fScale + fLocation 
        else: val = (-u1)*fScale + fLocation 

        # recursively iterate if value outside boundary
        if ((val<fmin) | (val>fmax)):
            val = rand_skew_norm(fAlpha, fLocation, fScale, fmin, fmax)

        return val

    # get a list of N samples for a skewed distribution
    def randn_skew(qmean, qsigma, N, skew, qmin, qmax):
        return [rand_skew_norm(skew, qmean, qsigma, qmin, qmax) for x in range(N)]

    # get list of axis ratios (q)
    q = np.array(randn_skew(qmean, qsigma, N, skew, qmin, qmax))

    # get a list of angles (phi)
    phi = (np.random.rand(N)-0.5)*np.pi

    # get distribution of ellipticities
    e1, e2 = phi_q2_ellipticity(phi, q)

    return e1, e2

# add ellipticity to the dataframe
def add_ellipticities(df, qmean, qsigma, skew, qmin, qmax):

    # get number of rows
    N = df.shape[0]

    # get ellipticities based off skewed distribution
    e1, e2 = get_ellipticities(qmean, qsigma, N, skew, qmin, qmax)

    # add in distributions to the df
    df['halo_e1'] = e1
    df['halo_e2'] = e2

    return df

# Note: masses are in log10 units in log(Msol/h)
def fix_masses(df):

    # multiply certain columns by 10^10 
    # Lightcone gives mass in units of 10^10 Msol/h but we want in units of log(Msol/h)
    cols = ['M200c', 'M200m']
    df[cols] = np.log10(df[cols]) + 10

    return df

# get rid of halo masses 
def apply_cutoffs(df):

    # identify rows with halo mass over 10^12.5 sol mass and drop
    rows = df[df.M200c > 12.5].index
    df = df.drop(rows)

    return df

# given hdf5 file from lightcone, add in einstein rings, ellipticities, fix masses, and apply cutoffs and save as h5 file
def format_halo_file(filename, folder, save_root, qmean, qsigma, skew, qmin, qmax):
    # open file and convert to pandas datagrame
    df = convert_hdf5_file(folder + filename)

    # add in einstein rings
    print("adding in einstein rings")
    add_einstein_rings(df)

    # add in ellipticities
    print("adding in ellipticities")
    add_ellipticities(df, qmean, qsigma, skew, qmin, qmax)

    # fix masses (only once!)
    print("fixing masses")
    fix_masses(df)

    # apply mass cutoffs
    print("applying mass cutoffs")
    apply_cutoffs(df)

    # save as h5 file
    df.to_hdf(save_root + filename+'.h5', key = 'df', mode = 'w', format = 't')






