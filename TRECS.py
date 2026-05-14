
import numpy as np
import pandas as pd
import gc

from astropy import units as u
from astropy.cosmology import Planck15


# determine heading for file 
def det_agn_sfg(filename):

    # type of data
    pre = filename[-8:-5]

    if pre == 'agn':
        return(['Lum1400', 'I150', 'I160', 'I220', 'I300', 'I410', 'I560', 'I780', 'I1000', 'I1400', 'I1900', 'I2700', 'I3000', 'I3600', 'I5000', 'I6700', 'I9200', 'I12500', 'I20000', 'P150', 'P160', 'P220', 'P300', 'P410', 'P560', 'P780', 'P1000', 'P1400', 'P1900', 'P2700', 'P3000', 'P3600', 'P5000', 'P6700', 'P9200', 'P12500', 'P20000', 'logMh', 'xcoord', 'ycoord', 'GLAT', 'GLON', 'z', 'physSize', 'angle', 'Size', 'Rs', 'PopFlag'])
    
    elif pre == 'sfg': 
        return(['logSFR', 'I150', 'I160', 'I220', 'I300', 'I410', 'I560', 'I780', 'I1000', 'I1400', 'I1900', 'I2700', 'I3000', 'I3600', 'I5000', 'I6700', 'I9200', 'I12500', 'I20000', 'P150', 'P160', 'P220', 'P300', 'P410', 'P560', 'P780', 'P1000', 'P1400', 'P1900', 'P2700', 'P3000', 'P3600', 'P5000', 'P6700', 'P9200', 'P12500', 'P20000', 'logMh', 'xcoord', 'ycoord', 'GLAT', 'GLON', 'z', 'Size', 'e1', 'e2', 'PopFlag'])
    
    else: raise('Not AGN or SFG.')


# create h5 file from dat file  
# drop certain rows
# AGN: add in ellipticity, size, amplitude
def create_dataset(filename):
    byteL = det_agn_sfg(filename)

    # create list to hold all of the data
    L = []
    for i in range(len(byteL)):
        L += [[]]

    # agns - 584 bytes, sfgs - 475 bytes
    with open(filename + '.dat', "rb") as f:
        for line in f.readlines():
            # make list of numbers in each line
            word_list = str(line.rstrip())[2:-1].split()

            if len(word_list) != len(byteL):

                print(len(word_list))
                print(len(byteL))
                print(word_list)
                print(byteL)

            assert(len(word_list) == len(byteL))

            # add each number to the corresponding 
            for j in range(len(byteL)):
                L[j] += [float(word_list[j])]
    L_array = np.array(L)

    # save data into pandas DF
    df = pd.DataFrame(L_array.T, columns=byteL)

    # add in ellipticities
    pre = filename[-8:-5]
    if pre == 'agn':
        e1 = np.zeros(df.shape[0])
        e2 = np.zeros(df.shape[0])
        df["e1"] = e1
        df["e2"] = e2
    elif pre == 'sfg': pass
    else: raise Exception('Not AGN or SFG.')

    # identify rows that are ss-agn (PopFlag == 6) and drop because will be too faint to be detected 
    rows = df[df.PopFlag == 6].index
    df = df.drop(rows)

    return df

# open TRECS file (.h5 not .dat) as pandas dataframe
def open_TRECS_file(filename):
    df = pd.read_hdf(filename+'.h5', key = 'df', mode = 'r+')
    return df

# save as a csv file
def save_TRECS_file(df, filename):
    # save to hdf5 file
    df.to_hdf(filename + '.h5', key = 'df', mode = 'w')  
    del df
    gc.collect()

# given angular size and redshift, calculate the physical size
def convert_to_physSize(zL, sizeL):

    # convert size from arcseconds to radians
    sizeL = sizeL * u.arcsec
    sizeL = sizeL.to(u.rad, u.dimensionless_angles())

    # use Planck15 cosmolgy (was used for TRECS)
    cosmo = Planck15

    # get the physical size in kpc
    angular_diameter_distance = cosmo.angular_diameter_distance(zL)
    physical_size = sizeL* angular_diameter_distance
    physical_size_kpc = physical_size.to(u.kpc, u.dimensionless_angles())

    # return physical_size_kpc.value
    return physical_size_kpc


# given the physical size and the angle btw the LOS and jet, calculate the projected physical size (takes into accout comoving coordinate)
def calc_proj_physSize(physSizeL, angleL, zL):

    # convert angle from degrees to radians
    angles = np.radians(angleL)

    return physSizeL * np.sin(angles) * ((1+zL)**(-1))



##############################################################

# filename = '/afs/hep.wisc.edu/user/jlin475/private/estimation_code/TRECS_RFC_files/agnswide'

# print(filename[-8:-5])

# # change from .dat file to .h5 file
# df = create_dataset(filename)
# save_TRECS_file(df, filename)




