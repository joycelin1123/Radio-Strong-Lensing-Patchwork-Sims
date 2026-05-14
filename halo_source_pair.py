
# standard libraries
import numpy as np 
import pandas as pd
from multiprocessing import Pool
from itertools import repeat

from lenstronomy.Util.param_util import ellipticity2phi_q


# helper function for parallelization
# identify the indices of sources that might be lensed by the halo
def get_source_ind(halo_id, halo_x, halo_y, source_x, source_y, source_z, halo_theta, halo_e1, halo_e2, halo_z):

    # takes into account the ellipticity of the halo
    _, q = ellipticity2phi_q(halo_e1, halo_e2)
    max_theta_range = 4 * halo_theta/q

    # if outside bounds (10 degrees each way), don't even do the math
    if halo_x > 10 or halo_x < -10 or halo_y > 10 or halo_y < -10:
        return []
    
    # get indices of sources that are within the k*theta_E and behind the halo
    r = np.sqrt((halo_x - source_x)**2 + (halo_y - source_y)**2)
    ind = np.where((r < max_theta_range) & (halo_z < source_z))[0]

    # save the indices if a source is found
    ind_len = len(ind)

    # loop through all of the indices
    halo_source_L = []
    if (ind_len != 0):
        for ind_i in ind:
            halo_source_L += [[halo_id, ind_i]]
    
    return halo_source_L

# identify the indices of the lens-source pairs
def find_halo_source_pairs(halo_df, source_df, halo_x_center, halo_y_center, source_x_center, source_y_center):

    # get halo attributes
    # x,y in degrees, theta_E converted to degrees, z_cos is the cosmological redshift
    halo_x = np.array(halo_df['xcoord']) - halo_x_center
    halo_y = np.array(halo_df['ycoord']) - halo_y_center
    halo_theta = 0.5*np.array(halo_df['theta_E']) / 3600  # convert to degrees (originally in arcseconds)
    halo_z = np.array(halo_df['z_cos'])
    halo_mxxl_id = np.array(halo_df['mxxl_id'])

    halo_e1 = np.array(halo_df['halo_e1'])
    halo_e2 = np.array(halo_df['halo_e2'])

    # get source attributes
    # x,y in degrees, z is the redshift
    source_x = np.array(source_df['xcoord']) - source_x_center
    source_y = np.array(source_df['ycoord']) - source_y_center
    source_z = np.array(source_df['z'])

    # create lists 
    # halo_id_L: all of the halo ids
    # source_id_L: all of the source ids for each halo
    halo_id_L = []
    source_id_L = []

    # print("Looping through halos and sources to find pairs...")
    # print("Number of halos: ", len(halo_x))

    # parallelize the loop using multiprocessing Pool
    pool = Pool(16)
    data = pool.starmap(get_source_ind, zip(halo_mxxl_id, halo_x, halo_y, repeat(source_x), repeat(source_y), repeat(source_z), halo_theta, halo_e1, halo_e2, halo_z))

    # print('Finished finding pairs!')

    # reformat data 
    halo_id_L = []
    source_id_L = []
    for row in data:
        if row == []: continue
        else: 
            halo_id_L += [temp_row[0] for temp_row in row]
            source_id_L += [temp_row[1] for temp_row in row]

    return halo_id_L, source_id_L

# save pair ids into pandas dataframe
def format_halo_source_pairs(halo_df, source_df, halo_x_center, halo_y_center, source_x_center, source_y_center):

    halo_id_L, source_id_L = find_halo_source_pairs(halo_df, source_df, halo_x_center, halo_y_center, source_x_center, source_y_center)

    # create dataframe holding ids and save to h5 file
    pair_df = pd.DataFrame({'halo_id': halo_id_L, 'source_id': source_id_L})

    return pair_df





