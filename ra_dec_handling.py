

# standard libraries
import pandas as pd
from multiprocessing import Pool

import lensing_simulation
import RFC
import sys


# return halo box parameters
def get_halo_box(halo_filename):
    halo_L = halo_filename.split('_')

    halo_x = int(halo_L[2])
    halo_y = int(halo_L[3])
    halo_size = 1

    halo_x_center = halo_x + halo_size/2
    halo_y_center = halo_y + halo_size/2
    
    return halo_x, halo_y, halo_x_center, halo_y_center

# return source box parameters
def get_source_box(source_filename):
    source_L = source_filename.split('_')

    source_x = int(source_L[2])
    source_y = int(source_L[3])
    source_size = 1

    source_x_center = source_x + source_size/2
    source_y_center = source_y + source_size/2

    return source_x, source_y, source_x_center, source_y_center

# run simulations given the files
def run_simulation(halo_filename, RFC_filename, source_filename1, source_filename2, source_filename3):

    # open source and RFC file
    halo_df = lensing_simulation.open_file(halo_filename)
    source_df1 = lensing_simulation.open_file(source_filename1)
    source_df2 = lensing_simulation.open_file(source_filename2)
    source_df3 = lensing_simulation.open_file(source_filename3)
    RFC_df = RFC.open_RFC_file(RFC_filename)

    # list to hold the simulation results
    dict_dataL = []

    # get relevant information for the simulation
    halo_x, halo_y, halo_x_center, halo_y_center = get_halo_box(halo_filename)
    source_x1, source_y1, source_x_center1, source_y_center1 = get_source_box(source_filename1)
    source_x2, source_y2, source_x_center2, source_y_center2 = get_source_box(source_filename2)
    source_x3, source_y3, source_x_center3, source_y_center3 = get_source_box(source_filename3)

    dict_data1 = lensing_simulation.run_one_file_simulation(halo_df, source_df1, RFC_df, halo_x, halo_y, halo_x_center, halo_y_center, source_x1, source_y1, source_x_center1, source_y_center1)
    dict_data2 = lensing_simulation.run_one_file_simulation(halo_df, source_df2, RFC_df, halo_x, halo_y, halo_x_center, halo_y_center, source_x2, source_y2, source_x_center2, source_y_center2)
    dict_data3 = lensing_simulation.run_one_file_simulation(halo_df, source_df3, RFC_df, halo_x, halo_y, halo_x_center, halo_y_center, source_x3, source_y3, source_x_center3, source_y_center3)

    dict_dataL += dict_data1
    dict_dataL += dict_data2
    dict_dataL += dict_data3

    df = pd.DataFrame(dict_dataL)
    df.to_csv(str(halo_x) + '_' + str(halo_y) + '_results.csv', index=False)


#######################################################################################################################

# storing the arguments
program = sys.argv[0]
halo_filename = sys.argv[1]
RFC_filename = sys.argv[2]
source_filename1 = sys.argv[3]
source_filename2 = sys.argv[4]
source_filename3 = sys.argv[5]



# run simulation for all of the files
run_simulation(halo_filename, RFC_filename, source_filename1, source_filename2, source_filename3)

