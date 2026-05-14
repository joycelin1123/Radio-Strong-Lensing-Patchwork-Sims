

# standard libraries
import numpy as np 
import pandas as pd
import os
from multiprocessing import Pool
from itertools import repeat

from astropy.io import fits
from astropy.wcs import WCS
from ast import literal_eval

import shapely
from shapely.geometry import Polygon
from concave_hull import concave_hull
from sklearn.cluster import DBSCAN

import halo_source_pair

# lenstronomy lens model packages
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
import lenstronomy.Util.util as util
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions

# lenstronomy light model packages
from lenstronomy.LightModel.light_model import LightModel

# lenstronomy simulation packages
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.Util.param_util import phi_q2_ellipticity, ellipticity2phi_q

# lenstronomy image packages
from lenstronomy.ImSim.image_model import ImageModel



# open file as pandas dataframe
def open_file(filename):
    df = pd.read_hdf(filename, key = 'df', mode = 'r+')
    return df 

# write your own function to the paths of the RFC files
def get_filepath(RFC_object, RFC_filename):

    # try images folder
    catalog_path = 'RFC_data/' + RFC_filename + '_catalog.fits'

    # if still doesn't exist, raise exception
    if not os.path.exists(catalog_path):
        raise Exception(f"Catalog file for {catalog_path} not found!")
    
    return catalog_path

# create ellipse and shift by ra,dec
def get_ellipse(cat_maj, cat_min, theta, cat_x, cat_y, num_points = 100):

    # get the semi-major and semi-minor axes
    a = cat_maj
    b = cat_min

    # generate points on the ellipse
    t = np.linspace(0, 2*np.pi, num_points)
    x = a * np.cos(t)
    y = b * np.sin(t)

    # rotate the points by the position angle
    x_rot = x * np.cos(theta) - y * np.sin(theta)
    y_rot = x * np.sin(theta) + y * np.cos(theta)

    # translate the points to the correct position
    x_rot += cat_x
    y_rot += cat_y

    return x_rot, y_rot

# adjust the RFC image to fit TRECS params (halo_x already adjusted for declination effects)
def get_TRECS_RFC_locs(source_row, source_x_center, source_y_center, source_rot_angle, halo_x, halo_y, RFC_df, conv_type, point_maj):

    # TRECS x and y coordinates in arcseconds (adjust for declination effects)
    TRECS_xcoord = (source_row['xcoord'] - source_x_center) * 3600
    TRECS_ycoord = (source_row['ycoord'] - source_y_center) * 3600

    # TRECS size --> used for scaling factor
    TRECS_size = source_row['Size']

    # get the corresponding RFC object
    RFC_index = int(source_row['RFC_index'])
    RFC_row = RFC_df.loc[RFC_df['RFC_inds'] == RFC_index]
    RFC_row = RFC_row.iloc[0]
    RFC_object = RFC_row['Obj_Name']
    RFC_filename = RFC_row['Filename']

    # RFC size --> used for scaling factor
    RFC_size = RFC_row['Size']

    # get only the blobs part of the object
    RFC_blob_inds = RFC_row['Blob_Inds']
    RFC_blob_inds = literal_eval(RFC_blob_inds)

    # get scaling factor for flux and size
    size_scaling = TRECS_size/RFC_size

    catalog_path = get_filepath(RFC_object, RFC_filename)
    cat_hdul = fits.open(catalog_path)
    cat_hdul.verify('fix')
    cat_data = cat_hdul[1].data

    # get ras and decs for the blobs
    # peak flux blob used as location of TRECS object
    cat_ras = cat_data['RA']
    cat_decs = cat_data['DEC'] * 3600
    cat_peak_fluxes = cat_data['Peak_flux']

    # get the major, minor axes, and the rotation angle
    # deconvolve or convolve based on conv_type
    # change position angle from degrees to radians
    if conv_type == 'C':
        cat_majs   = cat_data['Maj']*3600
        cat_mins   = cat_data['Min']*3600
        cat_thetas = (cat_data['PA']*np.pi/180)
    elif conv_type == 'DC':
        cat_majs   = cat_data['DC_Maj']*3600
        cat_mins   = cat_data['DC_Min']*3600
        cat_thetas = (cat_data['DC_PA']*np.pi/180)

    # get ras and decs for the blobs
    cat_ras  = cat_ras[RFC_blob_inds] * size_scaling
    cat_decs = cat_decs[RFC_blob_inds] * size_scaling
    cat_peak_fluxes = cat_peak_fluxes[RFC_blob_inds]

    # get the major, minor axes, and the rotation angle
    cat_majs   = cat_majs[RFC_blob_inds] * size_scaling
    cat_mins   = cat_mins[RFC_blob_inds] * size_scaling
    cat_thetas = cat_thetas[RFC_blob_inds]

    # get the location of the peak blob
    peak_ind = np.argmax(cat_peak_fluxes)
    peak_ra  = cat_ras[peak_ind]
    peak_dec = cat_decs[peak_ind]

    # list to hold source positions and ellipses
    xL = []
    yL = []
    x_ellipse_pointsL = []
    y_ellipse_pointsL = []  

    # loop through each blob
    for cat_ra, cat_dec, cat_maj, cat_min, cat_theta in zip(cat_ras, cat_decs, cat_majs, cat_mins, cat_thetas):

        # calculate the new rotation angle by adding the source rotation angle to the original angle
        new_rot_angle = (np.pi/2) - cat_theta + source_rot_angle

        # rotate the blob around the peak blob
        x_shift = cat_ra - peak_ra
        y_shift = cat_dec - peak_dec
        x_rot = x_shift * np.cos(source_rot_angle) - y_shift * np.sin(source_rot_angle)
        y_rot = x_shift * np.sin(source_rot_angle) + y_shift * np.cos(source_rot_angle)
        
        # calculate the x and y coordinate
        # halo_x should have been already adjusted
        x = x_rot + TRECS_xcoord - halo_x
        y = y_rot + TRECS_ycoord - halo_y

        # get the ellipse for the blobs
        # point source becomes a gaussian with a fixed radius (point_maj)
        if (cat_maj != 0) and (cat_min > 0.01*cat_maj):
            x_ellipse_points, y_ellipse_points = get_ellipse(cat_maj, cat_min, new_rot_angle, x, y, num_points = 20)
        else: 
            x_ellipse_points, y_ellipse_points = get_ellipse(point_maj, point_maj, new_rot_angle, x, y, num_points = 20)
        
        # add to ellipse lists
        x_ellipse_pointsL += [x_ellipse_points]
        y_ellipse_pointsL += [y_ellipse_points]

        # add to a position list
        xL += [x]
        yL += [y]

    # flatten ellipse lists
    x_ellipse_pointsL = [item for sublist in x_ellipse_pointsL for item in sublist]
    y_ellipse_pointsL = [item for sublist in y_ellipse_pointsL for item in sublist]

    return xL, yL, x_ellipse_pointsL, y_ellipse_pointsL

# return both quick and deep sim params
def get_source_params(source_row, source_x_center, source_y_center, source_rot_angle, flux_type, halo_x, halo_y, RFC_df, conv_type, point_maj):

    # TRECS x and y coordinates in arcseconds
    TRECS_xcoord = (source_row['xcoord'] - source_x_center) * 3600
    TRECS_ycoord = (source_row['ycoord'] - source_y_center) * 3600
    TRECS_z = source_row['z']

    # TRECS size --> used for scaling factor
    TRECS_flux = source_row[flux_type]
    TRECS_size = source_row['Size']

    # get the corresponding RFC object
    RFC_index = int(source_row['RFC_index'])
    RFC_row = RFC_df.loc[RFC_df['RFC_inds'] == RFC_index]
    RFC_row = RFC_row.iloc[0]
    RFC_folder = RFC_row['Obj_Name']
    RFC_filename = RFC_row['Filename']

    # RFC size --> used for scaling factor
    RFC_flux = RFC_row['Tot_Flux']*1000
    RFC_size = RFC_row['Size']

    # get only the blobs part of the object
    RFC_blob_inds = RFC_row['Blob_Inds']
    RFC_blob_inds = literal_eval(RFC_blob_inds)

    # get scaling factor for flux and size
    flux_scaling = TRECS_flux/RFC_flux
    size_scaling = TRECS_size/RFC_size

    catalog_path = get_filepath(RFC_folder, RFC_filename)
    cat_hdul = fits.open(catalog_path)
    cat_hdul.verify('fix')
    cat_data = cat_hdul[1].data

    # get ras and decs for the blobs
    # peak flux blob used as location of TRECS object
    cat_ras = cat_data['RA'] * 3600
    cat_decs = cat_data['DEC'] * 3600
    cat_peak_fluxes = cat_data['Peak_flux']

    # get the major, minor axes, and the rotation angle
    # deconvolve or convolve based on conv_type
    # change position angle from degrees to radians
    if conv_type == 'C':
        cat_majs   = cat_data['Maj']*3600
        cat_mins   = cat_data['Min']*3600
        cat_thetas = (cat_data['PA']*np.pi/180)
    elif conv_type == 'DC':
        cat_majs   = cat_data['DC_Maj']*3600
        cat_mins   = cat_data['DC_Min']*3600
        cat_thetas = (cat_data['DC_PA']*np.pi/180)
    
    # get fluxes in mJy
    cat_fluxes = cat_data['Total_flux']*1000
    cat_peak_fluxes = cat_data['Peak_flux']*1000

    # get ras and decs for the blobs
    cat_ras  = cat_ras[RFC_blob_inds] * size_scaling
    cat_decs = cat_decs[RFC_blob_inds] * size_scaling
    cat_peak_fluxes = cat_peak_fluxes[RFC_blob_inds]

    # get the major, minor axes, and the rotation angle
    cat_majs   = cat_majs[RFC_blob_inds] * size_scaling
    cat_mins   = cat_mins[RFC_blob_inds] * size_scaling
    cat_thetas = cat_thetas[RFC_blob_inds]
    cat_fluxes = cat_fluxes[RFC_blob_inds] * flux_scaling

    # get the location of the peak blob
    peak_ind = np.argmax(cat_peak_fluxes)
    peak_ra  = cat_ras[peak_ind]
    peak_dec = cat_decs[peak_ind]

    # add kwarg for each gaussian
    guassian_kwrgs = []
    gaussian_modelL = []
    gaussian_zL = []

    # to hold all of the major axes of each blob
    new_cat_maj = []

    # loop through each blob
    for cat_ra, cat_dec, cat_maj, cat_min, cat_theta, cat_flux in zip(cat_ras, cat_decs, cat_majs, cat_mins, cat_thetas, cat_fluxes):

        # rotate the blob around the peak blob
        x_shift = cat_ra - peak_ra
        y_shift = cat_dec - peak_dec
        x_rot = x_shift * np.cos(source_rot_angle) - y_shift * np.sin(source_rot_angle)
        y_rot = x_shift * np.sin(source_rot_angle) + y_shift * np.cos(source_rot_angle)
        
        # calculate the x and y coordinates (halo_x already adjusted for declination effects)
        x = x_rot + TRECS_xcoord - halo_x
        y = y_rot + TRECS_ycoord - halo_y

        # if maj = min = 0, then point source
        # if maj and min =/ 0, then gaussian ellipse
        # if maj =/ 0 and min = 0, use point source 
        if (cat_maj != 0) and (cat_min > 0.01*cat_maj):

            # calculate the ellipticities from the angle and min/maj axis ratio
            q = cat_min/cat_maj
            phi = (np.pi/2)-cat_theta + source_rot_angle
            e1, e2 = phi_q2_ellipticity(phi, q)
            
            # add kwrgs to master list
            guassian_kwrgs += [{'amp': cat_flux, 'sigma': (cat_maj+cat_min)/(2.355*2), 'e1': e1, 'e2': e2, 'center_x': x, 'center_y': y}]
            gaussian_modelL += ['GAUSSIAN_ELLIPSE']
            gaussian_zL += [TRECS_z]
            new_cat_maj += [cat_maj]

        # point source case
        else: 

            # setting the point sources as gaussians with fixed radius
            guassian_kwrgs += [{'amp': cat_flux, 'sigma': point_maj/2.355, 'center_x': x, 'center_y': y}]
            gaussian_modelL += ['GAUSSIAN']
            gaussian_zL += [TRECS_z]
            new_cat_maj += [point_maj]
            

    return guassian_kwrgs, gaussian_modelL, np.max(np.array(new_cat_maj))

# function to run the quick sim and get the image positions, critical curves, and magnifications for each source position
# also quick_sim a gaussian border to see if any of the blobs become quadruply lensed
def quick_sim(halo_row, source_row, source_rot_angle, halo_x_center, halo_y_center, source_x_center, source_y_center, RFC_df, conv_type, point_maj, pixel_scale):

    # get lens attributes
    lens_model_list = ['SIE']
    kwargs_lens = {'theta_E': 0.5*halo_row['theta_E'], "e1": halo_row['halo_e1'], "e2": halo_row['halo_e2'], 'center_x': 0, 'center_y': 0}
    kwargs_lens_list = [kwargs_lens]
    lens_redshift_list = [halo_row['z_cos']]

    # use this to center the halo and adjust source position (originally in degrees, adjust to arcseconds and adjust for declination effects)
    halo_x = (halo_row['xcoord'] - halo_x_center) * 3600
    halo_y = (halo_row['ycoord'] - halo_y_center) * 3600
    halo_theta = 0.5*halo_row['theta_E']

    halo_e1 = halo_row['halo_e1']
    halo_e2 = halo_row['halo_e2']
    phi, q = ellipticity2phi_q(halo_e1, halo_e2)

    # color counter
    color_num = 0

    # get source attributes
    source_z = source_row['z']
    
    # set a cosmology for distance calculations
    cosmo = WCS

    # create LensModel
    lens_model_class = LensModel(lens_model_list = lens_model_list, cosmo=cosmo, 
                                lens_redshift_list = lens_redshift_list, 
                                z_source = source_z, multi_plane = False)
    
    # create LensEquationSolver
    lensEquationSolver = LensEquationSolver(lens_model_class)
    lensModelExtensions = LensModelExtensions(lens_model_class)

    # QUICK LENSING CODE
    ########################################################################################################################
    # let window size be 1.5*Einstein radius (for caustics and critical curve calculation)
    ra_crit_list, dec_crit_list, _, _ = lensModelExtensions.critical_curve_caustics(kwargs_lens_list, compute_window= 2.5*halo_theta/q, grid_scale = pixel_scale*3, center_x=0, center_y=0)
    
    # get a list of position locations using RFC morphology
    x_vals, y_vals, x_ellipse_points, y_ellipse_points = get_TRECS_RFC_locs(source_row, source_x_center, source_y_center, source_rot_angle, halo_x, halo_y, RFC_df, conv_type, point_maj)

    # list of image position locations and color values (used for plotting later)
    x_images = []
    y_images = []
    color_lensed = []
    num_images = []
    magnifications = []
    
    # loop through each blob in the image and assign color for each image
    for i in range(len(x_vals)):

        # get x and y 
        x = x_vals[i]
        y = y_vals[i]

        # get locations of images
        x_image, y_image = lensEquationSolver.image_position_from_source(kwargs_lens=kwargs_lens_list, sourcePos_x = x, sourcePos_y = y, min_distance=0.01, search_window=5, precision_limit=10**(-10), num_iter_max=100)

        # add to list
        x_images += [x_image]
        y_images += [y_image]
        color_lensed += [color_num for _ in range(len(x_image))]
        num_images += [len(x_image)]

        # calculate the magnification for each lensed position
        for x_img, y_img in zip(x_image, y_image):
            mag = lens_model_class.magnification(x = x_img, y = y_img, kwargs = kwargs_lens_list)
            magnifications += [mag]

        # change color for the next blob
        color_num += 1
    
    # flatten list
    x_images = [item for sublist in x_images for item in sublist]
    y_images = [item for sublist in y_images for item in sublist]
    
    # list to show how many images for each lensed ellipse point (used to determine the number of max images)
    max_lensed_ellipse_points = 0

    # check to see if any doubles are really quads
    if max(num_images) > 1:
        # lens each ellipse point
        for j in range(len(x_ellipse_points)):
            x_ellipse_point = x_ellipse_points[j]
            y_ellipse_point = y_ellipse_points[j]

            # get locations of images
            x_ellipse_image, _ = lensEquationSolver.image_position_from_source(kwargs_lens=kwargs_lens_list, sourcePos_x = x_ellipse_point, sourcePos_y = y_ellipse_point, min_distance=0.01, search_window=5, precision_limit=10**(-10), num_iter_max=100)

            # add to list
            if len(x_ellipse_image) > max_lensed_ellipse_points: max_lensed_ellipse_points = len(x_ellipse_image)

    # check to see if ellipses were multiply lensed, only care abt quad case for ellipses
    # max_images is the correct classification of number of images
    if max_lensed_ellipse_points > 2:
        classification = max_lensed_ellipse_points
    else: 
        classification = max(num_images)

    # this list holds the max number each blob's center was lensed
    max_images = max(num_images)

    return x_images, y_images, color_lensed, classification, max_images, ra_crit_list, dec_crit_list, magnifications

# separate two images inside or outside critical curve
def sep_images_color_helper(color_list, points_ra, points_dec, magsL):

    # initiate variables
    max_color_out = np.max(color_list)
    multi_color = 0

    # find the color that appears twice
    for color_i in range(int(max_color_out) + 1):
        if color_list.count(color_i) == 2:
            multi_color = color_i
            break

    # did not find a color that appears twice, raise exception
    if multi_color == -1:
        print("Multiple image color: ", multi_color)
        print(color_list)
        raise Exception((f"Did not find a color that appears twice inside/outside critical curve!"))

    # find indices of the multiple images
    color_list = np.array(color_list)
    color_inds = np.where(color_list == multi_color)
    color_inds = color_inds[0]
    if len(color_inds) != 2: 
        raise Exception(f"Number of multiple images inside/outside critical curve not equal to 2! Equal to {len(color_inds)}")
    
    # get ra and dec of the two points
    point_1_ra = points_ra[color_inds[0]]
    point_1_dec = points_dec[color_inds[0]]
    point_2_ra = points_ra[color_inds[1]]
    point_2_dec = points_dec[color_inds[1]]

    # turn lists to np arrays for easier calculations
    points_ra = np.array(points_ra)
    points_dec = np.array(points_dec)
    mag_list = np.array(magsL)

    # calculate distances to each point
    dist1 = np.sqrt((points_ra - point_1_ra)**2 + (points_dec - point_1_dec)**2)
    dist2 = np.sqrt((points_ra - point_2_ra)**2 + (points_dec - point_2_dec)**2)
    dist = dist1 - dist2

    # find indices of which points are closer
    points_1_ind = np.where(dist < 0)
    points_2_ind = np.where(dist >= 0)

    # separate into respective lists
    points_1_ra_L = points_ra[points_1_ind]
    points_1_dec_L = points_dec[points_1_ind]
    mag_1_L = mag_list[points_1_ind]

    points_2_ra_L = points_ra[points_2_ind]
    points_2_dec_L = points_dec[points_2_ind]
    mag_2_L = mag_list[points_2_ind]

    return points_1_ra_L, points_1_dec_L, mag_1_L, points_2_ra_L, points_2_dec_L, mag_2_L

# separate image points into separate points based on critical curve
def sep_images_color(ra_crit, dec_crit, x_lensed_images, y_lensed_images, color_lensedL, max_images, magnifications):

    # create a polygon for the critical curve
    polygonsL = []
    for ra, dec in zip(ra_crit[0], dec_crit[0]):
        polygonsL += [(ra, dec)]
    polygon = Polygon(polygonsL)

    # list to hold lists of ra, decs, and colors of each image
    image_ras = []
    image_decs = []
    image_magnifications = []

    # make list of points from lensed images
    pointL = []
    for x_image, y_image in zip(x_lensed_images, y_lensed_images):
        pointL += [(x_image, y_image)]
    pointL = np.array(pointL)

    # make bool list for points inside critical curve
    point_crit_boolL = shapely.contains_xy(polygon, pointL)

    # lists to hold points inside and outside critical curve
    points_in = pointL[point_crit_boolL]
    points_out = pointL[~point_crit_boolL]

    color_lensedL = np.array(color_lensedL)
    color_in = color_lensedL[point_crit_boolL]
    color_out = color_lensedL[~point_crit_boolL]

    magnifications = np.array(magnifications)
    mags_in = magnifications[point_crit_boolL]
    mags_out = magnifications[~point_crit_boolL]
    
    # separate into ra and dec lists
    image_in_crit_ra = points_in[:,0]
    image_in_crit_dec = points_in[:,1]
    image_out_crit_ra = points_out[:,0]
    image_out_crit_dec = points_out[:,1]

    # if only two images, only need to separate into inside and outside critical curve
    if max_images == 2: 

        # see if any duplicate colors in color list
        max_color_out = np.max(color_lensedL)

        # 0 means images separated, 1 means two images inside critical curve, 2 means two images outside critical curve
        list_num = 0

        # find if color that appears twice inside of outside the critical curve
        for color_i in range(int(max_color_out) + 1):
            if list(color_in).count(color_i) == 2:
                list_num = 1
                break
            elif list(color_out).count(color_i) == 2:
                list_num = 2
                break

        # may have the case where quad but some of the images are too faint to be detected, so only 2 images
        # separate images inside the critical curve
        if list_num == 1:
            points_1_ra_in, points_1_dec_in, mag_1_in, points_2_ra_in, points_2_dec_in, mag_2_in = sep_images_color_helper(list(color_in), image_in_crit_ra, image_in_crit_dec, mags_in)

            # create lists
            image_ras = [points_1_ra_in, points_2_ra_in]
            image_decs = [points_1_dec_in, points_2_dec_in]
            image_magnifications = [mag_1_in, mag_2_in]  
            true_max_image = 4

        # separate images outside the critical curve
        elif list_num == 2: 
            points_1_ra_out, points_1_dec_out, mag_1_out, points_2_ra_out, points_2_dec_out, mag_2_out = sep_images_color_helper(list(color_out), image_out_crit_ra, image_out_crit_dec, mags_out)

            image_ras = [points_1_ra_out, points_2_ra_out]
            image_decs = [points_1_dec_out, points_2_dec_out]
            image_magnifications = [mag_1_out, mag_2_out]
            true_max_image = 4

        # one image inside and one image outside critical curve, assign accordingly
        else:
            image_ras = [image_in_crit_ra, image_out_crit_ra]
            image_decs = [image_in_crit_dec, image_out_crit_dec]
            image_magnifications = [mags_in, mags_out]
            true_max_image = 2

    # if 4 images, need to further separate the images
    elif max_images == 4: 

        # separate images inside critical curve
        points_1_ra_in, points_1_dec_in, mag_1_in, points_2_ra_in, points_2_dec_in, mag_2_in = sep_images_color_helper(list(color_in), image_in_crit_ra, image_in_crit_dec, mags_in)

        # separate images outside critical curve
        points_1_ra_out, points_1_dec_out, mag_1_out, points_2_ra_out, points_2_dec_out, mag_2_out = sep_images_color_helper(list(color_out), image_out_crit_ra, image_out_crit_dec, mags_out)

        # add to respective lists in order
        image_ras = [points_1_ra_in, points_2_ra_in, points_1_ra_out, points_2_ra_out]
        image_decs = [points_1_dec_in, points_2_dec_in, points_1_dec_out, points_2_dec_out]
        image_magnifications = [mag_1_in, mag_2_in, mag_1_out, mag_2_out]
        true_max_image = 4

    # this only happens when one of the images is too faint to be detected
    # find out if two images are inside or outside the critical curve and assign accordingly
    elif max_images == 3:

        # see if any duplicate colors in color list
        max_color_out = np.max(color_lensedL)
        true_max_image = 4

        # find the color that appears twice
        for color_i in range(int(max_color_out) + 1):

            # separate images inside the critical curve
            if list(color_in).count(color_i) == 2:
                points_1_ra_in, points_1_dec_in, mag_1_in, points_2_ra_in, points_2_dec_in, mag_2_in = sep_images_color_helper(list(color_in), image_in_crit_ra, image_in_crit_dec, mags_in)

                # create lists
                image_ras = [points_1_ra_in, points_2_ra_in, image_out_crit_ra]
                image_decs = [points_1_dec_in, points_2_dec_in, image_out_crit_dec]
                image_magnifications = [mag_1_in, mag_2_in, mags_out]   

                break

            # separate images outside the critical curve
            if list(color_out).count(color_i) == 2:
                points_1_ra_out, points_1_dec_out, mag_1_out, points_2_ra_out, points_2_dec_out, mag_2_out = sep_images_color_helper(list(color_out), image_out_crit_ra, image_out_crit_dec, mags_out)

                image_ras = [image_in_crit_ra, points_1_ra_out, points_2_ra_out]
                image_decs = [image_in_crit_dec, points_1_dec_out, points_2_dec_out]
                image_magnifications = [mags_in, mag_1_out, mag_2_out]

                break

    # not 2, 3, or 4 images, raise exception
    else: 
        raise Exception(f"Max images is {max_images}, only 2, 3, or 4 images supported!")
    
    return image_ras, image_decs, image_magnifications, true_max_image

# runs individual deep simulation patches
def deep_sim_optimized_helper(numPix, x_mid, y_mid, pixel_scale, fwhm, lens_model_class, source_model_class, kwargs_lens_list, kwargs_gaussian_source):
    
    # data specifics (10 hours = 36000 seconds)
    background_rms = 0.000  #  background noise per pixel (0 for noiseless)
    exp_time = 36000.  #  exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)

    # size of image is numPix * pixel_scale
    psf_type = 'GAUSSIAN'  # 'GAUSSIAN', 'PIXEL', 'NONE'
    psf_truncation = 5

    ############################################################################

    # generate the coordinate grid and image properties (we only read out the relevant lines we need)
    _, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, _ = util.make_grid_with_coordtransform(numPix = numPix, deltapix = pixel_scale, center_ra = x_mid, center_dec = y_mid, subgrid_res = 1, inverse = False)

    # keyword argument for background
    kwargs_data = {'background_rms': background_rms,  # rms of background noise
                'exposure_time': exp_time,  # exposure time (or a map per pixel)
                'ra_at_xy_0': ra_at_xy_0,  # RA at (0,0) pixel
                'dec_at_xy_0': dec_at_xy_0,  # DEC at (0,0) pixel 
                'transform_pix2angle': Mpix2coord,  # matrix to translate shift in pixel in shift in relative RA/DEC (2x2 matrix). Make sure it's units are arcseconds or the angular units you want to model.
                'image_data': np.zeros((numPix, numPix))  # 2d data vector, here initialized with zeros as place holders that get's overwritten once a simulated image with noise is created.
                }

    # create ImageData for ImageModel later
    data_class = ImageData(**kwargs_data)

    # create psf class - NEED TO FIX PSF 
    kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}
    kwargs_psf = {'psf_type': psf_type, 'fwhm': fwhm, 'pixel_size': pixel_scale, 'truncation': psf_truncation}
    psf_class = PSF(**kwargs_psf)

    # create ImageModel using all of the created classes above
    imageModel = ImageModel(data_class, psf_class, lens_model_class=lens_model_class, source_model_class=source_model_class, kwargs_numerics=kwargs_numerics)

    # generate lensed image
    lensed_image_model = imageModel.image(kwargs_lens_list, kwargs_gaussian_source)

    return lensed_image_model

# check to see if box has already been lensed
def check_box_exist(xL, yL, x_condition, y_condition):

    x_bool = (xL > x_condition - 0.5) & (xL < x_condition + 0.5)
    y_bool = (yL > y_condition - 0.5) & (yL < y_condition + 0.5)
    done_bool = x_bool & y_bool

    return np.any(done_bool)

# runs deep simulation for multiple patches based on image positions
def deep_sim_optimized(images_raL, images_decL, max_magnified_majs, max_numPix, pixel_scale, fwhm, lens_model_class, source_model_class, kwargs_lens_list, kwargs_gaussian_source, min_surface_brightness = -8):

    # list to keep track of the deep lensed images
    lensed_imageL = []

    # lists used for plotting purposes later (x_mid + y_mid also used for keeping track of which regions have already been lensed)
    x_mid_doneL = []
    y_mid_doneL = []
    numPix_doneL = []

    # lists to keep track of next batch to lens
    x_mid_nextL = []
    y_mid_nextL = []
    numPix_nextL = []

    # get the largest
    diff = 0
    for image_ras, image_decs, max_magnified_maj in zip(images_raL, images_decL, max_magnified_majs):

        # get bounds of image box based on spread of points
        x_mid = (np.min(image_ras) + np.max(image_ras))/2
        y_mid = (np.min(image_decs) + np.max(image_decs))/2
        x_diff = abs(np.max(image_ras) - np.min(image_ras)) + (2 * max_magnified_maj)
        y_diff = abs(np.max(image_decs) - np.min(image_decs)) + (2 * max_magnified_maj)

        diff = max(x_diff, y_diff)
        numPix_next = int(2*diff//pixel_scale)

        # only lens if patch exists
        if numPix_next > 1:
            # add to next patches to lens list
            x_mid_nextL += [x_mid]
            y_mid_nextL += [y_mid]
            numPix_nextL += [numPix_next]

    # optimized lensing window size
    box_edge_lower = -(max_numPix * pixel_scale)/2
    box_edge_upper = (max_numPix * pixel_scale)/2

    # max number times to loop (number of potential patches)
    max_loop = (max_numPix//max(numPix_nextL) + 1)**2

    # loop through each patch (stop when done lensing)
    for _ in range(max_loop):

        # check if there any more patches to lens
        if len(x_mid_nextL) == 0: break

        # x_mid_next_temp and y_mid_next_temp to hold next batch of patches to lens
        x_mid_next_temp = []
        y_mid_next_temp = []
        numPix_next_temp = []

        # loop through all of the patches to lens
        for x_mid, y_mid, new_numPix in zip(x_mid_nextL, y_mid_nextL, numPix_nextL):

            # get the lensed image for the patch
            lensed_image = deep_sim_optimized_helper(new_numPix, x_mid, y_mid, pixel_scale, fwhm, lens_model_class, source_model_class, kwargs_lens_list, kwargs_gaussian_source)

            # add to lists (marked as done)
            lensed_imageL += [lensed_image]
            numPix_doneL += [new_numPix]
            x_mid_doneL += [x_mid]
            y_mid_doneL += [y_mid]

            # get the border values
            first_row = np.log10(lensed_image[0,:])
            first_col = np.log10(lensed_image[:,0][1:])
            last_row = np.log10(lensed_image[-1,:][1:])
            last_col = np.log10(lensed_image[:,-1][1:-1])

            # check if any of the border surface brightness values are above the min surface brightness
            top_edge_bool = np.any(first_row > min_surface_brightness)
            left_edge_bool = np.any(first_col > min_surface_brightness)
            bottom_edge_bool = np.any(last_row > min_surface_brightness)
            right_edge_bool = np.any(last_col > min_surface_brightness)

            # get which boxes have already been lensed based on the potision with respect to the current box
            box_size = new_numPix * pixel_scale
            x_done_inds = (x_mid_doneL - x_mid)/box_size
            y_done_inds = (y_mid_doneL - y_mid)/box_size
            x_next_inds = (x_mid_nextL - x_mid)/box_size
            y_next_inds = (y_mid_nextL - y_mid)/box_size
            x_temp_inds = (np.array(x_mid_next_temp) - x_mid)/box_size
            y_temp_inds = (np.array(y_mid_next_temp) - y_mid)/box_size
            x_box_inds = np.concatenate((x_done_inds, x_next_inds, x_temp_inds))
            y_box_inds = np.concatenate((y_done_inds, y_next_inds, y_temp_inds))

            # add next patches to lens based on border values (also check if already lensed)
            if top_edge_bool:
                x_condition = 0
                y_condition = 1
                if not check_box_exist(x_box_inds, y_box_inds, x_condition, y_condition) and (y_mid + box_size <= box_edge_upper):
                    x_mid_next_temp += [x_mid]
                    y_mid_next_temp += [y_mid + box_size]
                    numPix_next_temp += [new_numPix]
            if left_edge_bool:
                x_condition = -1
                y_condition = 0
                if not check_box_exist(x_box_inds, y_box_inds, x_condition, y_condition) and (x_mid - box_size >= box_edge_lower):
                    x_mid_next_temp += [x_mid - box_size]
                    y_mid_next_temp += [y_mid]
                    numPix_next_temp += [new_numPix]
            if bottom_edge_bool:
                x_condition = 0
                y_condition = -1
                if not check_box_exist(x_box_inds, y_box_inds, x_condition, y_condition) and (y_mid - box_size >= box_edge_lower):
                    x_mid_next_temp += [x_mid]
                    y_mid_next_temp += [y_mid - box_size]
                    numPix_next_temp += [new_numPix]
            if right_edge_bool:
                x_condition = 1
                y_condition = 0
                if not check_box_exist(x_box_inds, y_box_inds, x_condition, y_condition) and (x_mid + box_size <= box_edge_upper):
                    x_mid_next_temp += [x_mid + box_size]
                    y_mid_next_temp += [y_mid]
                    numPix_next_temp += [new_numPix]
            if top_edge_bool and left_edge_bool:
                x_condition = -1
                y_condition = 1
                if not check_box_exist(x_box_inds, y_box_inds, x_condition, y_condition) and (x_mid - box_size >= box_edge_lower) and (y_mid + box_size <= box_edge_upper):
                    x_mid_next_temp += [x_mid - box_size]
                    y_mid_next_temp += [y_mid + box_size]
                    numPix_next_temp += [new_numPix]
            if top_edge_bool and right_edge_bool:
                x_condition = 1
                y_condition = 1
                if not check_box_exist(x_box_inds, y_box_inds, x_condition, y_condition) and (x_mid + box_size <= box_edge_upper) and (y_mid + box_size <= box_edge_upper):
                    x_mid_next_temp += [x_mid + box_size]
                    y_mid_next_temp += [y_mid + box_size]
                    numPix_next_temp += [new_numPix]
            if bottom_edge_bool and left_edge_bool:
                x_condition = -1
                y_condition = -1
                if not check_box_exist(x_box_inds, y_box_inds, x_condition, y_condition) and (x_mid - box_size >= box_edge_lower) and (y_mid - box_size >= box_edge_lower):
                    x_mid_next_temp += [x_mid - box_size]
                    y_mid_next_temp += [y_mid - box_size]
                    numPix_next_temp += [new_numPix]
            if bottom_edge_bool and right_edge_bool:    
                x_condition = 1
                y_condition = -1
                if not check_box_exist(x_box_inds, y_box_inds, x_condition, y_condition) and (x_mid + box_size <= box_edge_upper) and (y_mid - box_size >= box_edge_lower):
                    x_mid_next_temp += [x_mid + box_size]
                    y_mid_next_temp += [y_mid - box_size]
                    numPix_next_temp += [new_numPix]

        x_mid_nextL = x_mid_next_temp
        y_mid_nextL = y_mid_next_temp
        numPix_nextL = numPix_next_temp

    return lensed_imageL, numPix_doneL, x_mid_doneL, y_mid_doneL

# patch the images together and return one big image
def combine_deep_images(lensed_imageL, x_midL, y_midL, numPixL, pixel_scale):

    # determine size of final image
    x_min = np.min(np.array(x_midL) - (np.array(numPixL)*pixel_scale)/2)
    y_min = np.min(np.array(y_midL) - (np.array(numPixL)*pixel_scale)/2)
    x_max = np.max(np.array(x_midL) + (np.array(numPixL)*pixel_scale)/2)
    y_max = np.max(np.array(y_midL) + (np.array(numPixL)*pixel_scale)/2)

    # get the number of pixels in the x and y direction
    y_pix = int(np.ceil((y_max - y_min)/pixel_scale))
    x_pix = int(np.ceil((x_max - x_min)/pixel_scale))

    # create final image (rows, columns)
    final_image = np.zeros((y_pix, x_pix))

    # loop through each lensed image and place in final image
    for i in range(len(lensed_imageL)):
        x_start = int((x_midL[i] - x_min)/pixel_scale - numPixL[i]/2)
        y_start = int((y_midL[i] - y_min)/pixel_scale - numPixL[i]/2)
        x_end = x_start + numPixL[i]
        y_end = y_start + numPixL[i]

        # check if every pixel is zero (if not zero, part or all of area already has data - don't want duplicates)
        if (final_image[y_start:y_end, x_start:x_end] != 0).any():

            # set overlapping pixels to zero (since we will add the new image on top of it, we can just set the overlapping pixels to zero and add the new image on top of it)
            ind_not_zero = np.nonzero(final_image[y_start:y_end, x_start:x_end])
            final_image[y_start:y_end, x_start:x_end][ind_not_zero] = 0

        # add the patches to the right place
        final_image[y_start:y_end, x_start:x_end] += lensed_imageL[i]

    # array with the corresponding ra and dec for each pixel (center)
    x_ra_val = np.linspace(x_min, x_max, x_pix)
    y_dec_val = np.linspace(y_min, y_max, y_pix)

    return final_image, x_ra_val, y_dec_val

# find the points where the surface brightness is above a certain threshold and filter out the rest of the image
def filter_above_sb(image, sb_threshold, x_ra_vals, y_dec_vals):

    # get indices of pixels above surface brightness threshold
    filter_mask = image > 10**sb_threshold

    # find the ras and decs of the pixels above the surface brightness threshold
    ra_inds = []
    dec_inds = []
    x_inds = []
    y_inds = []
    for i in range(len(filter_mask)):
        filter_row = filter_mask[i]
        # if any of the pixels in the row are above the surface brightness threshold, add the corresponding x/ra and y/dec values to the list 
        if filter_row.any():
            filter_x = np.where(filter_row)[0]
            filter_y = [i for _ in range(len(filter_x))]
            x_inds.extend(filter_x)
            y_inds.extend(filter_y)
            
            filter_ra = x_ra_vals[filter_row]
            filter_dec = [y_dec_vals[i] for _ in range(np.sum(filter_row))]
            ra_inds.extend(filter_ra)
            dec_inds.extend(filter_dec)

    return np.array(ra_inds), np.array(dec_inds), np.array(x_inds), np.array(y_inds)

# get a concave hull given a set of points (should be one of the lensed images)
# length threshold (in arcseconds) controls how tightly the hull fits around the points (smaller values = tighter fit)
def apply_concave_hull(points_ra, points_dec, length_threshold = 0.1):

    # reformat points into list of [x,y] points
    pos_L = []
    for x_pos, y_pos in zip(points_ra, points_dec):
        pos_L += [[x_pos, y_pos]]

    # get concave hull points using length threshold
    concave_hull_points = concave_hull(pos_L, length_threshold=length_threshold)

    return concave_hull_points

# use DBScan to identify which blobs are part of the object
def run_dbscan(x_inds, y_inds, pixel_scale, min_samples):

    # data and weights
    data = np.column_stack((x_inds, y_inds))
    
    # create DBSCAN
    db = DBSCAN(eps = 3*pixel_scale, min_samples = min_samples).fit(data)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    return n_clusters_, labels

# inverse function: given isoperimetric ratio, get arc length
def func_inv(x, a, b):
    return 2 + a/(x+b)

def apply_sb_isoperimteric_calc(patch_image, sb_threshold_val, x_ra_val, y_dec_val, pixel_scale, min_samples, length_threshold, a, b):
    
    # get the indices of the pixels above the surface brightness threshold
    ra_inds, dec_inds, x_inds, y_inds = filter_above_sb(patch_image, sb_threshold_val, x_ra_val, y_dec_val)

    # if no pixels above surface brightness threshold, skip
    if len(ra_inds) == 0:
        # print("No pixels above surface brightness threshold for source id ", source_id)
        return [0], [0], [0], [0]

    # use DBScan to separate images and blobs in images if disconnected
    n_clusters, labels = run_dbscan(ra_inds, dec_inds, pixel_scale, min_samples)

    image_fluxes = []
    # concave hull each identified object/image
    concave_hull_pointsL = []
    for label_i in range(n_clusters):
        # get the indices of points in this cluster
        cluster_inds = np.where(labels == label_i)[0]

        # get the ra and dec values for these points
        ra_inds_cluster = ra_inds[cluster_inds]
        dec_inds_cluster = dec_inds[cluster_inds]
        x_inds_cluster = x_inds[cluster_inds]
        y_inds_cluster = y_inds[cluster_inds]

        # get the flux of these points (for potential future use)
        flux_cluster = patch_image[y_inds_cluster, x_inds_cluster]
        total_flux = float(np.sum(flux_cluster))
        image_fluxes += [total_flux]

        # get concave hull points of the pixels above the surface brightness threshold
        concave_hull_points = apply_concave_hull(ra_inds_cluster, dec_inds_cluster, length_threshold)

        # need to add back first point to complete a closed shape for area and perimeter calculations
        first_point = concave_hull_points[0]
        concave_hull_points = np.array(concave_hull_points + [first_point])
        concave_hull_pointsL += [concave_hull_points]

    # list to hold all of the polygon related measurements
    image_arc_lengths = []
    image_areas = []
    image_perimeters = []
    # loop through all of the concave hull shapes and plot
    for concave_hull_points in concave_hull_pointsL:

        # create a polygon to get the area and the perimter
        image_polygon = Polygon(concave_hull_points)
        image_area = image_polygon.area
        image_perimeter = image_polygon.length
        isoperimetric_ratio = (image_perimeter**2)/image_area

        # calculate the arc length using the isoperimetric ratio and the inverse function
        perimeter_over_arc_length = func_inv(isoperimetric_ratio, a, b)
        image_arc_length = image_perimeter / perimeter_over_arc_length

        image_arc_lengths += [image_arc_length]
        image_areas += [image_area]
        image_perimeters += [image_perimeter]

    return image_fluxes, image_arc_lengths, image_areas, image_perimeters


# get the image arc length, area, and isoperimetric ratio for each lensed image and blob in the image
def get_image_area_length(halo_row, source_row, halo_x_center, halo_y_center, source_x_center, source_y_center, source_rot_angle, ra_crit, dec_crit, x_lensed_image, y_lensed_image, color_lensed, max_image, magL, flux_type, RFC_df, conv_type, point_maj, max_numPix, pixel_scale, fwhm, sb_threshold, min_samples, length_threshold, a, b):

    # use this to center the halo and adjust source position
    halo_x = (halo_row['xcoord'] - halo_x_center) * 3600
    halo_y = (halo_row['ycoord'] - halo_y_center) * 3600

    # all of the lens + source parameters needed for the LensModel
    kwargs_lens = {"theta_E": 0.5*halo_row['theta_E'], "e1": halo_row['halo_e1'], "e2": halo_row['halo_e2'], "center_x": 0, "center_y": 0}
    lens_model_list = ['SIE']
    kwargs_lens_list = [kwargs_lens]
    lens_redshift_list = [halo_row['z_cos']]
    source_z = source_row['z']

    # set a cosmology for distance calculations
    cosmo = WCS

    # create LensModel
    lens_model_class = LensModel(lens_model_list = lens_model_list, cosmo=cosmo, 
                        lens_redshift_list = lens_redshift_list, 
                        z_source = source_z, multi_plane = False)
    
    # get relevant source parameters
    guassian_kwrgs, gaussian_modelL, max_cat_maj = get_source_params(source_row, source_x_center, source_y_center, source_rot_angle, flux_type, halo_x, halo_y, RFC_df, conv_type, point_maj)

    # create light models
    source_model_list = gaussian_modelL
    kwargs_gaussian_source = guassian_kwrgs
    source_model_class = LightModel(source_model_list) 


    # 2 or 4 imgae case
    if (max_image == 2) or (max_image == 3) or (max_image == 4):

        # separate the images based on critical curve 
        image_ras, image_decs, image_magL, true_max_image = sep_images_color(ra_crit, dec_crit, x_lensed_image, y_lensed_image, color_lensed, max_image, magL)

        # get maximium magnified major axis for scaling
        max_magnified_majL = [max_cat_maj * np.max(np.abs(mag)) for mag in image_magL]

        # make sure there is something to lens
        tmpL = []
        for image_ra, image_dec, max_magnified_maj in zip(image_ras, image_decs, max_magnified_majL):

            # get bounds of image box based on spread of points
            x_diff = abs(np.max(image_ra) - np.min(image_ra)) + (2 * max_magnified_maj)
            y_diff = abs(np.max(image_dec) - np.min(image_dec)) + (2 * max_magnified_maj)

            diff = max(x_diff, y_diff)
            numPix_next = int(2*diff//pixel_scale)

            # only lens if patch exists
            if numPix_next > 1: 
                tmpL += [numPix_next]
    
        # only lens if images exist and are large enough (>1 pixel)
        if tmpL == []:
            # print("No images to lens for source id ", source_id)
            return None, None, None, None, None

        # deep lensing simulation of box
        lensed_imageL, numPixL, x_midL, y_midL = deep_sim_optimized(image_ras, image_decs, max_magnified_majL, max_numPix, pixel_scale, fwhm, lens_model_class, source_model_class, kwargs_lens_list, kwargs_gaussian_source)

        # combine deep lensed images into one big image
        patch_image, x_ra_val, y_dec_val = combine_deep_images(lensed_imageL, x_midL, y_midL, numPixL, pixel_scale)

    else: raise Exception(f"Max image is {max_image}, only 2, 3, or 4 images supported!")

    # lists to hold all of the fluxes, arc lengths, areas, perimeters
    image_fluxL = []
    image_arc_lengthL = []
    image_areaL = []
    image_perimeterL = []

    for sb_threshold_val in sb_threshold:
        image_fluxes, image_arc_lengths, image_areas, image_perimeters = apply_sb_isoperimteric_calc(patch_image, sb_threshold_val, x_ra_val, y_dec_val, pixel_scale, min_samples, length_threshold, a, b)

        image_fluxL += [image_fluxes]
        image_arc_lengthL += [image_arc_lengths]
        image_areaL += [image_areas]
        image_perimeterL += [image_perimeters]


    return image_fluxL, image_arc_lengthL, image_areaL, image_perimeterL, true_max_image

# ray trace only necessary parts of the image based on image positions and magnified major axis (for scaling)
# return dictionary
def run_deep_sim_optimized(halo_id, source_id, halo_row, source_row, halo_x_box, halo_y_box, halo_x_center, halo_y_center, source_x_box, source_y_box, source_x_center, source_y_center, source_rot_angle, flux_type, max_numPix, pixel_scale, fwhm, conv_type, point_maj, RFC_df, sb_threshold, length_threshold, min_samples, a, b, q):
    
    # get the lensed image positions, colors, and magnifications for the source
    x_lensed_image, y_lensed_image, color_lensed, classification, max_image, ra_crit, dec_crit, magL = quick_sim(halo_row, source_row, source_rot_angle, halo_x_center, halo_y_center, source_x_center, source_y_center, RFC_df, conv_type, point_maj, pixel_scale)    

    # only lens if multiple images
    if max_image > 1: 

        image_fluxL, image_arc_lengthL, image_areaL, image_perimeterL, true_max_image = get_image_area_length(halo_row, source_row, halo_x_center, halo_y_center, source_x_center, source_y_center, source_rot_angle, ra_crit, dec_crit, x_lensed_image, y_lensed_image, color_lensed, max_image, magL, flux_type, RFC_df, conv_type, point_maj, max_numPix, pixel_scale, fwhm, sb_threshold, min_samples, length_threshold, a, b)

    # assume no image relevant for length and area calculations
    else:
        image_fluxL = []
        image_arc_lengthL = []
        image_areaL = []       
        image_perimeterL = []
        true_max_image = max_image

    # dictionary holding relevant information
    row = {'Halo_id': halo_id, 'Source_id': source_id, 'Halo_x_box': halo_x_box, 'Halo_y_box': halo_y_box, 'Source_x_box': source_x_box, 'Source_y_box': source_y_box, 'Rotation_angle': source_rot_angle, 'Halo_theta': 0.5*halo_row['theta_E'], 'Halo_q': q, 'Pixel_size': pixel_scale, 'Halo_z': halo_row['z_cos'], 'Source_z': source_row['z'], 'sb_threshold': sb_threshold, 'Classification': classification, 'True classification': true_max_image, 'Image_fluxes': image_fluxL, 'Image_arc_lengths': image_arc_lengthL, 'Image_areas': image_areaL, 'Image_perimeters': image_perimeterL}

    return row

# loop function for parallelization
def loop_fn(row_i, halo_x_box, halo_y_box, halo_x_center, halo_y_center, source_x_box, source_y_box, source_x_center, source_y_center, source_rotation_angle, pair_df, halo_df, source_df, RFC_df):

    # lensing sim parameters
    conv_type = 'DC'
    flux_type = 'I5000'     # 5GHz
    pixel_scale = 0.001
    point_maj = 0.001
    # -3.5733, -3.2723 (5, 10 sigma - ngVLA), -1.788, -1.487 (5, 10 sigma - realistic config)
    sb_threshold = [-3.5733, -3.2723, -1.788, -1.487]
    length_threshold = 0.015
    min_samples = 10

    # row containing id info
    row = pair_df.iloc[row_i]
    rows = pair_df.shape[0]

    # get the halo and source ids
    halo_id = row['halo_id']
    source_id = row['source_id']

    # get the rows associated with the halo and source ids
    halo_row = halo_df[halo_df['mxxl_id'] == halo_id].iloc[0]
    halo_theta = 0.5*halo_row['theta_E']

    halo_e1 = halo_row['halo_e1']
    halo_e2 = halo_row['halo_e2']
    _, q = ellipticity2phi_q(halo_e1, halo_e2)

    max_theta_range = 4 * halo_theta/q
    max_numPix = int(max_theta_range / pixel_scale)

    # don't want a super long simulation (may change for keep resolution consistent later)
    if max_numPix > 15000: 
        max_numPix = 15000
        pixel_scale = max_theta_range / max_numPix 
    
    # fwhm of beam
    fwhm = pixel_scale * 2

    # source row
    source_row = source_df.iloc[source_id]

    print('halo_x_center: ', halo_x_center)
    print('halo_y_center: ', halo_y_center)
    print('rows: ', rows)
    print('row_i: ', row_i)
    print('halo_theta: ', halo_theta)
    print('halo_q: ', q)
    print('pixel_scale: ', pixel_scale)
    print()

    # inverse function parameters (for getting arc length from isoperimetric ratio)
    a = 3.2936
    b = -9.6812

    # get the classification, image arc length, image area, and image perimeter for the lensed image of the source
    info_row = run_deep_sim_optimized(halo_id, source_id, halo_row, source_row, halo_x_box, halo_y_box, halo_x_center, halo_y_center, source_x_box, source_y_box, source_x_center, source_y_center, source_rotation_angle, flux_type, max_numPix, pixel_scale, fwhm, conv_type, point_maj, RFC_df, sb_threshold, length_threshold, min_samples, a, b, q)

    return [info_row]

# main function to run the simulation for all halo-source pairs within the theta_E threshold and save results in a csv file
def run_one_file_simulation(halo_df, source_df, RFC_df, halo_x_box, halo_y_box, halo_x_center, halo_y_center, source_x_box, source_y_box, source_x_center, source_y_center):


    # find the halo-source pairs that are within ~2*major axis of einstein radius
    pair_df = halo_source_pair.format_halo_source_pairs(halo_df, source_df, halo_x_center, halo_y_center, source_x_center, source_y_center)

    # get number of rows to loop through
    rows = pair_df.shape[0]

    # assign a random rotation angle to each source (in radians)
    rotation_angleL = np.random.uniform(0, 2*np.pi, rows)
    # rotation_angleL = np.zeros(rows)

    # array holding the number of rows to loop through
    row_vals = np.arange(rows)

    # parallelize the loop using multiprocessing Pool
    pool = Pool(16)
    data = pool.starmap(loop_fn, zip(row_vals, repeat(halo_x_box), repeat(halo_y_box), repeat(halo_x_center), repeat(halo_y_center), repeat(source_x_box), repeat(source_y_box), repeat(source_x_center), repeat(source_y_center), rotation_angleL, repeat(pair_df), repeat(halo_df), repeat(source_df), repeat(RFC_df)))

    # reformat data into one list of dictionaries (each dictionary holds the results for one halo-source pair)
    dict_data = []
    for row in data:
        if row == []: continue
        else: dict_data += row

    return dict_data



