#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Map displacements to image, scaled to global absolute maximum

Specify rootnames and pixel sizes.

Expects the following files in the current working directory:
    ROOTNAME_mapping_x.txt
    ROOTNAME_mapping_y.txt
    ROOTNAME_after_mask.tif
    ROOTNAME_squaremask.tif
'''

import numpy as np
import tifffile
import csv

def get_displacements(rootname,pxsize):
    xdash = np.loadtxt(rootname+'_mapping_x.txt')
    ydash = np.loadtxt(rootname+'_mapping_y.txt')
    x, y = np.meshgrid(np.arange(xdash.shape[1]),np.arange(xdash.shape[0]))
    delx = (xdash - x) * pxsize
    dely = (ydash - y) * pxsize
    D = np.sqrt(np.square(delx)+np.square(dely))
    return D, delx, dely

def get_pixel_size(rootname):
    if '200702' in rootname:
        pxsize = 168.688
    elif '200819' in rootname:
        pxsize = 67.448
    elif '201014' in rootname:
        pxsize = 84.310
    elif '201030' in rootname:
        pxsize = 84.310
    else:
        return None
    return pxsize

rootnames = ['200702_2',
             '200702_4',
             '200702_7',
             '200702_8',
             '200819_3',
             '200819_4',
             '200819_6',
             '200819_7',
             '201014_4',
             '201014_7',
             '201014_14',
             '201030_15',
             '201030_16',
             '201030_18']

# Determine global y displacement range
max_Ds = []
max_pos_delx = []
max_neg_delx = []
max_pos_dely = []
max_neg_dely = []
for rootname in rootnames:
    pxsize = get_pixel_size(rootname)
    if pxsize is not None:
        D, delx, dely = get_displacements(rootname,pxsize)
        dely = -dely
        pattern_mask = (tifffile.imread(rootname+'_after_mask.tif') > 0)
        D *= pattern_mask
        delx *= pattern_mask
        dely *= pattern_mask
        abs_D = np.max(D)
        pos_delx = np.max(delx)
        neg_delx = np.min(delx)
        pos_dely = np.max(dely)
        neg_dely = np.min(dely)
        max_Ds.append(abs_D)
        max_pos_delx.append(pos_delx)
        max_neg_delx.append(neg_delx)
        max_pos_dely.append(pos_dely)
        max_neg_dely.append(neg_dely)
        print(f'{rootname}: {abs_D:.0f} nm [x: {neg_delx:.0f}-{pos_delx:.0f} / y: {neg_dely:.0f}-{pos_dely:.0f}')
    else:
        print(f'Pixel size undefined for {rootname}')

max_D = np.max(max_Ds)
max_delx = max(np.max(max_pos_delx),abs(np.min(max_neg_delx)))
max_dely = max(np.max(max_pos_dely),abs(np.min(max_neg_dely)))
print(f'Max absolute displacement: {max_D:.0f} nm')
print(f'Max absolute x-displacement: {max_delx:.0f} nm')
print(f'Max absolute y-displacement: {max_dely:.0f} nm')

# Write displacements as uint8 image, import LUT (tab-separated values) in fiji
print('Displacements normalised to global max, zero set at 127, and saved as uint8 image:')
for rootname in rootnames:
    pxsize = get_pixel_size(rootname)
    if pxsize is not None:
        D, delx, dely = get_displacements(rootname,pxsize)
        dely = -dely
        D = D / max_D * 255
        D[D<0] = 0
        D[D>255] = 255
        image = np.ndarray.astype(D, np.uint8)
        tifffile.imsave(rootname+'_after_displacements.tif',image,photometric='minisblack',compress=0)
        delx = delx / 2 / max_D * 255 + 127
        delx[delx<0] = 0
        delx[delx>255] = 255
        image = np.ndarray.astype(delx, np.uint8)
        tifffile.imsave(rootname+'_after_x_displacements.tif',image,photometric='minisblack',compress=0)
        dely = dely / 2 / max_D * 255 + 127
        dely[dely<0] = 0
        dely[dely>255] = 255
        image = np.ndarray.astype(dely, np.uint8)
        tifffile.imsave(rootname+'_after_y_displacements.tif',image,photometric='minisblack',compress=0)

# Mask for square, mask out lamella and write displacements in columns
print('Applying square mask')
max_Ds = []
max_pos_delx = []
max_neg_delx = []
max_pos_dely = []
max_neg_dely = []
for rootname in rootnames:
    pxsize = get_pixel_size(rootname)
    if pxsize is not None:
        D, delx, dely = get_displacements(rootname,pxsize)
        dely = -dely
        square_mask = (tifffile.imread(rootname+'_squaremask.tif') > 0)
        pattern_mask = (tifffile.imread(rootname+'_after_mask.tif') > 0)
        square_mask *= pattern_mask
        D *= square_mask
        delx *= square_mask
        dely *= square_mask
        abs_D = np.max(D)
        pos_delx = np.max(delx)
        neg_delx = np.min(delx)
        pos_dely = np.max(dely)
        neg_dely = np.min(dely)
        max_Ds.append(abs_D)
        max_pos_delx.append(pos_delx)
        max_neg_delx.append(neg_delx)
        max_pos_dely.append(pos_dely)
        max_neg_dely.append(neg_dely)
        print(f'{rootname}: {abs_D:.0f} nm [x: {neg_delx:.0f}-{pos_delx:.0f} / y: {neg_dely:.0f}-{pos_dely:.0f}')

max_D = np.max(max_Ds)
max_delx = max(np.max(max_pos_delx),abs(np.min(max_neg_delx)))
max_dely = max(np.max(max_pos_dely),abs(np.min(max_neg_dely)))
print(f'Max absolute displacement: {max_D:.0f} nm')
print(f'Max absolute x-displacement: {max_delx:.0f} nm')
print(f'Max absolute y-displacement: {max_dely:.0f} nm')

print('Displacements normalised to global max, zero set at 127, and saved as uint8 image:')
for rootname in rootnames:
    pxsize = get_pixel_size(rootname)
    if pxsize is not None:
        D, delx, dely = get_displacements(rootname,pxsize)
        dely = -dely
        square_mask = (tifffile.imread(rootname+'_squaremask.tif') > 0)
        pattern_mask = (tifffile.imread(rootname+'_after_mask.tif') > 0)
        square_mask *= pattern_mask
        D *= square_mask
        delx *= square_mask
        dely *= square_mask
        image = np.ndarray.astype(D / max_D * 255, np.uint8)
        tifffile.imsave(rootname+'_after_masked_displacements.tif',image,photometric='minisblack',compress=0)
        image = np.ndarray.astype(delx / 2 / max_D * 255 + 127, np.uint8)
        tifffile.imsave(rootname+'_after_masked_x_displacements.tif',image,photometric='minisblack',compress=0)
        image = np.ndarray.astype(dely / 2 / max_D * 255 + 127, np.uint8)
        tifffile.imsave(rootname+'_after_masked_y_displacements.tif',image,photometric='minisblack',compress=0)

# from itertools import zip_longest
# with open('displacements_double_masked.csv','w') as f:
#     f.write('# '+' '.join(rootnames))
#     writer = csv.writer(f)
#     for values in zip_longest(*masked_values):
#         writer.writerow(values)

# # write displacements in matrix form
# np.savetxt('/Users/kf656/Desktop/Current/Deformation/deformation_noSIFT_displacements.txt',displacements)

# write data for vector field plotting in gnuplot
# sample_freq = 50
# x = x[::sample_freq,::sample_freq]
# y = y[::sample_freq,::sample_freq]
# delx = delx[::sample_freq,::sample_freq]
# dely = dely[::sample_freq,::sample_freq]
# displacements = displacements[::sample_freq,::sample_freq]
# np.savetxt(f'/Users/kf656/Desktop/Current/Deformation/deformation_noSIFT_vectors_every{sample_freq}.dat', np.vstack([np.ravel(x),np.ravel(y),np.ravel(delx),np.ravel(dely),np.ravel(displacements)]).T )
