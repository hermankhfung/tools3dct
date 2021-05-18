#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Register by translation in 2D and output aligned and cropped images

Run 'python3 register_2d.py --help' for command line usage information.

Run this script to align images by translation before elastic registration.
Run unwarp_slurm_array.sh to perform elastic registration.
Run find_displacements.py to map displacements to image.

Copyright (C) 2021  EMBL/Herman Fung, EMBL/Julia Mahamid

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import argparse
from pathlib import Path

import numpy as np
from skimage.registration import phase_cross_correlation
import tifffile
import cv2

def normalized_uint8(image):
    lower_bound = np.min(image)
    upper_bound = np.max(image)
    lut = np.concatenate([
        np.zeros(lower_bound, dtype=image.dtype),
        np.linspace(0, 255, upper_bound - lower_bound).astype(image.dtype),
        np.ones(2 ** (8*image.dtype.itemsize) - upper_bound, dtype=image.dtype) * 255
    ])
    return lut[image].astype(np.uint8)

def padded_cv_matchTemplate(image_8bit,template_8bit,comparison_method=cv2.TM_CCOEFF_NORMED):

    image_height, image_width = image_8bit.shape
    template_height, template_width = template_8bit.shape
    pad_height = int(template_height / 2)
    pad_width = int(template_width / 2)

    # Pad image
    image_8bit_pad = cv2.copyMakeBorder(image_8bit, pad_height, pad_height, pad_width, pad_width, cv2.BORDER_CONSTANT, value=np.mean(image_8bit)-np.std(image_8bit))

    # Perform template matching
    res = cv2.matchTemplate(image_8bit_pad, template_8bit, comparison_method)
    min_value, max_value, min_location, max_location = cv2.minMaxLoc(res)

    # Take minimum when one of SQDIFF* methods is used, maximum otherwise
    if comparison_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        match_corner_x = min_location[0] - pad_width
        match_corner_y = min_location[1] - pad_height
    else:
        match_corner_x = max_location[0] - pad_width
        match_corner_y = max_location[1] - pad_height

    shift = [match_corner_y, match_corner_x]

    return shift

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Register in 2D and output aligned and cropped images')
    parser.add_argument('--moving', dest='movingFile', required=True, metavar='FILE', help='Moving image')
    parser.add_argument('--fixed', dest='fixedFile', required=True, metavar='FILE', help='Fixed image')
    parser.add_argument('--fixedpx', dest='fixedPxSize', type=float, default=None, metavar='NUMBER', help='Unbinned pixel size of fixed image')
    parser.add_argument('--movingpx', dest='movingPxSize', type=float, default=None, metavar='NUMBER', help='Pixel size of moving image')
    parser.add_argument('--outdir', dest='outDir', default='out', metavar='DIR', help='Output directory')

    args = parser.parse_args()

    # Read images, pixel size, and normalize #

    movingFile = args.movingFile
    fixedFile = args.fixedFile
    outDir = args.outDir
    
    with tifffile.TiffFile(fixedFile) as tif:
        if args.fixedPxSize is None:
            try:
                fixedPxSize = tif.pages[0].tags['FEI_HELIOS'].value['Scan']['PixelWidth']
            except KeyError:
                raise KeyError('No FEI metadata tag')
        else:
            fixedPxSize = args.fixedPxSize
        fixed = normalized_uint8(tif.pages[0].asarray())

    with tifffile.TiffFile(movingFile) as tif:
        if args.movingPxSize is None:
            try:
                movingPxSize = tif.pages[0].tags['FEI_HELIOS'].value['Scan']['PixelWidth']
            except KeyError:
                raise KeyError('No FEI metadata tag')
        else:
            movingPxSize = args.movingPxSize
        moving = normalized_uint8(tif.pages[0].asarray())

    scale_factor = movingPxSize/fixedPxSize

    print('Resizing...')
    dst_size = tuple([int(x * scale_factor) for x in moving.shape[::-1]])
    moving = cv2.resize(moving, dst_size, scale_factor, scale_factor)

    # Phase correlation with upsampled matrix-multiplication DFT #

    fixed_height, fixed_width = fixed.shape
    moving_height, moving_width = moving.shape
    target_height = max(fixed_height,moving_height)
    target_width = max(fixed_width,moving_width)

    print('Padding...')
    fixed_pad = np.ones([target_height,target_width],dtype=np.uint8) * np.mean(fixed)
    fixed_pad[0:fixed_height,0:fixed_width] = fixed
    moving_pad = np.ones([target_height,target_width],dtype=np.uint8) * np.mean(moving)
    moving_pad[0:moving_height,0:moving_width] = moving

    print('Correlating...')
    upsample_factor = 5
    shift, error, phasediff = phase_cross_correlation(moving_pad, fixed_pad, upsample_factor=upsample_factor)

    # # Cross-correlation in real space #

    # # if moving.shape[0]*moving.shape[1] > fixed.shape[0]*fixed.shape[1]:
    # #     shift = padded_cv_matchTemplate(moving,fixed,comparison_method=cv2.TM_CCOEFF_NORMED)
    # # else:
    # #     shift = padded_cv_matchTemplate(fixed,moving,comparison_method=cv2.TM_CCOEFF_NORMED)
    # #     shift *= -1

    print(f'scale = {scale_factor:.5f}  After scaling, shift in x = {-shift[1]:.2f}  shift in y = {-shift[0]:.2f}')

    with tifffile.TiffFile(movingFile) as tif:
        moving = normalized_uint8(tif.pages[0].asarray())
    m = np.float32([ [scale_factor,0,-int(shift[1])], [0,scale_factor,-int(shift[0])] ])
    moving_adjust = cv2.warpAffine(moving, m, (fixed_width, fixed_height))

    x1 = max(0,0-int(shift[1]))
    x2 = min(fixed_width,fixed_width-int(shift[1]))
    y1 = max(0,0-int(shift[0]))
    y2 = min(fixed_height,fixed_height-int(shift[0]))

    Path(outDir).mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(outDir+'/'+movingFile,moving_adjust[y1:y2,x1:x2],photometric='minisblack',compress=0)
    tifffile.imwrite(outDir+'/'+fixedFile,fixed[y1:y2,x1:x2],photometric='minisblack',compress=0)
