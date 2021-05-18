#!/usr/bin/env python3

'''Compute transform from FIB space to SEM space based on 3DCT output

Run 'python3 fib2sem_matrix.py --help' for command line usage information.
Initial z-offset possible, useful for after running transform_fluo.

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

import numpy as np
import os, sys, traceback, psutil
import argparse

import tifffile
from scipy import ndimage

from typing import Tuple

from tools3dct.core import Param3D, corr_transform
from tools3dct import docs

import xml.etree.ElementTree as ET

# Write xml file
def write_icy_xml(mat, xmlFile):
    root = ET.Element('root')
    items = ET.SubElement(root, 'MatrixTransformation', attrib={'m00': f'{mat[0,0]:.16f}',
                                                                'm01': f'{mat[0,1]:.16f}',
                                                                'm02': f'{mat[0,2]:.16f}',
                                                                'm03': f'{mat[0,3]:.16f}',
                                                                'm10': f'{mat[1,0]:.16f}',
                                                                'm11': f'{mat[1,1]:.16f}',
                                                                'm12': f'{mat[1,2]:.16f}',
                                                                'm13': f'{mat[1,3]:.16f}',
                                                                'm20': f'{mat[2,0]:.16f}',
                                                                'm21': f'{mat[2,1]:.16f}',
                                                                'm22': f'{mat[2,2]:.16f}',
                                                                'm23': f'{mat[2,3]:.16f}',
                                                                'm30': f'{mat[3,0]:.16f}',
                                                                'm31': f'{mat[3,1]:.16f}',
                                                                'm32': f'{mat[3,2]:.16f}',
                                                                'm33': f'{mat[3,3]:.16f}',})
    outstring = ET.tostring(root, encoding='unicode')
    f = open(xmlFile, 'w')
    f.write(outstring)
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Calculate tranfrom from FIB to SEM space',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--offset', dest='offset', action='store', default=0, metavar='FLOAT', help='Intial offset in z')
    parser.add_argument('--fcorr', dest='fibparamFile', metavar='FILE', help='3DCT text output for the FIB image')
    parser.add_argument('--scorr', dest='semparamFile', metavar='FILE', help='3DCT text output for the SEM image')
    args = parser.parse_args()

    offsetmat = np.identity(4)
    offsetmat[2,3] = args.offset

    param = Param3D(args.fibparamFile)
    phi = param.phi
    psi = param.psi
    theta = param.theta
    s =  param.s
    tx = param.tx
    ty = param.ty
    tz = param.tz
    fibmat, fibmat_inv = corr_transform(s,tx,ty,tz,phi,theta,psi,0,0,1)

    param = Param3D(args.semparamFile)
    phi = param.phi
    psi = param.psi
    theta = param.theta
    s =  param.s
    tx = param.tx
    ty = param.ty
    tz = param.tz
    semmat, semmat_inv = corr_transform(s,tx,ty,tz,phi,theta,psi,0,0,1)

    m = semmat @ fibmat_inv @ offsetmat

    write_icy_xml(m, 'FIB2SEM.xml')
