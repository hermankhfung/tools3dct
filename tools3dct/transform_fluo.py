#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' Transform volume according to 3DCT output and transformation matrices in XML format

Run 'python3 transform_fluo.py --help' for command line usage information.

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
from scipy import ndimage, linalg

from typing import Tuple

from tools3dct.core import Param3D, corr_transform
from tools3dct import docs

import xml.etree.ElementTree as ET
from PyQt5 import QtGui, QtWidgets, QtCore

### Base code ###

class WorkerSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(tuple)
    result = QtCore.pyqtSignal(object)

class Worker(QtCore.QRunnable):

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @QtCore.pyqtSlot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # return result of processing
        finally:
            self.signals.finished.emit()  # done

# Class for multi-threaded calculation
class Transformer:

    def __init__(self, CF, m, m_inv, xdim, ydim, parentWidget=None):
        self.CF = CF
        self.m = m
        self.m_inv = m_inv
        self.xdim = xdim
        self.ydim = ydim
        self.zdim = 0
        self.parentWidget = parentWidget
        self.threadpool = QtCore.QThreadPool()
        ncpu = os.cpu_count()
        self.threadpool.setMaxThreadCount(ncpu)
        mem = psutil.virtual_memory()
        if mem.available > 2 * 1024 * 1024 * 1024 * ncpu:
            self.patch_size = 512
        else:
            self.patch_size = 256
        self.npatch = int( np.ceil(xdim/self.patch_size) * np.ceil(ydim/self.patch_size) )
        self.start_coords = []
        self.end_coords = []
        self.patches = []
        self.counter = 0

    def store_result(self, result):
        self.start_coords.append(result[0])
        self.end_coords.append(result[1])
        self.patches.append(result[2])
        
    def thread_complete(self):
        self.counter += 1
        if self.counter == self.npatch:
            self.progressDialog.accept()

    # Find z range to evaluate in output space, saves cpu and memory
    def patch_zbounds(self, start_coord, end_coord, m, m_inv, shape, cnrs):

        x1, y1 = start_coord
        x2, y2 = end_coord

        # intersection of confocal volume boundaries with patch edges
        l = np.array([cnrs[1]-cnrs[0],cnrs[2]-cnrs[0],cnrs[4]-cnrs[0],cnrs[3]-cnrs[1],cnrs[5]-cnrs[1],cnrs[3]-cnrs[2],cnrs[6]-cnrs[2],cnrs[7]-cnrs[3],cnrs[5]-cnrs[4],cnrs[6]-cnrs[4],cnrs[7]-cnrs[5],cnrs[7]-cnrs[6]]).repeat(4,axis=0)
        l0 = np.array([cnrs[0],cnrs[0],cnrs[0],cnrs[1],cnrs[1],cnrs[2],cnrs[2],cnrs[3],cnrs[4],cnrs[4],cnrs[5],cnrs[6]]).repeat(4,axis=0)
        p0 = np.array([[x1,y1,0],[x1,y1,0],[x2,y2,0],[x2,y2,0]]).T.repeat(12,axis=0).reshape(3,-1).T
        n = np.array([[1,0,0],[0,1,0],[1,0,0],[0,1,0]]).T.repeat(12,axis=0).reshape(3,-1).T
        d = np.zeros([48])    
        for i in range(48):
            d[i] = np.dot(p0[i]-l0[i],n[i])/np.dot(l[i],n[i])
        pts = np.vstack([d.repeat(3).reshape(-1,3)*l+l0,cnrs]) # calculate points and add vertices
        within = np.logical_and.reduce([pts[:,0]>=x1,pts[:,1]>=y1,pts[:,0]<=(x2),pts[:,1]<=(y2)], dtype=bool)
        pts = np.compress(within,pts,axis=0)
        patch_cnrs = np.array(np.meshgrid([x1,x2],[y1,y2])).T.reshape(4,2) # append corners of patch
        pts_xy = np.vstack([pts[:,[0,1]],patch_cnrs])
        # pts_xy = np.vstack([np.hstack([np.ones(y2-y1)*x1,range(x2-x1+1),np.ones(y2-y1)*(x2-x1),range(1,x2-x1)]),np.hstack([range(y2-y1),np.ones(x2-x1+1)*(y2-y1),range(y2-y1),np.ones(x2-x1-1)*y1])]).T  # all pixels along edge

        # solve inequality for each corner or xy-position where there is an intersection
        zbounds = np.zeros([pts_xy.shape[0],2])
        for i, pcnr in enumerate(pts_xy):
            u = pcnr[0]
            v = pcnr[1]
            a = np.array([(-1*m_inv[0,0]*u-m_inv[0,1]*v-m_inv[0,3])/m_inv[0,2], \
                        (-1*m_inv[1,0]*u-m_inv[1,1]*v-m_inv[1,3])/m_inv[1,2], \
                        (-1*m_inv[2,0]*u-m_inv[2,1]*v-m_inv[2,3])/m_inv[2,2]])
            b = np.array([(shape[0]-m_inv[0,0]*u-m_inv[0,1]*v-m_inv[0,3])/m_inv[0,2], \
                        (shape[1]-m_inv[1,0]*u-m_inv[1,1]*v-m_inv[1,3])/m_inv[1,2], \
                        (shape[2]-m_inv[2,0]*u-m_inv[2,1]*v-m_inv[2,3])/m_inv[2,2]])
            c = np.array([a,b])
            if m_inv[0,2]<0:
                c[:,0]=np.flip(c[:,0])
            if m_inv[1,2]<0:
                c[:,1]=np.flip(c[:,1])
            if m_inv[2,2]<0:
                c[:,2]=np.flip(c[:,2])
            zbounds[i] = [np.amax(c[0]), np.amin(c[1])]    
        pts_z = np.ravel(np.compress(zbounds[:,0]<=zbounds[:,1],zbounds,axis=0))
        pts_z = np.append(pts_z,np.ravel(np.compress(np.logical_and.reduce([cnrs[:,0]>=x1,cnrs[:,0]<=x2,cnrs[:,1]>=y1,cnrs[:,1]<=y2], dtype=bool),cnrs,axis=0)[:,2])) # append corner of confocal volume if wihtin range
        if pts_z.size != 0:            
            return int(np.floor(np.amin(pts_z))), int(np.ceil(np.amax(pts_z)))
        else:
            return None, None

    # Compute patch
    def vol_patch(self, start_coord, end_coord, m, m_inv, CF_xyz, dest_cnrs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        patch_shape = end_coord - start_coord
        patch = np.zeros([patch_shape[0],patch_shape[1],self.zdim],dtype='float64')
        z_lb, z_ub = self.patch_zbounds(start_coord,end_coord,m,m_inv,CF_xyz.shape,dest_cnrs)
        if z_lb is None:
            return start_coord, end_coord, patch
        else:
            dest_zoffset = -1 * z_lb
            dest_zmax = z_ub - z_lb
        zoffset_inv = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,-1*dest_zoffset],[0,0,0,1]], dtype='float64')
        coord_shift_inv = np.array([[1,0,0,start_coord[0]],[0,1,0,start_coord[1]],[0,0,1,0],[0,0,0,1]], dtype='float64')
        complete_inv = np.matmul(np.matmul(m_inv,zoffset_inv),coord_shift_inv)
        patch = np.zeros([patch_shape[0],patch_shape[1],self.zdim], dtype=np.uint8)
        foo = np.ndarray.astype(ndimage.affine_transform(CF_xyz,complete_inv,output_shape=(patch_shape[0],patch_shape[1],dest_zmax),order=1), dtype=np.uint8)
        patch[:,:,-1*dest_zoffset:-1*dest_zoffset+dest_zmax] = foo
        return start_coord, end_coord, patch

    # Compute transform by interpolation, multithreading
    # affine_transform performs spline filtering when order greater than 1, memory heavy
    def volume_transform(self) -> np.ndarray:        
        
        # rearrange axes for affine transform
        CF_xyz = np.moveaxis(np.moveaxis(self.CF,0,2),0,1)

        # calculate volume corners after transformation
        src_cnrs = np.vstack([np.array(np.meshgrid([0,CF_xyz.shape[0]], [0,CF_xyz.shape[1]], [0,CF_xyz.shape[2]])).reshape(3,-1), np.ones(8)])
        dest_cnrs = np.matmul(self.m,src_cnrs)[0:3].T

        # calculate global z-offset and output z-dimension
        z_lb, z_ub = self.patch_zbounds([0,0],[xdim,ydim],self.m,self.m_inv,CF_xyz.shape,dest_cnrs)
        global_zoffset = -1 * z_lb
        # print([0,0], [xdim,ydim], z_ub, z_lb, global_zoffset)
        self.zdim = z_ub - z_lb
        # compute matrices with z offset
        global_zoffset_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,global_zoffset],[0,0,0,1]], dtype='float64')
        global_zoffset_mat_inv = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,-1*global_zoffset],[0,0,0,1]], dtype='float64')
        final_mat = np.matmul(global_zoffset_mat,self.m)
        final_mat_inv = np.matmul(self.m_inv,global_zoffset_mat_inv)

        # new volume corners after z offset
        dest_cnrs = np.matmul(final_mat,src_cnrs)[0:3].T

        # prepare patches
        start_pixel_coords = np.array(np.mgrid[0:self.xdim:self.patch_size,0:self.ydim:self.patch_size]).T.reshape(-1,2).astype(int)
        end_pixel_coords = start_pixel_coords + self.patch_size
        end_pixel_coords[end_pixel_coords[:,0]>xdim,0] = self.xdim
        end_pixel_coords[end_pixel_coords[:,1]>ydim,1] = self.ydim

        # cancel dialog
        self.progressDialog = QtWidgets.QDialog(self.parentWidget)
        verticalLayout = QtWidgets.QVBoxLayout(self.progressDialog)
        label = QtWidgets.QLabel("Computing volume...",self.progressDialog)
        verticalLayout.addWidget(label)
        buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Cancel,self.progressDialog)
        buttonBox.rejected.connect(self.progressDialog.reject)
        verticalLayout.addWidget(buttonBox)

        # test one patch
        # self.vol_patch(start_pixel_coords[1],end_pixel_coords[1],final_mat,final_mat_inv,CF_xyz,dest_cnrs)
        # return None

        for start_coord, end_coord in zip(start_pixel_coords,end_pixel_coords):
            worker = Worker(self.vol_patch,start_coord,end_coord,final_mat,final_mat_inv,CF_xyz,dest_cnrs) # fn, args, kwargs
            worker.signals.result.connect(self.store_result) # return coord, patch in a tuple
            worker.signals.finished.connect(self.thread_complete)
            self.threadpool.start(worker)
        if self.progressDialog.exec() == QtWidgets.QDialog.Rejected:
            print("Aborting calculation...")
            self.threadpool.clear()

        print("Assembling volume...")
        out_vol = np.zeros([self.xdim,self.ydim,self.zdim], dtype=np.uint8)
        for start_coord, end_coord, patch in zip(self.start_coords,self.end_coords,self.patches):
            out_vol[start_coord[0]:end_coord[0],start_coord[1]:end_coord[1],:] = patch
        out_vol = np.moveaxis(out_vol,[0,1,2],[2,1,0])

        return out_vol, global_zoffset

### Functions ###

# Wrapper to compute volume
def transform_volume_single(CF,m,m_inv,xdim,ydim):
    fluotransformer = Transformer(CF,m,m_inv,xdim,ydim)
    out_vol, z_offset = fluotransformer.volume_transform()
    return out_vol, z_offset

# Read xml file and compute matrices, assumes affine and isotropic scaling
def read_matrix_xml(xmlFile):
    tree = ET.parse(xmlFile)
    root = tree.getroot()
    xmlmat = np.identity(4)
    for child in root.findall('MatrixTransformation'):
        m = np.asarray([ [child.get('m00'),child.get('m01'),child.get('m02'),child.get('m03')],
                         [child.get('m10'),child.get('m11'),child.get('m12'),child.get('m13')],
                         [child.get('m20'),child.get('m21'),child.get('m22'),child.get('m23')],
                         [child.get('m30'),child.get('m31'),child.get('m32'),child.get('m33')] ], dtype=np.float64)
        xmlmat = np.matmul(m,xmlmat)
        xmlmat_inv = find_inverse_transform(m)
    return xmlmat, xmlmat_inv

# Find inverse affine transform
#   RQ decomposition: A = SKQ, where Q is rotation matrix,
#                                    K is scaling matrix, and
#                                    S is shear matrix
def find_inverse_transform(m):
    T_inv = np.identity(4)
    T_inv[0:3,3] = -1 * m[0:3,3]
    m_inv = np.identity(4)
    m_inv[:-1,:-1] = linalg.inv(m[:-1,:-1])
    m_inv = m_inv @ T_inv
    if np.linalg.norm((m_inv @ m)-np.identity(4)) > 2 * np.finfo(float).eps:
        print('Inverse matrix not found')
    return m_inv

def normalized_uint8(image):
    lower_bound = np.min(image)
    upper_bound = np.max(image)
    lut = np.concatenate([
        np.zeros(lower_bound, dtype=image.dtype),
        np.linspace(0, 255, upper_bound - lower_bound).astype(image.dtype),
        np.ones(2 ** (8*image.dtype.itemsize) - upper_bound, dtype=image.dtype) * 255
    ])
    return lut[image].astype(np.uint8)

class ActionAddXmlTransform(argparse._AppendAction):
    def __call__(self, parser, namespace, values, option_string=None):
        xmlmat, xmlmat_inv = read_matrix_xml(values)
        values = [xmlmat, xmlmat_inv]
        super(ActionAddXmlTransform, self).__call__(parser, namespace, values, option_string)

class ActionAdd3dctTransform(argparse._AppendAction):
    def __call__(self, parser, namespace, values, option_string=None):
        param = Param3D(values)
        phi = param.phi
        psi = param.psi
        theta = param.theta
        s =  param.s
        tx = param.tx
        ty = param.ty
        tz = param.tz
        corrmat, corrmat_inv = corr_transform(s,tx,ty,tz,phi,theta,psi,0,0,1)
        values = [corrmat, corrmat_inv]
        super(ActionAdd3dctTransform, self).__call__(parser, namespace, values, option_string)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Transform fluorescence volume(s)\n\nVolume(s) will be transformed in the order that --corr and --xml are specified. Multiple -corr and --xml can be specified.\nThe whole z-range of the transformed volume will be outputted with an offset.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--corr', dest='matrices', action=ActionAdd3dctTransform, metavar='FILE', help='Output by 3DCT describing transformation')
    parser.add_argument('--xml', dest='matrices', action=ActionAddXmlTransform, metavar='FILE', help='Transformation matrices in XML format')
    parser.add_argument('--fluo', dest='cfFiles', required=True, nargs='+', metavar='FILE', help='resliced fluorescence volume(s)')
    parser.add_argument('--xdim', dest='xdim', required=True, type=int, metavar='INTEGER', help='Output x dimension (px)')
    parser.add_argument('--ydim', dest='ydim', required=True, type=int, metavar='INTEGER', help='Output y dimension (px)')
    parser.add_argument('--genvol', dest='genvol', action='store_true', help='Transform and write out volume')
    parser.add_argument('--flipZ', dest='flipZ', action='store_true', help='Flip Z in volume output')
    parser.add_argument('--suffix', dest='suffix', metavar='STRING', action='store', help='Suffix for volume output')
    args = parser.parse_args()

    cfFiles = []
    for i, cfFile in enumerate(args.cfFiles):
        cfFiles.append(os.path.abspath(cfFile))
    xdim = args.xdim
    ydim = args.ydim
    genvol = args.genvol
    suffix = args.suffix
    flipZ = args.flipZ

    m = np.identity(4)
    m_inv = np.identity(4)
    for pair in args.matrices:
        m = pair[0] @ m
        m_inv = m_inv @ pair[1]

    print('\n# Transformation Matrix (Forward) #\n')
    print(m)

    print('\n\n# Inverse Matrix (Inverse) #\n')
    print(m_inv)
    print('\n')

    workdir = os.path.dirname(cfFiles[0])

    # Loop through fluorescence volumes, transform and save
    if genvol:
        app = QtWidgets.QApplication(sys.argv)
        for cfFile in cfFiles:
            CF = tifffile.imread(cfFile)
            if CF.dtype != np.uint8:
                if CF.dtype in [np.uint16, np.uint32, np.uint64]:
                    CF = normalized_uint8(CF)
                else:
                    print('Error: unsupported bit depth.')
            out_vol, z_offset = transform_volume_single(CF,m,m_inv,xdim,ydim)
            prefix = '.'.join(os.path.basename(cfFile).split('.')[0:-1])
            if suffix:
                savepath = os.path.normpath(os.path.join(workdir,'_'.join([prefix,suffix,'transformed.tif'])))
            else:
                savepath = os.path.normpath(os.path.join(workdir,'_'.join([prefix,'transformed.tif'])))
            if flipZ:
                out_vol = np.flip(out_vol, axis=0)
            print('Writing out volume...')
            tifffile.imwrite(savepath,out_vol,photometric='minisblack',compress=0)
            print('Volume saved as: '+savepath)
            if flipZ:
                pass
            else:
                print(f'Original origin at z = {z_offset:.3f} px')
