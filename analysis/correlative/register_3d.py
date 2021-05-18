#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Affine registration of 3D coordinates using multiple approaches

Requires CSV files containing x,y,z coordinates of matching points (fluo_csv and sav_csv)
The transformation from fluo_csv to sav_csv is calculated. The following approaches are used.

(1) OpenCV RANSAC implmentation
(2) L-BFGS-B-based local minimization with SciPy based on different
    starting positions generated with TEASER++ by Yang et al.
    (https://github.com/MIT-SPARK/TEASER-plusplus)
    Building of Python extension module required.
(3) Basin-hopping global optimization with L-BFGS-B local minimization.

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
from scipy import spatial, linalg
from scipy.optimize import minimize, basinhopping
from scipy.spatial.transform import Rotation
import xml.etree.ElementTree as ET
import cv2

from tools3dct.core import scale, translate

import teaserpp_python

## Transforms ##

def iso_transform(s,tx,ty,tz,a,b,c,d):

    tr = translate(tx,ty,tz)
    tr_inv = translate(-1*tx,-1*ty,-1*tz)

    sca = scale(s)
    sca_inv = scale(1/s)

    r = Rotation.from_quat([a,b,c,d])
    rot = np.identity(4)
    rot[:-1,:-1] = r.as_matrix()
    rot_inv = rot.T

    m = tr @ sca @ rot
    m_inv = rot_inv @ sca_inv @ tr_inv

    return m, m_inv

def iso_Z_transform(s1,s,tx,ty,tz,a,b,c,d):

    tr = translate(tx,ty,tz)
    tr_inv = translate(-1*tx,-1*ty,-1*tz)

    sca = scale(s)
    sca_inv = scale(1/s)

    sca1 = [[1,0,0,0],[0,1,0,0],[0,0,s1,0],[0,0,0,1]]
    sca1_inv = [[1,0,0,0],[0,1,0,0],[0,0,1/s1,0],[0,0,0,1]]

    r = Rotation.from_quat([a,b,c,d])
    rot = np.identity(4)
    rot[:-1,:-1] = r.as_matrix()
    rot_inv = rot.T

    m = sca1 @ tr @ sca @ rot
    m_inv = rot_inv @ sca_inv @ tr_inv @ sca1_inv

    return m, m_inv

def aniso_transform(sx,sy,sz,tx,ty,tz,a,b,c,d):

    tr = translate(tx,ty,tz)
    tr_inv = translate(-1*tx,-1*ty,-1*tz)

    sca = [[sx,0,0,0],[0,sy,0,0],[0,0,sz,0],[0,0,0,1]]
    sca_inv = [[1/sx,0,0,0],[0,1/sy,0,0],[0,0,1/sz,0],[0,0,0,1]]

    r = Rotation.from_quat([a,b,c,d])
    rot = np.identity(4)
    rot[:-1,:-1] = r.as_matrix()
    rot_inv = rot.T

    m = tr @ sca @ rot
    m_inv = rot_inv @ sca_inv @ tr_inv

    return m, m_inv

def aniso_shearY_transform(sx,sy,sz,shzy,tx,ty,tz,a,b,c,d):

    tr = translate(tx,ty,tz)
    tr_inv = translate(-1*tx,-1*ty,-1*tz)

    sca = [[sx,0,0,0],[0,sy,0,0],[0,0,sz,0],[0,0,0,1]]
    sca_inv = [[1/sx,0,0,0],[0,1/sy,0,0],[0,0,1/sz,0],[0,0,0,1]]

    sh = [[1,0,0,0],[0,1,shzy,0],[0,0,1,0],[0,0,0,1]]
    sh_inv = [[1,0,0,0],[0,1,-shzy,0],[0,0,1,0],[0,0,0,1]]

    r = Rotation.from_quat([a,b,c,d])
    rot = np.identity(4)
    rot[:-1,:-1] = r.as_matrix()
    rot_inv = rot.T

    m = tr @ sh @ sca @ rot
    m_inv = rot_inv @ sca_inv @ sh_inv @ tr_inv

    return m, m_inv

def full_affine_transform(sx,sy,sz,shyx,shzx,shxy,shzy,shxz,shyz,tx,ty,tz,a,b,c,d):

    tr = translate(tx,ty,tz)
    tr_inv = translate(-1*tx,-1*ty,-1*tz)

    sca = [[sx,0,0,0],[0,sy,0,0],[0,0,sz,0],[0,0,0,1]]
    sca_inv = [[1/sx,0,0,0],[0,1/sy,0,0],[0,0,1/sz,0],[0,0,0,1]]

    sh = [[1,shyx,shzx,0],[shxy,1,shzy,0],[shxz,shyz,1,0],[0,0,0,1]]
    sh_inv = [[1,-shyx,-shzx,0],[-shxy,1,-shzy,0],[-shxz,-shyz,1,0],[0,0,0,1]]

    r = Rotation.from_quat([a,b,c,d])
    rot = np.identity(4)
    rot[:-1,:-1] = r.as_matrix()
    rot_inv = rot.T

    m = tr @ sh @ sca @ rot
    m_inv = rot_inv @ sca_inv @ sh_inv @ tr_inv

    return m, m_inv


## Objective functions returning sum of residuals ##

def rigid_objective_function(x,s,fluo_pts,sav_pts):
    m, m_inv = iso_transform(s,*x)
    fluo_pts_transformed = m @ np.vstack([fluo_pts.T, np.ones(fluo_pts.shape[0])])
    ssr = np.sum(spatial.minkowski_distance_p(sav_pts,(fluo_pts_transformed[:-1,:]).T))
    return ssr

def iso_objective_function(x,fluo_pts,sav_pts):
    m, m_inv = iso_transform(*x)
    fluo_pts_transformed = m @ np.vstack([fluo_pts.T, np.ones(fluo_pts.shape[0])])
    ssr = np.sum(spatial.minkowski_distance_p(sav_pts,(fluo_pts_transformed[:-1,:]).T))
    return ssr

def iso_Z_objective_function(x,fluo_pts,sav_pts):
    m, m_inv = iso_Z_transform(*x)
    fluo_pts_transformed = m @ np.vstack([fluo_pts.T, np.ones(fluo_pts.shape[0])])
    ssr = np.sum(spatial.minkowski_distance_p(sav_pts,(fluo_pts_transformed[:-1,:]).T))
    return ssr

def aniso_objective_function(x,fluo_pts,sav_pts):
    m, m_inv = aniso_transform(*x)
    fluo_pts_transformed = m @ np.vstack([fluo_pts.T, np.ones(fluo_pts.shape[0])])
    ssr = np.sum(spatial.minkowski_distance_p(sav_pts,(fluo_pts_transformed[:-1,:]).T))
    return ssr

def aniso_shearY_objective_function(x,fluo_pts,sav_pts):
    m, m_inv = aniso_shearY_transform(*x)
    fluo_pts_transformed = m @ np.vstack([fluo_pts.T, np.ones(fluo_pts.shape[0])])
    ssr = np.sum(spatial.minkowski_distance_p(sav_pts,(fluo_pts_transformed[:-1,:]).T))
    return ssr

def full_affine_objective_function(x,fluo_pts,sav_pts):
    m, m_inv = full_affine_transform(*x)
    fluo_pts_transformed = m @ np.vstack([fluo_pts.T, np.ones(fluo_pts.shape[0])])
    ssr = np.sum(spatial.minkowski_distance_p(sav_pts,(fluo_pts_transformed[:-1,:]).T))
    return ssr


## Basin-hopping step class ##

class FullAffineStep(object):
   def __init__(self, stepsize=0.5):
       self.stepsize = stepsize
   def __call__(self, x):
       s = self.stepsize
       x[0:3] *= 1.01**np.random.uniform(-s, s)
       x[3:9] += np.random.uniform(-0.001*s, 0.001*s, x[3:9].shape)
       x[9:12] += np.random.uniform(-100.0*s, 100.0*s, x[9:12].shape)
       x[12:16] += np.random.uniform(-0.1*s, 0.1*s, x[12:16].shape)
       return x


## I/O ##

def write_matrix_xml(mat, xmlFile):
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

    # Retrieve points, unflipped and scaling not corrected
    fluo_csv = '/Users/kf656/Desktop/Current/sav2/fluo_ld_round5_noshear_final.csv'
    sav_csv = '/Users/kf656/Desktop/Current/sav2/sav_ld_round5_noshear_final.csv'
    fluo_pts = np.loadtxt(fluo_csv, delimiter=',')
    sav_pts = np.loadtxt(sav_csv, delimiter=',')

    # Retrieve second set of points, concatenate
    fluo_csv = '/Users/kf656/Desktop/Current/sav2/fluo_bead_noshear_final.csv'
    sav_csv = '/Users/kf656/Desktop/Current/sav2/sav_bead_noshear_final.csv'
    fluo_pts2 = np.loadtxt(fluo_csv, delimiter=',')
    sav_pts2 = np.loadtxt(sav_csv, delimiter=',')
    fluo_pts = np.concatenate((fluo_pts,fluo_pts2))
    sav_pts = np.concatenate((sav_pts,sav_pts2))

    # Remove outliers
    idx = []
    # idx = [38,46,32,24,33,12,20]
    # idx = [38,46,32,24,33,12,20,101,60,63,58]
    if len(idx) > 0:
        fluo_pts = np.delete(fluo_pts,idx,0)
        sav_pts = np.delete(sav_pts,idx,0)

    # Filter out points less than 2 frames from ends of SAV volume
    boolean_filter = (sav_pts[:,2] > 2) & (sav_pts[:,2] < (63-1-2))
    fluo_pts = fluo_pts[boolean_filter,:]
    sav_pts = sav_pts[boolean_filter,:]

    # Correct flip and z-spacing, check pixel size
    # fluo_pts[:,2] = 109 - fluo_pts[:,2]  # flip z for fluo
    # sav_pts[:,2] = (63 - sav_pts[:,2]) * 50 / 19.2708  # flip z for sav
    sav_pts[:,2] = sav_pts[:,2] * 100 / 19.2708  # no flip in sav

    # Set pixel size
    fluo_pxsize = 110  # in nm
    sav_pxsize = 19.2708

    # Fit results
    summaryfile = open('fit_summary.txt', 'w')

    ## OpenCV Affine3D RANSAC ##
    retval, m0, inliers = cv2.estimateAffine3D(fluo_pts,sav_pts)
    m = np.identity(4)
    m[:-1,:] = m0
    fluo_pts_transformed = m @ np.vstack([fluo_pts.T, np.ones(fluo_pts.shape[0])])
    ssr = np.sum(spatial.minkowski_distance_p(sav_pts,(fluo_pts_transformed[:-1,:]).T))
    rmsd0 = np.sqrt(ssr/fluo_pts.shape[0])
    # Decomposition based on RQ decomposition, A = SKQ, where Q is rotation matrix, K is scaling matrix and S is shear matrix
    R, Q = linalg.rq(m[:-1,:-1])
    K = np.diag(np.diag(R))
    S = R @ np.diag([1/x for x in np.diag(R)])
    if K[0,0] < 0 and K[1,1] < 0:
        K = K @ np.diag([-1,-1,1])
        Q = np.diag([-1,-1,1]) @ Q
    elif K[1,1] < 0 and K[2,2] < 0:
        K = K @ np.diag([1,-1,-1])
        Q = np.diag([1,-1,-1]) @ Q
    r_euler = Rotation.from_matrix(Q).as_euler('zxz',degrees=True)
    outstring = ('OpenCV Affine 3D, RANSAC\n',
          f'  rotation:  {r_euler[0]:.1f}, {r_euler[1]:.1f}, {r_euler[2]:.1f}\n',
          f'  scale:  {K[0,0]:.3f}, {K[1,1]:.3f}, {K[2,2]:.3f}\n',
          f'  shear:  {S[0,1]:.3f}, {S[0,2]:.3f}, {S[1,0]:.3f}, {S[1,2]:.3f}, {S[2,0]:.3f}, {S[2,1]:.3f}\n',
          f'  translation:  {m[0,3]:.1f}, {m[1,3]:.1f}, {m[2,3]:.1f}\n',
          f'  rmsd:  {rmsd0:.3f}\n')
    summaryfile.write(''.join(outstring))

    # Initialise results
    res_rigid = None
    res_iso = None
    res_iso_Z = None
    res_aniso = None
    res_aniso_shearY = None
    res_full_affine = None

    # Set bounds
    bounds_rigid = [(-np.inf,np.inf), (-np.inf,np.inf), (-np.inf,np.inf), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi), (-1, 1)]
    bounds_iso = [(1e-12,np.inf), (-np.inf,np.inf), (-np.inf,np.inf), (-np.inf,np.inf), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi), (-1, 1)]
    bounds_iso_Z = [(1e-12,np.inf),(1e-12,np.inf), (-np.inf,np.inf), (-np.inf,np.inf), (-np.inf,np.inf), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi), (-1, 1)]
    bounds_aniso = [(1e-12,np.inf),(1e-12,np.inf),(1e-12,np.inf), (-np.inf,np.inf), (-np.inf,np.inf), (-np.inf,np.inf), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi), (-1, 1)]
    bounds_aniso_shearY = [(1e-12,np.inf),(1e-12,np.inf),(1e-12,np.inf), (-np.inf,np.inf), (-np.inf,np.inf), (-np.inf,np.inf), (-np.inf,np.inf), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi), (-1, 1)]
    bounds_full_affine = [(1e-12,np.inf),(1e-12,np.inf),(1e-12,np.inf), (-np.inf,np.inf), (-np.inf,np.inf), (-np.inf,np.inf), (-np.inf,np.inf), (-np.inf,np.inf), (-np.inf,np.inf), (-np.inf,np.inf), (-np.inf,np.inf), (-np.inf,np.inf), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi), (-1, 1)]

    # Local optimisation, L-BFGS-B, full affine, based on OpenCV Affine3D RANSAC estimates
    r0_quat = Rotation.from_matrix(Q).as_quat()
    x0_full_affine = [K[0,0],K[1,1],K[2,2],S[0,1],S[0,2],S[1,0],S[1,2],S[2,0],S[2,1],m0[0,3],m0[1,3],m0[2,3],*r0_quat]
    res_full_affine = minimize(full_affine_objective_function, x0_full_affine, args=(fluo_pts,sav_pts), bounds=bounds_full_affine, method='L-BFGS-B')

    # Basin-hopping with L-BFGS-B local minimzation, full affine, based on OpenCV Affine3D RANSAC estimates (perturbed already in first round)
    custom_step_callable = FullAffineStep()
    res_full_affine_bh = basinhopping(full_affine_objective_function, x0_full_affine, niter=200, T=1, stepsize=0.5, take_step=custom_step_callable, minimizer_kwargs={'method':'L-BFGS-B', 'args':(fluo_pts,sav_pts),'bounds':bounds_full_affine})

    # Generate initial transforms with TEASER 
    for noise_bound_factor in [0.5, 1, 2, 3, 4, 5]:

        # ## TEASER  ##
        
        solver_params = teaserpp_python.RobustRegistrationSolver.Params()
        solver_params.cbar2 = 1
        solver_params.noise_bound = rmsd0 * noise_bound_factor
        solver_params.estimate_scaling = True
        solver_params.rotation_estimation_algorithm = (teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS)
        solver_params.rotation_gnc_factor = 1.4
        solver_params.rotation_max_iterations = 1000
        solver_params.rotation_cost_threshold = 1e-6
        teaserpp_solver = teaserpp_python.RobustRegistrationSolver(solver_params)

        solver = teaserpp_python.RobustRegistrationSolver(solver_params)
        solver.solve(fluo_pts.T, sav_pts.T)

        solution = solver.getSolution()

        s0 = solution.scale
        trans0 = solution.translation
        r = Rotation.from_matrix(solution.rotation)
        r0 = r.as_quat()
        m, m_inv = iso_transform(s0,*trans0,*r0)
        fluo_pts_transformed = m @ np.vstack([fluo_pts.T, np.ones(fluo_pts.shape[0])])
        ssr = np.sum(spatial.minkowski_distance_p(sav_pts,(fluo_pts_transformed[:-1,:]).T))
        rmsd = np.sqrt(ssr/fluo_pts.shape[0])
        r0_euler = r.as_euler('zxz',degrees=True)
        outstring = ('TEASER\n',
            f'  rotation:  {r0_euler[0]:.1f}, {r0_euler[1]:.1f}, {r0_euler[2]:.1f}\n',
            f'  scale:  {s0:.3f}\n',
            f'  translation:  {trans0[0]:.1f}, {trans0[1]:.1f}, {trans0[2]:.1f}\n',
            f'  rmsd:  {rmsd:.3f}\n')
        summaryfile.write(''.join(outstring))

        ## Least squares ##

        # Set initial parameters
        # s0 = fluo_pxsize/sav_pxsize
        # trans0 = [0,0,0]
        # trans0 = [-1246.9, -5045.1, 314.4]
        # r0 = Rotation.from_euler('zxz', [-157.3, 9.6, -174.7], degrees=True).as_quat()
        x0_rigid = [*trans0,*r0]
        x0_iso = [s0,*trans0,*r0]
        x0_iso_Z = [1,s0,*trans0,*r0]
        x0_aniso = [s0,s0,s0,*trans0,*r0]
        x0_aniso_shearY = [s0,s0,s0,0,*trans0,*r0]
        x0_full_affine = [K[0,0],K[1,1],K[2,2],S[0,1],S[0,2],S[1,0],S[1,2],S[2,0],S[2,1],m0[0,3],m0[1,3],m0[2,3],*r0]

        # Local optimisation, L-BFGS-B, rigid
        res = minimize(rigid_objective_function, x0_rigid, args=(s0,fluo_pts,sav_pts), bounds=bounds_rigid, method='L-BFGS-B')
        if res_rigid is None:
            res_rigid = res
        elif res.fun < res_rigid.fun:
            res_rigid = res

        # Local optimisation, L-BFGS-B, iso
        res = minimize(iso_objective_function, x0_iso, args=(fluo_pts,sav_pts), bounds=bounds_iso, method='L-BFGS-B')
        if res_iso is None:
            res_iso = res
        elif res.fun < res_iso.fun:
            res_iso = res

        # # Local optimisation, L-BFGS-B, iso_Z
        # res = minimize(iso_Z_objective_function, x0_iso_Z, args=(fluo_pts,sav_pts), bounds=bounds_iso_Z, method='L-BFGS-B')
        # if res_iso_Z is None:
        #     res_iso_Z = res
        # elif res.fun < res_iso_Z.fun:
        #     res_iso_Z = res

        # Local optimisation, L-BFGS-B, aniso
        res = minimize(aniso_objective_function, x0_aniso, args=(fluo_pts,sav_pts), bounds=bounds_aniso, method='L-BFGS-B')
        if res_aniso is None:
            res_aniso = res
        elif res.fun < res_aniso.fun:
            res_aniso = res

        # # Local optimisation, L-BFGS-B, aniso+shear
        # res = minimize(aniso_shearY_objective_function, x0_aniso_shearY, args=(fluo_pts,sav_pts), bounds=bounds_aniso_shearY, method='L-BFGS-B')
        # if res_aniso_shearY is None:
        #     res_aniso_shearY = res
        # elif res.fun < res_aniso_shearY.fun:
        #     res_aniso_shearY = res

        # Local optimisation, L-BFGS-B, full affine
        res = minimize(full_affine_objective_function, x0_full_affine, args=(fluo_pts,sav_pts), bounds=bounds_full_affine, method='L-BFGS-B')
        if res_full_affine is None:
            res_full_affine = res
        elif res.fun < res_full_affine.fun:
            res_full_affine = res

    x = res_rigid.x
    r_euler = Rotation.from_quat(x[3:7]).as_euler('zxz',degrees=True)
    outstring = ('Rigid, L-BFGS-B\n',
    f'  rotation:  {r_euler[0]:.1f}, {r_euler[1]:.1f}, {r_euler[2]:.1f}\n',
    f'  scale:  {s0:.3f}\n',
    f'  translation:  {x[0]:.1f}, {x[1]:.1f}, {x[2]:.1f}\n',
    f'  rmsd:  {np.sqrt(res_rigid.fun/fluo_pts.shape[0]):.3f}\n')
    summaryfile.write(''.join(outstring))
    x = res_iso.x
    r_euler = Rotation.from_quat(x[4:8]).as_euler('zxz',degrees=True)
    outstring = ('Iso, L-BFGS-B\n',
    f'  rotation:  {r_euler[0]:.1f}, {r_euler[1]:.1f}, {r_euler[2]:.1f}\n',
    f'  scale:  {x[0]:.3f}\n',
    f'  translation:  {x[1]:.1f}, {x[2]:.1f}, {x[3]:.1f}\n',
    f'  rmsd:  {np.sqrt(res_iso.fun/fluo_pts.shape[0]):.3f}\n')
    summaryfile.write(''.join(outstring))
    # x = res_iso_Z.x
    # r_euler = Rotation.from_quat(x[5:9]).as_euler('zxz',degrees=True)
    # outstring = ('Iso Z, L-BFGS-B\n',
    # f'  rotation:  {r_euler[0]:.1f}, {r_euler[1]:.1f}, {r_euler[2]:.1f}\n',
    # f'  scale:  {x[1]:.3f}, {x[0]:.3f}\n',
    # f'  translation:  {x[2]:.1f}, {x[3]:.1f}, {x[4]:.1f}\n',
    # f'  rmsd:  {np.sqrt(res_iso_Z.fun/fluo_pts.shape[0]):.3f}\n')
    # summaryfile.write(''.join(outstring))
    x = res_aniso.x
    r_euler = Rotation.from_quat(res_aniso.x[6:10]).as_euler('zxz',degrees=True)
    outstring = ('Aniso, L-BFGS-B\n',
    f'  rotation:  {r_euler[0]:.1f}, {r_euler[1]:.1f}, {r_euler[2]:.1f}\n',
    f'  scale:  {x[0]:.3f}, {x[1]:.3f}, {x[2]:.3f}\n',
    f'  translation:  {x[3]:.1f}, {x[4]:.1f}, {x[5]:.1f}\n',
    f'  rmsd:  {np.sqrt(res_aniso.fun/fluo_pts.shape[0]):.3f}\n')
    summaryfile.write(''.join(outstring))
    # x = res_aniso_shearY.x
    # r_euler = Rotation.from_quat(x[7:11]).as_euler('zxz',degrees=True)
    # outstring = ('Aniso + ShearY, L-BFGS-B\n',
    #     f'  rotation:  {r_euler[0]:.1f}, {r_euler[1]:.1f}, {r_euler[2]:.1f}\n',
    #     f'  scale:  {x[0]:.3f}, {x[1]:.3f}, {x[2]:.3f}\n',
    #     f'  shear:  {x[3]:.3f}\n',
    #     f'  translation:  {x[4]:.1f}, {x[5]:.1f}, {x[6]:.1f}\n',
    #     f'  rmsd:  {np.sqrt(res_aniso_shearY.fun/fluo_pts.shape[0]):.3f}\n')
    # summaryfile.write(''.join(outstring))
    x = res_full_affine.x
    r_euler = Rotation.from_quat(x[12:16]).as_euler('zxz',degrees=True)
    outstring = ('Full affine, L-BFGS-B\n',
    f'  rotation:  {r_euler[0]:.1f}, {r_euler[1]:.1f}, {r_euler[2]:.1f}\n',
    f'  scale:  {x[0]:.3f}, {x[1]:.3f}, {x[2]:.3f}\n',
    f'  shear:  {x[3]:.3f}, {x[4]:.3f}, {x[5]:.3f}, {x[6]:.3f}, {x[7]:.3f}, {x[8]:.3f}\n',
    f'  translation:  {x[9]:.1f}, {x[10]:.1f}, {x[11]:.1f}\n',
    f'  rmsd:  {np.sqrt(res_full_affine.fun/fluo_pts.shape[0]):.3f}\n')
    summaryfile.write(''.join(outstring))
    x = res_full_affine_bh.x
    r_euler = Rotation.from_quat(x[12:16]).as_euler('zxz',degrees=True)
    outstring = ('Full affine, basin-hopping / L-BFGS-B\n',
    f'  rotation:  {r_euler[0]:.1f}, {r_euler[1]:.1f}, {r_euler[2]:.1f}\n',
    f'  scale:  {x[0]:.3f}, {x[1]:.3f}, {x[2]:.3f}\n',
    f'  shear:  {x[3]:.3f}, {x[4]:.3f}, {x[5]:.3f}, {x[6]:.3f}, {x[7]:.3f}, {x[8]:.3f}\n',
    f'  translation:  {x[9]:.1f}, {x[10]:.1f}, {x[11]:.1f}\n',
    f'  rmsd:  {np.sqrt(res_full_affine_bh.fun/fluo_pts.shape[0]):.3f}\n')
    summaryfile.write(''.join(outstring))

    summaryfile.write(f'n = {fluo_pts.shape[0]}')

    summaryfile.close()

    # Select optimisation result and apply transform to fluorescence points
    M = []
    # M_inv = []
    # rootnames = ['fit_combined_rigid','fit_combined_iso','fit_combined_iso_Z','fit_combined_aniso','fit_combined_aniso_shearY','fit_combined_full_affine']
    rootnames = ['fit_combined_rigid','fit_combined_iso','fit_combined_aniso','fit_combined_full_affine','fit_combined_full_affine_bh']
    x = res_rigid.x
    m, m_inv = iso_transform(s0,*x)
    M.append(m)
    # M_inv.append(m_inv)
    x = res_iso.x
    m, m_inv = iso_transform(*x)
    M.append(m)
    # M_inv.append(m_inv)
    # x = res_iso_Z.x
    # m, m_inv = iso_Z_transform(*x)
    # M.append(m)
    # # M_inv.append(m_inv)
    x = res_aniso.x
    m, m_inv = aniso_transform(*x)
    M.append(m)
    # M_inv.append(m_inv)
    # x = res_aniso_shearY.x
    # m, m_inv = aniso_shearY_transform(*x)
    # M.append(m)
    # # M_inv.append(m_inv)
    if np.sqrt(res_full_affine.fun/fluo_pts.shape[0]) < rmsd0:
        x = res_full_affine.x
        m, m_inv = full_affine_transform(*x)
    else:
        m = np.identity(4)
        m[:-1,:] = m0  # take OpenCV Affine3D RANSAC result
        # m_inv not calculated
    M.append(m)
    # M_inv.append(m_inv)
    x = res_full_affine_bh.x
    m, m_inv = full_affine_transform(*x)
    M.append(m)
    # # M_inv.append(m_inv)

    for k, rootname in enumerate(rootnames):
        
        # Transform FLUO to SAV and write points
        fluo_pts_transformed = M[k] @ np.vstack([fluo_pts.T, np.ones(fluo_pts.shape[0])])
        fluo_pts_transformed = fluo_pts_transformed[:-1,:].T

        # Projection of residuals in fluorescence space
        u = m @ np.array([[1,0,0,1],[0,1,0,1],[0,0,1,1]]).T
        v = m @ np.array([[0,0,0,1],[0,0,0,1],[0,0,0,1]]).T
        n = u - v
        nx = n[:-1,0]
        ny = n[:-1,1]
        nz = n[:-1,2]
        nx = nx / np.linalg.norm(nx)
        ny = ny / np.linalg.norm(ny)
        nz = nz / np.linalg.norm(nz)
        resi_sav = sav_pts - fluo_pts_transformed
        resi_fluo_x = np.einsum('ij, ij->i', resi_sav, np.tile(nx,(len(resi_sav),1)))
        resi_fluo_y = np.einsum('ij, ij->i', resi_sav, np.tile(ny,(len(resi_sav),1)))
        resi_fluo_z = np.einsum('ij, ij->i', resi_sav, np.tile(nz,(len(resi_sav),1)))

        # Write csv
        np.savetxt(
            rootname+'_savAxes.csv',
            np.c_[fluo_pts_transformed,sav_pts,fluo_pts,resi_fluo_x,resi_fluo_y,resi_fluo_z],
            delimiter=',',
            header='fluo_x_transformed,fluo_y_transformed,fluo_z_transformed,sav_x,sav_y,sav_z,fluo_x,fluo_y,fluo_z,resi_fluo_x,resi_fluo_y,resi_fluo_z')

        # Write xml
        write_matrix_xml(M[k], rootname+'_matrix.xml')

        # # Transform SAV to FLUO (inverse transform) and write points
        # sav_pts_transformed = M_inv[k] @ np.vstack([sav_pts.T, np.ones(sav_pts.shape[0])])
        # sav_pts_transformed = sav_pts_transformed[:-1,:].T
        # np.savetxt(
        #     rootname+'_fluoAxes.csv',
        #     np.c_[fluo_pts,sav_pts_transformed],
        #     delimiter=',',
        #     header='#fluo_x,fluo_y,fluo_z,sav_x_transformed,sav_y_transformed,sav_z_transformed')
