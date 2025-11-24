#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 15:30:55 2023

@author: Javiera Jilberto Vallejos
"""

import numpy as np
import meshio as io
import cheartio as chio
from uvcgen.uvc_utils import create_submesh, create_submesh_bdata, find_isoline, \
    get_normal_plane_svd, get_surface_mesh

from scipy.spatial import KDTree
import pymmg

class generalUVC:
    def __init__(self, mesh, bdata, boundaries, thresholds, out_folder, rvlv=None, subdomains=None):
        cells = mesh.cells[0]
        self.xyz, self.ien, self.elem = mesh.points, cells.data, cells.type
        self.bv_mesh = mesh
        self.bv_bdata = bdata
        self.bv_bdata_og = np.copy(bdata)

        self.thresholds = thresholds
        self.patches = boundaries
        self.mesh_folder = out_folder
        self.out_folder = out_folder

        # Check if AV exists in boundary data
        self.has_av = 'av' in self.patches

        # Computing a bunch of needed stuff
        self.get_mesh_size()
        self.get_valve_info()
        self.define_axis_vectors()

        # Optional inputs
        self.split_epi = False
        if rvlv is not None:
            self.bv_mesh.cell_data['rvlv'] = [rvlv]
            self.split_epi = True
        if subdomains is not None:
            self.bv_mesh.cell_data['subdomains'] = [subdomains]

        # Dictionary for easiness of handling
        self.meshes = {'bv': self.bv_mesh}
        self.bdatas = {'bv': self.bv_bdata}

        # flags
        if ('lv_epi' in self.patches) and('rv_epi' in self.patches):
            self.split_epi = bool(True + self.split_epi)
        elif 'epi' in self.patches:
            self.split_epi = bool(False + self.split_epi)
        else:
            raise('boundaries at epi can be either "epi" or "lv_epi" + "rv_epi"')
        self.truncated = False

    def get_mesh_size(self):
        elems = np.random.randint(0, self.ien.shape[0], size=20)    # Randomly taking 20 elements
        xyz = self.xyz[self.ien[elems]]

        l0 = np.linalg.norm(xyz[:,1,:]-xyz[:,0,:], axis=1)
        self.mesh_size = np.mean(l0)

    def get_base_valve(self):
        """Return 'av' if it exists, otherwise 'mv'."""
        return 'av' if self.has_av else 'mv'

    def get_valve_info(self):
        # Get centroids
        self.valve_centroids = {}
        self.valve_normals = {}
        # Only process valves that exist in patches
        valves_to_process = [v for v in ['av', 'mv', 'pv', 'tv'] if v in self.patches]
        for v in valves_to_process:
            nodes = np.unique(self.bv_bdata[self.bv_bdata[:,-1] == self.patches[v], 1:-1])
            centroid = np.mean(self.xyz[nodes], axis=0)
            self.valve_centroids[v] = centroid

            if v == 'mv':   # For the mv we need to get rid of the bridge nodes to compute the normal
                dist_mv = np.linalg.norm(self.xyz[nodes] - self.valve_centroids['mv'], axis=1)
                if self.has_av:
                    dist_av = np.linalg.norm(self.xyz[nodes] - self.valve_centroids['av'], axis=1)
                    dist_vv = np.linalg.norm(self.valve_centroids['mv'] - self.valve_centroids['av'])*1.3
                    between_nodes = (dist_av < dist_vv) * (dist_mv < dist_vv)
                    nodes = nodes[~between_nodes]
                    centroid = np.mean(self.xyz[nodes], axis=0)
                    self.mv_aux_centroid = centroid
                else:
                    # If no AV, just use MV nodes as-is
                    self.mv_aux_centroid = centroid

            # Fitting a plane to valve nodes
            svd = np.linalg.svd(self.xyz[nodes] - centroid)
            self.valve_normals[v] = svd[2][-1]

        # Flip if needed so normals are pointing outwards
        lv_endo_points = np.unique(self.bv_bdata[self.bv_bdata[:,-1] == self.patches['lv_endo'], 1:-1])
        rv_endo_points = np.unique(self.bv_bdata[self.bv_bdata[:,-1] == self.patches['rv_endo'], 1:-1])
        lv_cent = np.mean(self.xyz[lv_endo_points], axis=0)
        rv_cent = np.mean(self.xyz[rv_endo_points], axis=0)

        self.lv_centroid = lv_cent
        self.rv_centroid = rv_cent

        if self.has_av:
            if np.dot(self.valve_centroids['av']-lv_cent, self.valve_normals['av']) < 0:
                self.valve_normals['av'] = -self.valve_normals['av']
        if np.dot(self.valve_centroids['mv']-lv_cent, self.valve_normals['mv']) < 0:
            self.valve_normals['mv'] = -self.valve_normals['mv']
        if np.dot(self.valve_centroids['tv']-rv_cent, self.valve_normals['tv']) < 0:
            self.valve_normals['tv'] = -self.valve_normals['tv']
        if np.dot(self.valve_centroids['pv']-rv_cent, self.valve_normals['pv']) < 0:
            self.valve_normals['pv'] = -self.valve_normals['pv']


    def define_axis_vectors(self, method = 'orthogonal'):
        # long axis vector
        if method == 'all':
            self.long_axis_vector, valves_centroid = \
                get_normal_plane_svd(np.vstack(list(self.valve_centroids.values())))

        elif method == 'mv':
            self.long_axis_vector = self.valve_normals['mv']

        elif method == 'orthogonal':
            from scipy.optimize import minimize
            def pnorm(v, p):
                return np.sum(np.abs(v)**p)**(1/p)
            def func(la_vec, normals, p=1.373):
                la_vec = la_vec/np.linalg.norm(la_vec)
                dot_p = normal@la_vec
                return pnorm(dot_p, p)

            lv_endo_ien = self.bv_bdata[self.bv_bdata[:,-1] == self.patches['lv_endo'], 1:-1]

            v1 = self.xyz[lv_endo_ien[:,1]] - self.xyz[lv_endo_ien[:,0]]
            v2 = self.xyz[lv_endo_ien[:,2]] - self.xyz[lv_endo_ien[:,0]]

            normal = np.cross(v1, v2, axisa=1, axisb=1)
            normal = normal/np.linalg.norm(normal, axis=1)[:,None]

            sol = minimize(func, np.array([1,0,0]), args=(normal))

            if np.dot(sol.x, self.valve_normals['mv']) < 0:
                self.long_axis_vector = -sol.x/np.linalg.norm(sol.x)
            else:
                self.long_axis_vector = sol.x/np.linalg.norm(sol.x)

        else:
            raise 'Must specify a valid method (all or mv)'

        # Septum vector
        sep_nodes = np.unique(self.bv_bdata[self.bv_bdata[:,-1]==self.patches['rv_septum'],1:-1])
        septum_vector, self.septum_centroid = get_normal_plane_svd(self.xyz[sep_nodes])
        aux = np.cross(septum_vector, self.long_axis_vector)
        septum_vector = np.cross(self.long_axis_vector, aux)

        # Correct direction (septum vector should point out of the LV)
        lv_endo_nodes = np.unique(self.bv_bdata[self.bv_bdata[:,-1] == self.patches['lv_endo'], 1:-1])
        aux_vector = self.septum_centroid - np.mean(self.xyz[lv_endo_nodes], axis=0)
        if np.dot(septum_vector, aux_vector) < 0:
            self.septum_vector = -septum_vector/np.linalg.norm(septum_vector)
        else:
            self.septum_vector = septum_vector/np.linalg.norm(septum_vector)

        # third vector
        self.third_vector = np.cross(self.septum_vector, self.long_axis_vector)


    def compute_septum(self):
        if not 'rvlv' in self.bv_mesh.cell_data:
            raise('compute_septum() is only available if an rvlv split file was provided')

        marker = self.bv_mesh.cell_data['rvlv'][0]
        ien = self.ien

        enodes = ien.flatten()
        elem_id = np.repeat(np.arange(len(ien)),4)
        order = np.argsort(enodes)
        _, ind = np.unique(np.sort(enodes), return_index=True)

        marker_nodes = np.zeros(len(self.xyz))
        for i in range(len(self.xyz)):
            if (i+1) > (len(self.xyz)-1):
                vals = marker[elem_id[order[ind[i]:]]]
            else:
                vals = marker[elem_id[order[ind[i]:ind[i+1]]]]
            marker_nodes[i] = (np.max(vals) + np.min(vals))/2

        self.bv_mesh.point_data['septum'] = marker_nodes

        return marker_nodes


    def define_apex_nodes(self):
        xyz = self.lv_mesh.points

        # Finding septum apex nodes
        if self.split_epi:
            lv_epi_nodes = np.unique(self.lv_bdata[self.lv_bdata[:,-1]==self.patches['lv_epi'],1:-1])
        else:
            lv_epi_nodes = np.unique(self.lv_bdata[self.lv_bdata[:,-1]==self.patches['epi'],1:-1])

        # Epi_apex is the LV epicardial node farthest from the MV centroid
        mv_nodes = np.unique(self.lv_bdata[self.lv_bdata[:,-1]==self.patches['mv'],1:-1])
        mv_centroid = np.mean(xyz[mv_nodes], axis=0)
        dist = np.linalg.norm(xyz[lv_epi_nodes] - mv_centroid[None, :], axis=1)
        lv_epi_apex = lv_epi_nodes[np.argmax(dist)]

        dist = np.linalg.norm(xyz[lv_epi_nodes] - xyz[lv_epi_apex], axis=1)
        lv_epi_sep_apex = lv_epi_nodes[np.argmin(dist)]

        lv_endo_nodes = np.unique(self.lv_bdata[self.lv_bdata[:,-1]==self.patches['lv_endo'],1:-1])
        dist = np.linalg.norm(xyz[lv_endo_nodes] - xyz[lv_epi_apex], axis=1)
        lv_endo_sep_apex = lv_endo_nodes[np.argmin(dist)]

        sep_apex = np.array([lv_endo_sep_apex, lv_epi_sep_apex])

        self.lv_sep_endo_apex_node = lv_endo_sep_apex
        self.lv_sep_epi_apex_node = lv_epi_sep_apex


        # Finding septum nodes
        sep_nodes = np.unique(self.lv_bdata[self.lv_bdata[:,-1]==self.patches['rv_septum'],1:-1])
        if len(sep_nodes) == 0:
            raise (AttributeError, 'No septum defined')
        sep_nodes = np.setdiff1d(sep_nodes, self.lv_interface_nodes)

        # I also need the endo node at the septum
        dist = np.linalg.norm(xyz[sep_nodes] - xyz[lv_epi_apex], axis=1)
        self.lv_sep_sep_apex_node = sep_nodes[np.argmin(dist)]

        self.sep_endo_apex_node = self.map_lv_bv[self.lv_sep_sep_apex_node]
        self.sep_epi_apex_node = self.map_lv_bv[self.lv_sep_epi_apex_node]


        # Find apex for the rv
        bv_sep_apex_nodes = self.map_lv_bv[sep_apex]
        self.rv_sep_epi_apex_node = self.map_bv_rv[bv_sep_apex_nodes[1]]
        rv_endo_nodes = np.unique(self.rv_bdata[self.rv_bdata[:,-1]==self.patches['rv_endo'],1:-1])
        dist = np.linalg.norm(self.rv_mesh.points[rv_endo_nodes] - self.rv_mesh.points[self.rv_sep_epi_apex_node], axis=1)
        self.rv_sep_endo_apex_node = rv_endo_nodes[np.argmin(dist)]

        self.bv_sep_apex_nodes = np.append(bv_sep_apex_nodes, self.map_rv_bv[self.rv_sep_endo_apex_node])
        self.lv_sep_apex_nodes = np.array([self.lv_sep_endo_apex_node, self.lv_sep_epi_apex_node, self.lv_sep_sep_apex_node])

        # LV apex
        dist = self.long_axis_vector@(xyz[lv_endo_nodes]).T
        self.lv_lv_endo_apex_node = lv_endo_nodes[np.argmin(dist)]
        dist = np.linalg.norm(xyz[lv_epi_nodes] - xyz[self.lv_lv_endo_apex_node], axis=1)
        self.lv_lv_epi_apex_node = lv_epi_nodes[np.argmin(dist)]

        self.bv_lv_endo_apex_node = self.map_lv_bv[self.lv_lv_endo_apex_node]
        self.bv_lv_epi_apex_node = self.map_lv_bv[self.lv_lv_epi_apex_node]


    @staticmethod
    def map_results(child_mesh, parent_mesh, mapping, coords=None, data_type='points'):
        if coords is None:
            coords = ['circ', 'trans', 'long', 'long_plane']

        for coord in coords:
            if data_type == 'points':
                if (coord not in child_mesh.point_data): continue
                # Create (or not) the point data in the parent mesh
                if (coord not in parent_mesh.point_data):
                    if len(child_mesh.point_data[coord].shape) == 1:
                        var = np.zeros(parent_mesh.points.shape[0])
                        dim = 1
                    else:
                        dim = child_mesh.point_data[coord].shape[1]
                        var = np.zeros([parent_mesh.points.shape[0], dim])
                else:
                    var = parent_mesh.point_data[coord]

                var[mapping] = child_mesh.point_data[coord]

                parent_mesh.point_data[coord] = var
            elif data_type == 'elems':
                if (coord not in child_mesh.cell_data): continue
                # Create (or not) the point data in the parent mesh
                if (coord not in parent_mesh.cell_data):
                    if len(child_mesh.cell_data[coord][0].shape) == 1:
                        var = np.zeros(len(parent_mesh.cells[0].data))
                        dim = 1
                    else:
                        dim = child_mesh.point_data[coord].shape[1]
                        var = np.zeros([len(parent_mesh.cells[0].data), dim])
                else:
                    var = parent_mesh.cell_data[coord][0]

                var[mapping] = child_mesh.cell_data[coord][0]

                parent_mesh.cell_data[coord] = [var]


    def merge_lv_rv_point_data(self, coords):
        # It needs to be rv first and lv last do lv values are kept in interface
        self.map_results(self.rv_mesh, self.bv_mesh, self.map_rv_bv, coords)
        self.map_results(self.lv_mesh, self.bv_mesh, self.map_lv_bv, coords)


    def merge_lv_rv_cell_data(self, coords):
        # It needs to be rv first and lv last do lv values are kept in interface
        self.map_results(self.rv_mesh, self.bv_mesh, self.map_bv_rv_elems, coords, data_type='elems')
        self.map_results(self.lv_mesh, self.bv_mesh, self.map_bv_lv_elems, coords, data_type='elems')

    def merge_rv_lat_to_bv(self):
        """
        Merge rv_lat coordinate from RV mesh to biventricular mesh.
        Also merges rv_lat_tv and rv_lat_pv unblended angles.
        LV region is set to 0.
        """
        if 'rv_lat' not in self.rv_mesh.point_data:
            raise ValueError("rv_lat not found in rv_mesh.point_data")
        
        # Initialize rv_lat in bv_mesh with zeros
        rv_lat_bv = np.zeros(self.bv_mesh.points.shape[0])
        
        # Map RV values to bv_mesh
        rv_lat_bv[self.map_rv_bv] = self.rv_mesh.point_data['rv_lat']
        
        # LV region remains 0 (already initialized)
        # No need to explicitly set LV to 0 since it's already zero
        
        # Store in bv_mesh
        self.bv_mesh.point_data['rv_lat'] = rv_lat_bv
        
        # Also merge unblended angles if they exist
        if 'rv_lat_tv' in self.rv_mesh.point_data:
            rv_lat_tv_bv = np.zeros(self.bv_mesh.points.shape[0])
            rv_lat_tv_bv[self.map_rv_bv] = self.rv_mesh.point_data['rv_lat_tv']
            self.bv_mesh.point_data['rv_lat_tv'] = rv_lat_tv_bv
        
        if 'rv_lat_pv' in self.rv_mesh.point_data:
            rv_lat_pv_bv = np.zeros(self.bv_mesh.points.shape[0])
            rv_lat_pv_bv[self.map_rv_bv] = self.rv_mesh.point_data['rv_lat_pv']
            self.bv_mesh.point_data['rv_lat_pv'] = rv_lat_pv_bv



    def compute_cartesion_apex_distance(self):
        xyz = self.xyz
        epi_point = xyz[self.bv_sep_apex_nodes[1]]
        endo_point = xyz[self.bv_sep_apex_nodes[0]]
        axis = epi_point - endo_point
        axis = axis/np.linalg.norm(axis)
        vector = xyz - epi_point
        dist_axis = np.linalg.norm(vector, axis=1)
        axis_vec = axis[None,:]*dist_axis[:,None]
        par_vec = axis_vec+vector
        dist_par = np.linalg.norm(par_vec, axis=1)

        self.bv_mesh.point_data['dist_apex_par'] = dist_par
        self.bv_mesh.point_data['dist_apex_axis'] = dist_axis


    def compute_aha_segments(self, aha_type = 'points'):
        long = self.lv_mesh.point_data['long_plane']
        xyz = self.lv_mesh.points

        # Find lv endo apex
        lv_endo_nodes = np.unique(self.lv_bdata[self.lv_bdata[:,-1]==self.patches['lv_endo'],1:-1])
        dist = self.long_axis_vector@(xyz[lv_endo_nodes] - self.valve_centroids['mv']).T
        self.lv_endo_apex_node = lv_endo_nodes[np.argmax(np.abs(dist))]

        # Rescale long to be 0 at lv_endo_apex_node
        # Use base valve (AV if exists, else MV) to find node where long==1
        base_valve = self.get_base_valve()
        base_valve_node = np.where(long==1)[0][0]
        long = long - long[self.lv_endo_apex_node]
        long = long/long[base_valve_node]

        if aha_type == 'points':
            xyz_aha = self.lv_mesh.points
        elif aha_type == 'elems':
            ien = self.lv_mesh.cells[0].data
            xyz_aha = np.mean(xyz[ien], axis=1)
            long = np.mean(long[ien], axis=1)

        # Project coordinates to local cardinal system
        Q = np.array([self.septum_vector, self.third_vector, self.long_axis_vector]).T
        proj_xyz = (xyz_aha-xyz[self.lv_endo_apex_node])@Q

        # Long axis division
        base_marker = (long>=2/3)*(long<=1)
        mid_marker = (long>=1/3)*(long<2/3)
        apex_marker = (long>=0)*(long<1/3)
        apex_apex_marker = (long<0)

        # Compute circ for easiness
        base_centroid = np.mean(proj_xyz[base_marker], axis=0)
        mid_centroid = np.mean(proj_xyz[mid_marker], axis=0)
        apex_centroid = np.mean(proj_xyz[apex_marker], axis=0)

        base_xyz = proj_xyz - base_centroid
        mid_xyz = proj_xyz - mid_centroid
        apex_xyz = proj_xyz - apex_centroid

        base_circ = np.rad2deg(np.arctan2(base_xyz[:,1], base_xyz[:,0]))
        mid_circ = np.rad2deg(np.arctan2(mid_xyz[:,1], mid_xyz[:,0]))
        apex_circ = np.rad2deg(np.arctan2(apex_xyz[:,1], apex_xyz[:,0]))

        # Define AHA
        aha_region = np.zeros(len(xyz_aha))
        aha_region[(base_marker)*(base_circ>60)*(base_circ<=120)] = 4
        aha_region[(base_marker)*(base_circ>120)*(base_circ<=180)] = 5
        aha_region[(base_marker)*(base_circ>=-180)*(base_circ<=-120)] = 6
        aha_region[(base_marker)*(base_circ>-120)*(base_circ<=-60)] = 1
        aha_region[(base_marker)*(base_circ>-60)*(base_circ<=0)] = 2
        aha_region[(base_marker)*(base_circ>=0)*(base_circ<=60)] = 3

        aha_region[(mid_marker)*(mid_circ>60)*(mid_circ<=120)] = 10
        aha_region[(mid_marker)*(mid_circ>120)*(mid_circ<=180)] = 11
        aha_region[(mid_marker)*(mid_circ>=-180)*(mid_circ<=-120)] = 12
        aha_region[(mid_marker)*(mid_circ>-120)*(mid_circ<=-60)] = 7
        aha_region[(mid_marker)*(mid_circ>-60)*(mid_circ<=0)] = 8
        aha_region[(mid_marker)*(mid_circ>=0)*(mid_circ<=60)] = 9

        aha_region[(apex_marker)*(apex_circ>45)*(apex_circ<=135)] = 15
        aha_region[(apex_marker)*((apex_circ>135)+(apex_circ<=-135))] = 16
        aha_region[(apex_marker)*(apex_circ>-135)*(apex_circ<=-45)] = 13
        aha_region[(apex_marker)*(apex_circ>-45)*(apex_circ<=45)] = 14

        aha_region[apex_apex_marker] = 17


        if aha_type == 'points':
            self.lv_mesh.point_data['aha'] = aha_region
        elif aha_type == 'elems':
            self.lv_mesh.cell_data['aha'] = [aha_region]



class fastUVC(generalUVC):

    def pass_landmark_nodes(self, landmarks):
        self.bv_sep_epi_apex = landmarks['sep_epi']
        self.bv_sep_apex_nodes = np.array([landmarks['lv_sep_endo'],
                                           landmarks['sep_epi'],
                                           landmarks['sep_endo']])

    def get_zero_nodes(self):
        ien = self.lv_mesh.cells[0].data
        subdomains = self.lv_mesh.cell_data['subdomains'][0]
        sep_ant_nodes = np.unique(ien[subdomains==1])
        lat_ant_nodes = np.unique(ien[subdomains==2])
        lat_post_nodes = np.unique(ien[subdomains==3])
        sep_post_nodes = np.unique(ien[subdomains==4])
        zero_nodes = np.intersect1d(sep_ant_nodes, lat_ant_nodes)
        zero_nodes = np.intersect1d(zero_nodes, lat_post_nodes)
        self.lv_zero_nodes = np.intersect1d(zero_nodes, sep_post_nodes)


    def compute_long_plane_coord(self, septum):
        xyz = self.xyz

        # To define cut plane, select lower node of the AV or MV
        mv_nodes = np.unique(self.bv_bdata[self.bv_bdata[:,-1] == self.patches['mv'], 1:-1])
        dist = self.long_axis_vector@(xyz - xyz[self.bv_sep_epi_apex]).T
        mv_low = np.min(dist[mv_nodes])
        
        if self.has_av:
            av_nodes = np.unique(self.bv_bdata[self.bv_bdata[:,-1] == self.patches['av'], 1:-1])
            av_low = np.min(dist[av_nodes])
            valve_min = [av_low, mv_low]
            which = np.argmin(valve_min)
            if which == 0:
                cut_node = av_nodes[np.argmin(dist[av_nodes])]
            else:
                cut_node = mv_nodes[np.argmin(dist[mv_nodes])]
        else:
            # If no AV, use MV
            cut_node = mv_nodes[np.argmin(dist[mv_nodes])]

        # Long plane coord
        la_length = np.dot(xyz[cut_node] - xyz[self.bv_sep_epi_apex], self.long_axis_vector)
        long = self.long_axis_vector@(xyz - xyz[self.bv_sep_epi_apex]).T
        long_apex = long[self.bv_sep_epi_apex]
        long_base = long[cut_node]
        la_length = long_base-long_apex
        long = long/la_length
        self.bv_base_cut_node = cut_node
        self.bv_mesh.point_data['long_plane'] = long

    def split_rv_lv(self, septum):
        lv_marker_elems = self.bv_mesh.cell_data['rvlv'][0] == 1
        rv_marker_elems = ~lv_marker_elems

        self.map_bv_lv_elems = np.where(lv_marker_elems)[0]
        self.map_bv_rv_elems = np.where(rv_marker_elems)[0]

        # Inverse element mapping
        self.map_lv_bv_elems = -np.ones(len(self.ien), dtype=int)
        self.map_lv_bv_elems[self.map_bv_lv_elems] = np.arange(len(self.map_bv_lv_elems))
        self.map_rv_bv_elems = -np.ones(len(self.ien), dtype=int)
        self.map_rv_bv_elems[self.map_bv_rv_elems] = np.arange(len(self.map_bv_rv_elems))

        # Create independent mesh for the LV and RV saving the corresponding mappings
        self.lv_mesh, self.map_bv_lv, self.map_lv_bv = create_submesh(self.bv_mesh, self.map_bv_lv_elems)
        self.lv_bdata = create_submesh_bdata(self.lv_mesh, self.bv_bdata, self.map_bv_lv, self.map_lv_bv_elems, 'parent')
        self.lv_bdata_og = np.copy(self.lv_bdata)

        self.rv_mesh, self.map_bv_rv, self.map_rv_bv = create_submesh(self.bv_mesh, self.map_bv_rv_elems)
        self.rv_bdata = create_submesh_bdata(self.rv_mesh, self.bv_bdata, self.map_bv_rv, self.map_rv_bv_elems, 'parent')
        self.rv_bdata_og = np.copy(self.rv_bdata)

        # Preserve GlobalNodeID and GlobalElementID from bv_mesh to submeshes
        if 'GlobalNodeID' in self.bv_mesh.point_data:
            # Map GlobalNodeID to LV submesh
            self.lv_mesh.point_data['GlobalNodeID'] = self.bv_mesh.point_data['GlobalNodeID'][self.map_lv_bv]
            # Map GlobalNodeID to RV submesh
            self.rv_mesh.point_data['GlobalNodeID'] = self.bv_mesh.point_data['GlobalNodeID'][self.map_rv_bv]
        
        if 'GlobalElementID' in self.bv_mesh.cell_data:
            # Map GlobalElementID to LV submesh
            self.lv_mesh.cell_data['GlobalElementID'] = [self.bv_mesh.cell_data['GlobalElementID'][0][self.map_bv_lv_elems]]
            # Map GlobalElementID to RV submesh
            self.rv_mesh.cell_data['GlobalElementID'] = [self.bv_mesh.cell_data['GlobalElementID'][0][self.map_bv_rv_elems]]

        # In the bv bfile, rv-lv interface faces are only assigned to one of the meshes..
        # Therefore, we need to transfer that info to the other mesh.
        missing_lv = np.any(self.rv_bdata[:,-1] == self.patches['rvlv_ant'])
        rvlv_ant_faces = self.bv_bdata[self.bv_bdata[:,-1]==self.patches['rvlv_ant'],1:-1]
        rvlv_post_faces = self.bv_bdata[self.bv_bdata[:,-1]==self.patches['rvlv_post'],1:-1]
        rvlv_faces = np.vstack([rvlv_ant_faces, rvlv_post_faces])

        if missing_lv:
            bfaces = self.map_bv_lv[rvlv_faces]
        else:
            bfaces = self.map_bv_rv[rvlv_faces]
        belem = np.where(np.sum(np.isin(self.rv_mesh.cells[0].data, bfaces), axis=1)==3)[0]
        marker = np.ones(len(rvlv_faces), dtype=int)*self.patches['rvlv_ant']
        marker[len(rvlv_ant_faces):] = self.patches['rvlv_post']
        bdata = np.vstack([belem, bfaces.T, marker]).T

        if missing_lv:
            self.lv_bdata = np.vstack([self.lv_bdata, bdata])
        else:
            self.rv_bdata = np.vstack([self.rv_bdata, bdata])

        # Find interface points
        lv_elems = self.ien[lv_marker_elems]
        rv_elems = self.ien[rv_marker_elems]

        self.interface_nodes = np.unique(lv_elems[np.isin(lv_elems, rv_elems)])
        self.lv_interface_nodes = self.map_bv_lv[self.interface_nodes]
        self.rvlv_xyz =  self.xyz[self.interface_nodes]

        # Pass data to submeshes
        self.lv_mesh.point_data['long_plane'] = self.bv_mesh.point_data['long_plane'][self.map_lv_bv]
        self.rv_mesh.point_data['long_plane'] = self.bv_mesh.point_data['long_plane'][self.map_rv_bv]
        if 'subdomains' in self.bv_mesh.cell_data:
            self.lv_mesh.cell_data['subdomains'] = [self.bv_mesh.cell_data['subdomains'][0][self.map_bv_lv_elems]]

        # Save in dictionary
        self.meshes['lv'] = self.lv_mesh
        self.meshes['rv'] = self.rv_mesh
        self.bdatas['lv'] = self.lv_bdata
        self.bdatas['rv'] = self.rv_bdata


    def correct_lv_circ_by_subdomain(self, lv_circ):
        from mesh_utils import cell_data_to_point_data
        subdomains = self.lv_mesh.cell_data['subdomains'][0]
        node_subdomain = cell_data_to_point_data(subdomains, self.lv_mesh.cells[0].data)
        lv_circ[node_subdomain >= 3] = -lv_circ[node_subdomain >= 3]

        return lv_circ



class UVC(generalUVC):

    def compute_long_plane_coord(self, septum):
        xyz = self.xyz

        # Need to define the sep apex point
        if self.split_epi:
            marker = (self.bv_bdata[:,-1] == self.patches['lv_epi']) \
                      + (self.bv_bdata[:,-1] == self.patches['rv_epi'])
        else:
            marker = self.bv_bdata[:,-1] == self.patches['epi']

        if self.split_epi:
            print("split_epi is True")
            epi_nodes = np.where(self.bv_mesh.point_data['septum'] == 0.5)[0]
        else:
            epi_nodes = np.unique(self.bv_bdata[marker, 1:-1])
        dist_long = self.long_axis_vector@xyz[epi_nodes].T
        largest_dist = np.min(dist_long)
        dist_long = -(largest_dist-dist_long)
        dist_long /= np.max(dist_long)

        # TODO fix either the LVLA or check the weights are ok
        aux = np.copy(septum)
        aux[septum<self.thresholds['septum']] = 1e3
        weight_norm = 100*(dist_long)**2 + (aux[epi_nodes]*2-1)**2
        self.bv_sep_epi_apex = epi_nodes[np.argmin(weight_norm)]

        # To define cut plane, select lower node of the AV or MV
        mv_nodes = np.unique(self.bv_bdata[self.bv_bdata[:,-1] == self.patches['mv'], 1:-1])
        dist = self.long_axis_vector@(xyz - xyz[self.bv_sep_epi_apex]).T
        mv_low = np.min(dist[mv_nodes])
        
        if self.has_av:
            av_nodes = np.unique(self.bv_bdata[self.bv_bdata[:,-1] == self.patches['av'], 1:-1])
            av_low = np.min(dist[av_nodes])
            valve_min = [av_low, mv_low]
            which = np.argmin(valve_min)
            if which == 0:
                cut_node = av_nodes[np.argmin(dist[av_nodes])]
            else:
                cut_node = mv_nodes[np.argmin(dist[mv_nodes])]
        else:
            # If no AV, use MV
            cut_node = mv_nodes[np.argmin(dist[mv_nodes])]
        self.bv_base_cut_node = cut_node

        # Long plane coord
        la_length = np.dot(xyz[cut_node] - xyz[self.bv_sep_epi_apex], self.long_axis_vector)
        long = self.long_axis_vector@(xyz - xyz[self.bv_sep_epi_apex]).T
        long_apex = long[self.bv_sep_epi_apex]
        long_base = long[cut_node]
        la_length = long_base-long_apex
        long = long/la_length
        self.bv_mesh.point_data['long_plane'] = long


    @staticmethod
    def clean_mesh_split(mesh, bndry):
        ien = mesh.cells[0].data
        cells = np.arange(len(ien))

        mesh_ = mesh

        var_sp_elems = np.zeros(3)

        while len(var_sp_elems)>0 :
            bdata = chio.get_mesh_boundary(mesh_)
            bnodes = np.unique(bdata[:,1:-1])
            marker = np.zeros(len(mesh.points))
            marker[bnodes] = 1
            marker[bndry] = 0
            mesh_ = io.Mesh(mesh.points, {'tetra': ien[cells]}, point_data={'marker':marker})

            elem_nodes = marker[ien[cells]]
            sume = np.sum(elem_nodes, axis=1)
            var_sp_elems = np.where(sume==4)[0]
            cells = np.delete(cells, var_sp_elems, axis=0)

        return cells


    def define_septum_bc(self, septum0):
        bien = self.bv_bdata[:,1:-1]

        if self.split_epi:
            marker = (self.bv_bdata[:,-1] == self.patches['lv_epi']
                      + self.bv_bdata[:,-1] == self.patches['rv_epi'])
        else:
            marker = self.bv_bdata[:,-1] == self.patches['epi']
        epi_elems = np.where(marker)[0]

        septum0_elems = np.mean(septum0[bien], axis=1)
        long_elems = np.mean(self.bv_mesh.point_data['long_plane'][bien], axis=1)
        lv_elems = np.where((septum0_elems <= 0.5)*(long_elems<0.95))[0]

        lv_epi_elems = np.intersect1d(lv_elems, epi_elems)

        self.bv_bdata[lv_epi_elems,-1] = 12


    # def cut_base(self, septum):
    #     self.bv_mesh.point_data['septum'] = septum
    #     long = self.bv_mesh.point_data['long_plane']
    #     la_marker = long <= self.thresholds['long']
    #     la_marker_elems = np.sum(la_marker[self.ien], axis=1) >= 4
    #     ot_marker_elems = ~la_marker_elems
    #     self.map_bvot_bv_elems = np.where(la_marker_elems)[0]
    #     self.map_bvot_ot_elems = np.where(ot_marker_elems)[0]

    #     # Generating domain marker
    #     arr = np.zeros(len(self.bv_mesh.cells[0].data), dtype=int)
    #     arr[self.map_bvot_ot_elems] = 3
    #     self.bv_mesh.cell_data['domain'] = [arr]

    #     # Generating submeshes
    #     self.bvot_mesh = self.bv_mesh
    #     self.bvot_bdata = self.bv_bdata

    #     self.bv_mesh, self.map_bvot_bv, self.map_bv_bvot = create_submesh(self.bvot_mesh, self.map_bvot_bv_elems)
    #     self.bv_bdata = create_submesh_bdata(self.bv_mesh, self.bvot_bdata, self.map_bvot_bv, 'boundary')

    #     self.ot_mesh, self.map_bvot_ot, self.map_ot_bvot = create_submesh(self.bvot_mesh, self.map_bvot_ot_elems)
    #     self.ot_bdata = create_submesh_bdata(self.ot_mesh, self.bvot_bdata, self.map_bvot_ot, 'boundary')

    #     # Traspassing info
    #     self.xyz = self.bv_mesh.points
    #     self.ien = self.bv_mesh.cells[0].data
    #     self.bv_mesh.point_data['long_plane'] = long[self.map_bv_bvot]
    #     self.bv_mesh.point_data['septum'] = self.bvot_mesh.point_data['septum'][self.map_bv_bvot]
    #     self.bv_mesh.cell_data['domain'] = [self.bvot_mesh.cell_data['domain'][0][self.map_bvot_bv_elems]]
    #     if 'rvlv' in self.bvot_mesh.cell_data:
    #         self.bv_mesh.cell_data['rvlv'] = [self.bvot_mesh.cell_data['rvlv'][0][self.map_bvot_bv_elems]]
    #     self.bv_bdata[self.bv_bdata[:,-1]==0,-1] = 12
    #     self.bv_sep_epi_apex = self.map_bvot_bv[self.bv_sep_epi_apex]

    #     self.ot_mesh.point_data['long_plane'] = long[self.map_ot_bvot]
    #     self.ot_mesh.point_data['septum'] = self.bvot_mesh.point_data['septum'][self.map_ot_bvot]
    #     self.ot_bdata[self.ot_bdata[:,-1]==0,-1] = 12

    #     self.truncated = True

    def cut_base(self):
        # Find long cut value
        xyz = self.lv_mesh.points
        long = self.lv_mesh.point_data['long']
        sep_nodes = np.unique(self.lv_bdata[self.lv_bdata[:,-1]==self.patches['rv_septum'], 1:-1])
        lv_interface_endo_nodes = np.intersect1d(self.lv_interface_nodes, sep_nodes)
        cand_nodes = lv_interface_endo_nodes[long[lv_interface_endo_nodes]>0.5]

        vectors = xyz[cand_nodes] - xyz[self.lv_sep_epi_apex_node]
        septum_axis_vector = np.mean(vectors, axis=0)
        self.septum_axis_vector = septum_axis_vector/np.linalg.norm(septum_axis_vector)

        axis_dist = self.septum_axis_vector@vectors.T
        axis_vectors = self.septum_axis_vector*axis_dist[:,None]
        perp_vectors = axis_vectors - vectors
        perp_dist = np.linalg.norm(perp_vectors, axis=1)
        base_node = cand_nodes[np.argmin(perp_dist)]
        long_cut = long[base_node]*0.95

        # Generate new mesh
        mmg_mesh = pymmg.mmg_isoline_meshing(self.bv_mesh, self.bv_mesh.point_data['long'],
                                             isovalue=long_cut)

        # Generate bdata for mmg mesh
        # TODO I think it is better to create an element map. i.e., find what parent
        # element contains the new elements
        # mmg_xyz = mmg_mesh.points
        # ien = mmg_mesh.cells[0].data
        # new_nodes = np.arange(len(self.xyz), len(mmg_xyz), dtype=int)   # New nodes are div nodes

        # bnodes = np.unique(self.bv_bdata[:,1:-1])

        # array = np.array([[0,1,2],[1,2,3],[0,1,3],[2,0,3]])
        # nelems = np.repeat(np.arange(ien.shape[0]),4)
        # faces = np.vstack(ien[:,array])

        # bface_marker = np.any(np.isin(faces, bnodes), axis=1)
        # faces = faces[np.any(np.isin(faces, bnodes), axis=1)]
        # trimesh = io.Mesh(mmg_xyz, {'triangle': faces})


        # Split meshes



    def split_rv_lv(self, septum):
        if self.truncated:
            septum = self.bv_mesh.point_data['septum']
        else:
            self.bv_mesh.point_data['septum'] = septum
            self.bv_mesh.cell_data['domain'] = [np.zeros(len(self.bv_mesh.cells[0].data))]

        if self.split_epi:
            lv_marker_elems = self.bv_mesh.cell_data['rvlv'][0] == 0
            rv_marker_elems = self.bv_mesh.cell_data['rvlv'][0] == 1

        else:
            lv_marker = septum <= septum[self.bv_sep_epi_apex] #self.thresholds['septum']
            lv_marker_elems = np.sum(lv_marker[self.ien], axis=1) >= 3
            rv_marker_elems = ~lv_marker_elems

            # Make an initial lv_mesh to check rv-lv faces

            lv_elems = np.where(lv_marker_elems)[0]

            # Delete elements that have three faces in RV
            bnodes = self.bv_bdata[:,1:-1].flatten()
            node_bdry = np.zeros(len(self.xyz))
            node_bdry[bnodes] = np.repeat(self.bv_bdata[:,-1],3)

            marker_epi = node_bdry==self.patches['epi']
            not_sep_nodes = np.where(~((node_bdry==0)+(node_bdry==self.patches['rv_endo'])+(marker_epi)+(node_bdry==self.patches['rv_septum'])))[0]
            lv_mesh = io.Mesh(self.xyz, {'tetra': self.ien[lv_marker_elems]}, point_data={'bndry': node_bdry})

            cells = self.clean_mesh_split(lv_mesh, not_sep_nodes)
            lv_marker_elems[:] = False
            lv_marker_elems[lv_elems[cells]] = True
            rv_marker_elems = ~lv_marker_elems

            if 'domain' not in self.bv_mesh.cell_data:
                arr = np.zeros(len(self.bv_mesh.cells[0].data), dtype=int)
                self.bv_mesh.cell_data['domain'] = [arr]

        self.bv_mesh.cell_data['domain'][0][lv_marker_elems*(self.bv_mesh.cell_data['domain'][0]!=3)] = 1
        self.bv_mesh.cell_data['domain'][0][lv_marker_elems*(self.bv_mesh.cell_data['domain'][0]==3)] = 3
        self.bv_mesh.cell_data['domain'][0][rv_marker_elems*(self.bv_mesh.cell_data['domain'][0]!=3)] = 2
        self.bv_mesh.cell_data['domain'][0][rv_marker_elems*(self.bv_mesh.cell_data['domain'][0]==3)] = 4

        self.map_bv_lv_elems = np.where(lv_marker_elems)[0]
        self.map_bv_rv_elems = np.where(rv_marker_elems)[0]

        # Inverse element mapping
        self.map_lv_bv_elems = -np.ones(len(self.ien), dtype=int)
        self.map_lv_bv_elems[self.map_bv_lv_elems] = np.arange(len(self.map_bv_lv_elems))
        self.map_rv_bv_elems = -np.ones(len(self.ien), dtype=int)
        self.map_rv_bv_elems[self.map_bv_rv_elems] = np.arange(len(self.map_bv_rv_elems))

        # Create independent mesh for the LV and RV saving the corresponding mappings
        self.lv_mesh, self.map_bv_lv, self.map_lv_bv = create_submesh(self.bv_mesh, self.map_bv_lv_elems)
        self.lv_bdata = create_submesh_bdata(self.lv_mesh, self.bv_bdata, self.map_bv_lv, self.map_lv_bv_elems, 'boundary')

        self.rv_mesh, self.map_bv_rv, self.map_rv_bv = create_submesh(self.bv_mesh, self.map_bv_rv_elems)
        self.rv_bdata = create_submesh_bdata(self.rv_mesh, self.bv_bdata, self.map_bv_rv, self.map_rv_bv_elems, 'boundary')
        
        # Preserve GlobalNodeID and GlobalElementID from bv_mesh to submeshes
        if 'GlobalNodeID' in self.bv_mesh.point_data:
            # Map GlobalNodeID to LV submesh
            self.lv_mesh.point_data['GlobalNodeID'] = self.bv_mesh.point_data['GlobalNodeID'][self.map_lv_bv]
            # Map GlobalNodeID to RV submesh
            self.rv_mesh.point_data['GlobalNodeID'] = self.bv_mesh.point_data['GlobalNodeID'][self.map_rv_bv]
        
        if 'GlobalElementID' in self.bv_mesh.cell_data:
            # Map GlobalElementID to LV submesh
            self.lv_mesh.cell_data['GlobalElementID'] = [self.bv_mesh.cell_data['GlobalElementID'][0][self.map_bv_lv_elems]]
            # Map GlobalElementID to RV submesh
            self.rv_mesh.cell_data['GlobalElementID'] = [self.bv_mesh.cell_data['GlobalElementID'][0][self.map_bv_rv_elems]]
        
        # Debug: Check boundary data patch IDs
        if self.split_epi:
            bv_patches = np.unique(self.bv_bdata[:,-1])
            lv_patches = np.unique(self.lv_bdata[:,-1])
            rv_patches = np.unique(self.rv_bdata[:,-1])
            print(f"  Debug: BV boundary patches: {bv_patches}")
            print(f"  Debug: LV boundary patches: {lv_patches} (total {len(self.lv_bdata)} faces)")
            print(f"  Debug: RV boundary patches: {rv_patches} (total {len(self.rv_bdata)} faces)")
            if 'lv_epi' in self.patches:
                expected_lv_epi = self.patches['lv_epi']
                lv_epi_count = np.sum(self.lv_bdata[:,-1] == expected_lv_epi)
                print(f"  Debug: LV epi patch ID {expected_lv_epi} found in {lv_epi_count} LV boundary faces")
                bv_lv_epi_count = np.sum(self.bv_bdata[:,-1] == expected_lv_epi)
                print(f"  Debug: LV epi patch ID {expected_lv_epi} found in {bv_lv_epi_count} BV boundary faces")
                if expected_lv_epi not in lv_patches:
                    # Check how many LV mesh nodes are in the BV lv_epi faces
                    bv_lv_epi_faces = self.bv_bdata[self.bv_bdata[:,-1] == expected_lv_epi, 1:-1]
                    bv_lv_epi_nodes = np.unique(bv_lv_epi_faces.flatten())
                    lv_mesh_nodes = set(range(len(self.lv_mesh.points)))
                    bv_lv_epi_nodes_in_lv = [n for n in bv_lv_epi_nodes if self.map_bv_lv[n] >= 0]
                    print(f"  Debug: BV lv_epi has {len(bv_lv_epi_nodes)} unique nodes, {len(bv_lv_epi_nodes_in_lv)} map to LV mesh")
                    print(f"  Debug: LV mesh has {len(self.lv_mesh.points)} points")

        lv_0elems = np.where(self.lv_bdata[:,-1]==0)[0]
        lv_bndry0_elems = self.lv_bdata[lv_0elems,1:-1]
        rv_0elems = np.where(self.rv_bdata[:,-1]==0)[0]
        rv_bndry0_elems = self.rv_bdata[rv_0elems,1:-1]

        bndry0_elems = np.vstack([self.map_lv_bv[lv_bndry0_elems], self.map_rv_bv[rv_bndry0_elems]])
        un, ind, counts = np.unique(np.sort(bndry0_elems, axis=1), axis=0, return_counts=True, return_index=True)
        self.lv_bdata[lv_0elems[ind[counts==2]],-1] = self.patches['rv_lv_junction']
        self.lv_bdata_og = np.copy(self.lv_bdata)

        bndry0_elems = np.vstack([self.map_rv_bv[rv_bndry0_elems], self.map_lv_bv[lv_bndry0_elems]])
        un, ind, counts = np.unique(np.sort(bndry0_elems, axis=1), axis=0, return_counts=True, return_index=True)
        self.rv_bdata[rv_0elems[ind[counts==2]],-1] = self.patches['rv_lv_junction']
        self.rv_bdata_og = np.copy(self.rv_bdata)

        # Find interface points
        lv_elems = self.ien[lv_marker_elems]
        rv_elems = self.ien[rv_marker_elems]

        self.interface_nodes = np.unique(lv_elems[np.isin(lv_elems, rv_elems)])
        self.lv_interface_nodes = self.map_bv_lv[self.interface_nodes]
        self.rvlv_xyz =  self.xyz[self.interface_nodes]

        # Create LV RV element mask
        self.bv_mesh.cell_data['rvlv'] = [lv_marker_elems.astype(int)]

        # Pass long plane to submeshes
        self.lv_mesh.point_data['long_plane'] = self.bv_mesh.point_data['long_plane'][self.map_lv_bv]
        self.rv_mesh.point_data['long_plane'] = self.bv_mesh.point_data['long_plane'][self.map_rv_bv]


    def define_septum_nodes(self):
        xyz = self.lv_mesh.points
        long = self.lv_mesh.point_data['long']

        # Finding septum nodes
        sep_nodes = np.unique(self.lv_bdata[self.lv_bdata[:,-1]==self.patches['rv_septum'],1:-1])
        sep_nodes = np.setdiff1d(sep_nodes, self.lv_interface_nodes)
        vectors = xyz[sep_nodes] - xyz[self.lv_sep_epi_apex_node]
        vectors = vectors/np.linalg.norm(vectors, axis=1)[:,None]
        vector = np.mean(vectors, axis=0)
        vector = vector/np.linalg.norm(vector)
        third_vector = np.cross(vector, self.septum_vector)
        third_vector = third_vector/np.linalg.norm(third_vector)


        # Find cut nodes
        sep_nodes = np.unique(self.lv_bdata[self.lv_bdata[:,-1]==self.patches['rv_septum'],1:-1])
        sep_nodes = np.intersect1d(sep_nodes, self.lv_interface_nodes)
        long_sep_nodes = long[sep_nodes]
        cut_long = np.quantile(long_sep_nodes,0.65)


        # Project interface coordinates to septum plane
        vectors = xyz[sep_nodes] - np.mean(xyz[sep_nodes], axis=0)
        proj_points = np.array([third_vector@vectors.T, vector@vectors.T]).T
        first_quad = np.where((proj_points[:,0]>0)*(proj_points[:,1]>0))[0]
        second_quad = np.where((proj_points[:,0]<0)*(proj_points[:,1]>0))[0]


        # Find points that maximize distance
        ant_point = first_quad[np.argmin((long_sep_nodes[first_quad]-cut_long)**2)]
        post_point = second_quad[np.argmin((long_sep_nodes[second_quad]-cut_long)**2)]

        self.sep_endo_base_nodes =  np.array([sep_nodes[ant_point], sep_nodes[post_point]])    # First is anterior


        # Finding septum endo nodes to define circ coordinates.
        vector = xyz[self.sep_endo_base_nodes[0]] - xyz[self.lv_sep_endo_apex_node]
        self.rvlv_ant_vector = vector/np.linalg.norm(vector)

        vector = xyz[self.sep_endo_base_nodes[1]] - xyz[self.lv_sep_endo_apex_node]
        self.rvlv_post_vector = vector/np.linalg.norm(vector)


    def define_septum_nodes2(self):
        xyz = self.lv_mesh.points
        long = self.lv_mesh.point_data['long']

        # Finding septum nodes
        sep_nodes = np.unique(self.lv_bdata[self.lv_bdata[:,-1]==self.patches['rv_septum'],1:-1])
        sep_nodes = np.intersect1d(sep_nodes, self.lv_interface_nodes)
        order = np.argsort(long[sep_nodes])
        sep_nodes = sep_nodes[order]

        vector = self.xyz[self.bv_base_cut_node] - xyz[self.lv_sep_epi_apex_node]
        vector = vector/np.linalg.norm(vector)
        third_vector = np.cross(vector, self.septum_vector)
        third_vector = third_vector/np.linalg.norm(third_vector)

        dist = third_vector@(xyz[sep_nodes] -  xyz[self.lv_sep_epi_apex_node]).T
        dist = dist/np.max(np.abs(dist))
        long = long[sep_nodes]
        long = (long - np.min(long))/(np.max(long) - np.min(long))

        up_nodes = np.where(long > 0.5)[0]
        cut_nodes = up_nodes[np.argmin(np.abs(dist[up_nodes]))]

        # Iterate until it finds points
        mult = 1.0
        up_ant_nodes = []
        up_post_nodes = []
        while (len(up_ant_nodes)==0) or (len(up_post_nodes)==0):
            up_ant_nodes = np.where((long > long[cut_nodes]*mult)*(dist>0))[0]
            up_post_nodes = np.where((long > long[cut_nodes]*mult)*(dist<0))[0]
            mult -= 0.05

        ant_node = sep_nodes[up_ant_nodes[np.argmax(np.abs(dist[up_ant_nodes]))]]
        post_node = sep_nodes[up_post_nodes[np.argmax(np.abs(dist[up_post_nodes]))]]

        self.sep_endo_base_nodes =  np.array([ant_node, post_node])    # First is anterior


        # Finding septum endo nodes to define circ coordinates.
        vector = xyz[self.sep_endo_base_nodes[0]] - xyz[self.lv_sep_endo_apex_node]
        self.rvlv_ant_vector = vector/np.linalg.norm(vector)

        vector = xyz[self.sep_endo_base_nodes[1]] - xyz[self.lv_sep_endo_apex_node]
        self.rvlv_post_vector = vector/np.linalg.norm(vector)

    def create_lv_circ_bc1(self):
        xyz = self.lv_mesh.points

        # Choose the LV interface node with the smallest longitudinal value
        if 'long' not in self.lv_mesh.point_data:
            raise ValueError("LV mesh is missing 'long' field required for circumferential BCs.")
        long_lv = self.lv_mesh.point_data['long']
        interface_nodes = self.lv_interface_nodes
        if interface_nodes is None or len(interface_nodes) == 0:
            raise ValueError("lv_interface_nodes is empty; cannot define circumferential BCs.")
        farthest_node_from_mv = interface_nodes[np.argmin(long_lv[interface_nodes])]

        nodes = self.lv_interface_nodes
        apex_distance1 = np.linalg.norm(xyz[nodes] - xyz[self.lv_sep_sep_apex_node], axis=1)
        apex_distance2 = np.linalg.norm(xyz[nodes] - xyz[farthest_node_from_mv], axis=1)
        nodes = nodes[(apex_distance1>self.mesh_size*3)*(apex_distance2>self.mesh_size*3)]

        interface_xyz = xyz[nodes]
        septum_center = np.mean(interface_xyz, axis=0)

        # Vector that splits septum in two
        vector = septum_center - xyz[farthest_node_from_mv]
        vlength = np.linalg.norm(vector)
        vector = vector/vlength
        third_vector = np.cross(vector, self.septum_vector)
        third_vector = third_vector/np.linalg.norm(third_vector)
        septum_center = septum_center - vector*vlength*0.5
        self.septum_mid_point = septum_center

        # Finding septum endo nodes to define circ coordinates.
        vector = xyz[self.sep_endo_base_nodes[0]] - septum_center
        self.rvlv_ant_vector = vector/np.linalg.norm(vector)

        vector = xyz[self.sep_endo_base_nodes[1]] - septum_center
        self.rvlv_post_vector = vector/np.linalg.norm(vector)

        dist_third = third_vector@(interface_xyz - xyz[self.lv_sep_sep_apex_node]).T
        ant_nodes = nodes[dist_third > 0]
        post_nodes = nodes[dist_third < 0]

        self.rvlv_normal_ant = np.cross(self.rvlv_ant_vector, self.septum_vector)       # point outwards
        self.lv_bc1_marker_ant = ant_nodes[self.rvlv_normal_ant@(xyz[ant_nodes] - septum_center).T >= 0]

        self.rvlv_normal_post = -np.cross(self.rvlv_post_vector, self.septum_vector)
        self.lv_bc1_marker_post = post_nodes[self.rvlv_normal_post@(xyz[post_nodes] - septum_center).T >= 0]

        marker = np.append(self.lv_bc1_marker_ant, self.lv_bc1_marker_post)
        vals = np.append(np.ones(len(self.lv_bc1_marker_ant)), -np.ones(len(self.lv_bc1_marker_post)))

        arr = np.zeros(self.lv_mesh.points.shape[0])
        arr[marker] = vals.flatten()
        self.lv_mesh.point_data['bc1'] = arr

    def create_lv_circ_bc1_old(self):
        xyz = self.lv_mesh.points

        nodes = self.lv_interface_nodes
        apex_distance1 = np.linalg.norm(xyz[nodes] - xyz[self.lv_sep_sep_apex_node], axis=1)
        apex_distance2 = np.linalg.norm(xyz[nodes] - xyz[self.lv_sep_epi_apex_node], axis=1)
        nodes = nodes[(apex_distance1>self.mesh_size*3)*(apex_distance2>self.mesh_size*3)]

        interface_xyz = xyz[nodes]
        septum_center = np.mean(interface_xyz, axis=0)

        # Vector that splits septum in two
        vector = septum_center - xyz[self.lv_sep_sep_apex_node]
        vlength = np.linalg.norm(vector)
        vector = vector/vlength
        third_vector = np.cross(vector, self.septum_vector)
        third_vector = third_vector/np.linalg.norm(third_vector)
        septum_center = septum_center - vector*vlength*0.5
        self.septum_mid_point = septum_center

        # Finding septum endo nodes to define circ coordinates.
        vector = xyz[self.sep_endo_base_nodes[0]] - septum_center
        self.rvlv_ant_vector = vector/np.linalg.norm(vector)

        vector = xyz[self.sep_endo_base_nodes[1]] - septum_center
        self.rvlv_post_vector = vector/np.linalg.norm(vector)

        dist_third = third_vector@(interface_xyz - xyz[self.lv_sep_sep_apex_node]).T
        ant_nodes = nodes[dist_third > 0]
        post_nodes = nodes[dist_third < 0]

        self.rvlv_normal_ant = np.cross(self.rvlv_ant_vector, self.septum_vector)       # point outwards
        self.lv_bc1_marker_ant = ant_nodes[self.rvlv_normal_ant@(xyz[ant_nodes] - septum_center).T >= 0]

        self.rvlv_normal_post = -np.cross(self.rvlv_post_vector, self.septum_vector)
        self.lv_bc1_marker_post = post_nodes[self.rvlv_normal_post@(xyz[post_nodes] - septum_center).T >= 0]

        marker = np.append(self.lv_bc1_marker_ant, self.lv_bc1_marker_post)
        vals = np.append(np.ones(len(self.lv_bc1_marker_ant)), -np.ones(len(self.lv_bc1_marker_post)))

        arr = np.zeros(self.lv_mesh.points.shape[0])
        arr[marker] = vals.flatten()
        self.lv_mesh.point_data['bc1'] = arr


    def create_lv_circ_bc2(self, lv_circ1):
        long = self.lv_mesh.point_data['long_plane']
        ien = self.lv_mesh.cells[0].data
        xyz = self.lv_mesh.points

        # Find elements at isoline 0
        pos_nodes, neg_nodes, div_nodes, div_elems = find_isoline(lv_circ1, 0, ien)
        self.circ_div_elems = div_elems

        # Define sep/nop sep nodes
        # I need to find the most lateral point of the mv first
        mv_nodes = np.unique(self.lv_bdata[self.lv_bdata[:,-1]==self.patches['mv'],1:-1])
        dist = self.septum_vector@(xyz[mv_nodes]-self.valve_centroids['mv']).T
        lat_mv = mv_nodes[np.argmin(dist)]

        vector = xyz[lat_mv] - xyz[self.lv_sep_endo_apex_node]
        aux_vector = np.cross(vector, self.long_axis_vector)
        sep_vector = np.cross(vector, aux_vector)
        sep_vector = sep_vector/np.linalg.norm(sep_vector)  # sep_vector should point outside the LV
        if np.dot(sep_vector, self.septum_vector) < 0: sep_vector = -sep_vector

        dist = sep_vector@(xyz[div_nodes] - xyz[self.lv_sep_endo_apex_node]).T
        sep_nodes = div_nodes[dist >= 0]

        apex_nodes = np.where(long < long[self.lv_sep_sep_apex_node])[0]
        sep_nodes = np.setdiff1d(sep_nodes, apex_nodes)

        lat_nodes = np.setdiff1d(div_nodes, sep_nodes)

        # Find bridge nodes
        midpoints = np.mean(xyz[ien], axis=1)
        if self.has_av:
            vec_valves = np.linalg.norm(self.valve_centroids['av'] - self.valve_centroids['mv'])
            vec_mv = np.linalg.norm(xyz[div_nodes] - self.valve_centroids['mv'], axis=1)
            vec_av = np.linalg.norm(xyz[div_nodes] - self.valve_centroids['av'], axis=1)
            bridge_nodes = (vec_mv < vec_valves)*(vec_av < vec_valves)
            bridge_nodes = div_nodes[bridge_nodes]
            vec_mv = np.linalg.norm(midpoints[div_elems] - self.valve_centroids['mv'], axis=1)
            vec_av = np.linalg.norm(midpoints[div_elems] - self.valve_centroids['av'], axis=1)
            bridge_elems = (vec_mv < vec_valves)*(vec_av < vec_valves)
            self.bridge_elems_bc = div_elems[bridge_elems]
        else:
            # If no AV, skip bridge node detection (no bridge between AV and MV)
            bridge_nodes = np.array([], dtype=int)
            self.bridge_elems_bc = np.array([], dtype=int)

        # Correcting lat and sep
        lat_nodes = np.setdiff1d(lat_nodes, bridge_nodes)
        sep_nodes = np.setdiff1d(sep_nodes, bridge_nodes)

        valve_nodes = np.where(long > 0.9)[0]
        bridge_nodes = np.intersect1d(bridge_nodes, valve_nodes)
        self.lv_bc_bridge_p = np.intersect1d(bridge_nodes, pos_nodes)
        self.lv_bc_bridge_n = np.intersect1d(bridge_nodes, neg_nodes)
        self.lv_bc_bridge_p_values = lv_circ1[self.lv_bc_bridge_p]
        self.lv_bc_bridge_n_values = lv_circ1[self.lv_bc_bridge_n]

        # Find septum nodes
        self.lv_bc_0p = np.intersect1d(sep_nodes, pos_nodes)
        self.lv_bc_0n = np.intersect1d(sep_nodes, neg_nodes)

        # Set lateral bc
        lat_nodes = np.setdiff1d(lat_nodes, bridge_nodes)

        # Find positive/negative nodes at the lateral wall
        pos_nodes = np.where(lv_circ1 >= 0)[0]
        neg_nodes = np.where(lv_circ1 < 0)[0]

        self.lv_bc_pi = np.intersect1d(lat_nodes, pos_nodes)
        self.lv_bc_values_pi = np.pi - lv_circ1[self.lv_bc_pi]*np.pi/2
        self.lv_bc_mpi = np.intersect1d(lat_nodes, neg_nodes)
        self.lv_bc_values_mpi = -np.pi - lv_circ1[self.lv_bc_mpi]*np.pi/2

        nodes = np.concatenate([self.lv_bc_0p, self.lv_bc_pi, self.lv_bc_mpi])
        values = np.concatenate([np.zeros(len(self.lv_bc_0p)),
                                 self.lv_bc_values_pi ,
                                 self.lv_bc_values_mpi])
        arr = np.zeros(self.lv_mesh.points.shape[0])
        arr[nodes] = values.flatten()
        self.lv_mesh.point_data['bc2'] = arr

        self.lv_mesh.point_data['lv_circ1'] = lv_circ1



    def correct_lv_circ2(self, lv_circ2):
        lv_circ1 = self.lv_mesh.point_data['lv_circ1']
        self.lv_ant_post_mask = (lv_circ1 > 0).astype(int)
        lv_circ2[self.lv_ant_post_mask==0] *= -1
        return lv_circ2


    def create_lv_circ_bc3(self, lv_circ2):
        bfaces = self.lv_bdata[:,1:-1]
        marker = self.lv_bdata[:,-1]
        long = self.lv_mesh.point_data['long_plane']
        aux_circ2 = self.correct_lv_circ2(np.copy(lv_circ2))

        # Boundaries of interest
        sep_nodes = np.unique(self.lv_bdata[self.lv_bdata[:,-1]==self.patches['rv_septum'],1:-1])
        ant_marker = np.intersect1d(sep_nodes, self.lv_bc1_marker_ant)
        post_marker = np.intersect1d(sep_nodes, self.lv_bc1_marker_post)

        lv_endo_nodes = np.unique(bfaces[marker==self.patches['lv_endo']])
        lv_endo_nodes = lv_endo_nodes[long[lv_endo_nodes]<=1.0]

        # Nodes defining septum boundary
        endo_circ2 = aux_circ2[lv_endo_nodes]
        endo_bc_nodes = np.where((endo_circ2 < np.median(aux_circ2[ant_marker])*0.75)*
                                  (endo_circ2 > np.median(aux_circ2[post_marker])*0.75))[0]
        endo_bc_nodes = lv_endo_nodes[endo_bc_nodes]
        arr = np.zeros(self.lv_mesh.points.shape[0])
        arr[endo_bc_nodes] = np.ones(len(endo_bc_nodes))
        self.lv_mesh.point_data['bc3'] = arr


    def create_lv_circ_bc4(self, lv_circ3):
        lv_circ2 = self.lv_mesh.point_data['lv_circ2']
        ien = self.lv_mesh.cells[0].data
        xyz = self.lv_mesh.points

        # Find elements at isoline 0
        pos_nodes, neg_nodes, div_nodes, div_elems = find_isoline(lv_circ3, 0, ien)
        ant_nodes = pos_nodes[lv_circ2[pos_nodes]>0]
        post_nodes = pos_nodes[lv_circ2[pos_nodes]<0]

        vec1 = self.rvlv_normal_ant@(xyz[post_nodes] - self.septum_mid_point).T
        post_nodes = post_nodes[ vec1 >= 0]
        vec2 = self.rvlv_normal_post@(xyz[ant_nodes] - self.septum_mid_point).T
        ant_nodes = ant_nodes[ vec2 <= 0]

        arr = np.zeros(self.lv_mesh.points.shape[0])
        arr[ant_nodes] = np.ones(len(ant_nodes))
        arr[post_nodes] = -np.ones(len(post_nodes))
        self.lv_mesh.point_data['bc4_p'] = arr

        ant_nodes = neg_nodes[lv_circ2[neg_nodes]>0]
        post_nodes = neg_nodes[lv_circ2[neg_nodes]<0]

        vec1 = self.rvlv_normal_ant@(xyz[post_nodes] - self.septum_mid_point).T
        post_nodes = post_nodes[ vec1 >= 0]
        vec2 = self.rvlv_normal_post@(xyz[ant_nodes] - self.septum_mid_point).T
        ant_nodes = ant_nodes[ vec2 >= 0]

        arr = np.zeros(self.lv_mesh.points.shape[0])
        arr[ant_nodes] = np.ones(len(ant_nodes))
        arr[post_nodes] = -np.ones(len(post_nodes))
        self.lv_mesh.point_data['bc4_n'] = arr


    def correct_lv_circ4(self, lv_circ4):
        lv_circ1 = self.lv_mesh.point_data['lv_circ1']
        circ4 = lv_circ4*np.sign(lv_circ1)

        return circ4


    def create_lv_circ_bc5(self, lv_circ4):
        ien = self.lv_mesh.cells[0].data
        pos_nodes, neg_nodes, div_nodes, div_elems = find_isoline(lv_circ4, 0, ien)
        arr = np.zeros(self.lv_mesh.points.shape[0])
        arr[div_nodes] = np.ones(len(div_nodes))
        self.lv_mesh.point_data['bc5'] = arr




    def create_rv_circ_bc(self):
        xyz = self.rv_mesh.points
        self.rv_interface_nodes = self.map_bv_rv[self.interface_nodes]

        # Getting apex distance
        nodes = self.rv_interface_nodes
        apex_distance1 = np.linalg.norm(xyz[nodes] - self.xyz[self.sep_epi_apex_node], axis=1)
        apex_distance2 = np.linalg.norm(xyz[nodes] - self.xyz[self.sep_endo_apex_node], axis=1)
        nodes = nodes[(apex_distance1>self.mesh_size*2)*(apex_distance2>self.mesh_size*2)]

        interface_xyz = xyz[nodes]

        rv_bc1_marker_post = nodes[self.rvlv_normal_post@(interface_xyz - self.septum_mid_point).T > 0]
        rv_bc1_marker_ant = nodes[self.rvlv_normal_ant@(interface_xyz - self.septum_mid_point).T > 0]

        # Find nodes that are in both
        inter_nodes = np.intersect1d(rv_bc1_marker_ant, rv_bc1_marker_post)
        rv_bc1_marker_ant = np.setdiff1d(rv_bc1_marker_ant, inter_nodes)
        rv_bc1_marker_post = np.setdiff1d(rv_bc1_marker_post, inter_nodes)

        # Split these nodes
        aux_vector = self.septum_mid_point - self.xyz[self.sep_epi_apex_node]
        aux_vector = aux_vector/np.linalg.norm(aux_vector)
        aux_vector = np.cross(self.septum_vector, aux_vector)

        dist = aux_vector@(xyz[inter_nodes] - self.septum_mid_point).T
        pos_nodes = inter_nodes[dist > 0]
        neg_nodes = inter_nodes[dist < 0]
        self.rv_bc1_marker_ant = np.union1d(rv_bc1_marker_ant, neg_nodes)
        self.rv_bc1_marker_post = np.union1d(rv_bc1_marker_post, pos_nodes)

        marker = np.append(self.rv_bc1_marker_ant, self.rv_bc1_marker_post)
        vals = np.append(np.ones(len(self.rv_bc1_marker_ant)), -np.ones(len(self.rv_bc1_marker_post)))

        arr = np.zeros(self.rv_mesh.points.shape[0])
        arr[marker] = vals.flatten()
        self.rv_mesh.point_data['bc1'] = arr


    def create_ot_circ_bc1(self):
        # Map to original mesh
        bvot_circ = np.zeros(len(self.bvot_mesh.points))
        bvot_circ[self.map_bv_bvot] = self.bv_mesh.point_data['circ']

        # Map back to OT mesh
        self.ot_mesh.point_data['bc1'] = bvot_circ[self.map_ot_bvot]


    def create_ot_circ_bc2(self, ot_circ1):
        ien = self.ot_mesh.cells[0].data
        xyz = self.ot_mesh.points

        # Find elements at isoline 0
        pos_nodes, neg_nodes, div_nodes, div_elems = find_isoline(ot_circ1, 0, ien)

        # Define sep/nop sep nodes
        dist = self.septum_vector@(xyz[div_nodes] - self.valve_centroids['mv']).T
        lat_nodes = div_nodes[dist <= 0]

        self.ot_bc_pi = np.intersect1d(lat_nodes, pos_nodes)
        self.ot_bc_mpi = np.intersect1d(lat_nodes, neg_nodes)

        arr = np.zeros(self.ot_mesh.points.shape[0])
        arr[self.ot_bc_pi] = np.ones(len(self.ot_bc_pi))
        arr[self.ot_bc_mpi] = -np.ones(len(self.ot_bc_mpi))
        self.ot_mesh.point_data['bc2'] = arr



    def merge_bv_ot_results(self):
        self.map_results(self.ot_mesh, self.bvot_mesh, self.map_ot_bvot)
        self.map_results(self.bv_mesh, self.bvot_mesh, self.map_bv_bvot)




def mmg_create_lv_circ_bc2(lv_circ1, uvc):
    # Split mesh
    mmg_mesh = pymmg.mmg_isoline_meshing(uvc.lv_mesh, lv_circ1)
    xyz = uvc.lv_mesh.points
    mmg_xyz = mmg_mesh.points

    # Find elements at isoline 0
    div_nodes = np.arange(len(xyz), len(mmg_xyz), dtype=int)   # New nodes are div nodes

    # Define sep/nop sep nodes
    # I need to find the most lateral point of the mv-endo edge first
    mv_nodes = np.unique(uvc.lv_bdata[uvc.lv_bdata[:,-1]==uvc.patches['mv'],1:-1])
    endo_nodes = np.unique(uvc.lv_bdata[uvc.lv_bdata[:,-1]==uvc.patches['lv_endo'],1:-1])
    mv_nodes = np.intersect1d(mv_nodes, endo_nodes)
    dist = uvc.septum_vector@(mmg_xyz[mv_nodes]-uvc.valve_centroids['mv']).T
    lat_mv = mv_nodes[np.argmin(dist)]
    cent_vector = uvc.valve_centroids['mv'] - mmg_xyz[lat_mv]
    point = mmg_xyz[lat_mv] + cent_vector*0.1

    vector = point - xyz[uvc.lv_sep_epi_apex_node]
    aux_vector = np.cross(vector, uvc.long_axis_vector)
    sep_vector = np.cross(vector, aux_vector)
    sep_vector = sep_vector/np.linalg.norm(sep_vector)
    if np.dot(sep_vector, uvc.septum_vector) < 0: sep_vector = -sep_vector

    dist = sep_vector@(mmg_xyz[div_nodes] - xyz[uvc.lv_sep_epi_apex_node]).T
    sep_nodes = div_nodes[dist >= 0]


    lat_nodes = np.setdiff1d(div_nodes, sep_nodes)

    bmarker = np.zeros(len(mmg_xyz))
    bmarker[sep_nodes] = 1
    bmarker[lat_nodes] = 2
    mmg_mesh.point_data['bc2'] = bmarker
    return mmg_mesh, bmarker


def mmg_create_lv_circ_bc4(lv_circ3, uvc):
    tol = 1e-3
    xyz = uvc.lv_mesh.points
    lv_circ1 = uvc.lv_mesh.point_data['lv_circ1']

    # Split mesh sep/lat
    mmg_mesh1, mmg_bdata1 = pymmg.mmg_isoline_meshing(uvc.lv_mesh, lv_circ1, funcs_to_interpolate=['lv_circ2', 'lv_circ3', 'trans'], bdata=uvc.lv_bdata)
    mmg_mesh1.point_data['lv_circ1'] = mmg_mesh1.point_data['f']
    mmg_xyz1 = mmg_mesh1.points
    div_nodes1 = np.arange(len(xyz), len(mmg_xyz1), dtype=int)   # New nodes are div nodes
    add_nodes = np.where(np.abs(lv_circ1) < tol)
    div_nodes1 = np.union1d(div_nodes1, add_nodes)

    # Split mesh ant/post
    mmg_mesh2, mmg_bdata2 = pymmg.mmg_isoline_meshing(mmg_mesh1, mmg_mesh1.point_data['lv_circ3'],
                                          funcs_to_interpolate=['lv_circ1', 'lv_circ2', 'trans'], bdata = mmg_bdata1)

    mmg_xyz2 = mmg_mesh2.points

    io.write('check1.vtu', mmg_mesh1)
    io.write('check2.vtu', mmg_mesh2)
    div_nodes2 = np.arange(len(mmg_xyz1), len(mmg_xyz2), dtype=int)   # New nodes are div nodes
    add_nodes = np.where(np.abs(mmg_mesh1.point_data['lv_circ3']) < tol)
    div_nodes2 = np.union1d(div_nodes2, add_nodes)
    mmg_mesh2.point_data['lv_circ3'] = mmg_mesh2.point_data['f']
    # mmg_mesh2.cell_data['map'] = [mmg_mesh1.cell_data['map'][0][mmg_mesh2.cell_data['map'][0]]]

    mmg_circ1 = mmg_mesh2.point_data['lv_circ1']
    mmg_circ2 = mmg_mesh2.point_data['lv_circ2']
    mmg_circ3 = mmg_mesh2.point_data['lv_circ3']
    div_nodes = np.append(div_nodes1, div_nodes2)

    # Assign boundaries
    sep_lat_nodes = div_nodes[np.abs(mmg_circ1[div_nodes])<tol]
    ant_post_nodes = div_nodes[np.abs(mmg_circ3[div_nodes])<tol]
    zero_nodes = np.intersect1d(sep_lat_nodes, ant_post_nodes)
    apex_dist = np.linalg.norm(mmg_xyz2[zero_nodes] - xyz[uvc.lv_sep_sep_apex_node], axis=1)
    zero_nodes = zero_nodes[apex_dist<np.mean(apex_dist)]

    lat_nodes = sep_lat_nodes[(mmg_circ3[sep_lat_nodes]<0)*(mmg_circ2[sep_lat_nodes]>0.5)]
    sep_nodes = np.setdiff1d(sep_lat_nodes, lat_nodes)

    ant_nodes = ant_post_nodes[mmg_circ1[ant_post_nodes]>0]
    post_nodes = ant_post_nodes[mmg_circ1[ant_post_nodes]<=0]

    vec1 = uvc.rvlv_normal_ant@(mmg_xyz2[ant_nodes] - uvc.septum_mid_point).T
    ant_nodes = ant_nodes[ vec1 >= 0]
    vec2 = uvc.rvlv_normal_post@(mmg_xyz2[post_nodes] - uvc.septum_mid_point).T
    post_nodes = post_nodes[ vec2 >= 0]
    ant_post_nodes = np.union1d(ant_nodes, post_nodes)

    # Saving origin
    zero_trans = mmg_mesh2.point_data['trans'][zero_nodes]
    order = np.argsort(zero_trans)
    origin = np.vstack([zero_trans[order], mmg_xyz2[zero_nodes][order].T]).T
    uvc.origin = origin

    bmarker = np.zeros(len(mmg_xyz2))
    bmarker[sep_nodes] = 1
    bmarker[lat_nodes] = 2
    # bmarker[len(mmg_xyz2):] = 3
    bmarker[ant_nodes] = 4
    bmarker[post_nodes] = 5
    bmarker[zero_nodes] = 10
    mmg_mesh2.point_data['bc4'] = bmarker

    io.write('check.vtu', mmg_mesh2)

    return mmg_mesh2, bmarker, mmg_bdata2


def mmg_create_lv_circ_bc5(lv_circ4, uvc):
    xyz = uvc.lv_mesh.points
    aux = lv_circ4*30

    # Split mesh
    mmg_mesh = pymmg.mmg_isoline_meshing(uvc.lv_mesh, aux, isovalue=10)
    mmg_xyz = mmg_mesh.points
    div_nodes = np.arange(len(xyz), len(mmg_xyz), dtype=int)   # New nodes are div nodes
    f = mmg_mesh.point_data['f']

    # Need to create a cell map from the og mesh to the mmg mesh
    og_midpoints = np.mean(uvc.lv_mesh.points[uvc.lv_mesh.cells[0].data], axis=1)
    mmg_midpoints = np.mean(mmg_mesh.points[mmg_mesh.cells[0].data], axis=1)

    tree = KDTree(og_midpoints)
    _, map_elems = tree.query(mmg_midpoints)

    # Passing circ1
    mmg_circ1 = np.zeros(len(mmg_xyz))
    mmg_circ1[:len(xyz)] = uvc.lv_mesh.point_data['lv_circ1']

    tree = KDTree(xyz)
    _, closest_og_node = tree.query(mmg_xyz[div_nodes])
    mmg_circ1[div_nodes] = uvc.lv_mesh.point_data['lv_circ1'][closest_og_node]

    ant_nodes = div_nodes[mmg_circ1[div_nodes]>0]
    post_nodes = div_nodes[mmg_circ1[div_nodes]<0]

    # Create submeshes
    ien = mmg_mesh.cells[0].data
    sep_elem_marker = np.all(f[ien] <= 10, axis=1)
    lat_elem_marker = ~sep_elem_marker
    sep_map_elem = map_elems[sep_elem_marker]
    lat_map_elem = map_elems[lat_elem_marker]

    # Septum
    sep_elems = np.where(sep_elem_marker)[0]
    sep_mesh, map_lv_sep, map_sep_lv = create_submesh(mmg_mesh, sep_elems)
    sep_belems, sep_bfaces = get_surface_mesh(sep_mesh)

    sep_ant = np.isin(sep_bfaces, map_lv_sep[ant_nodes])
    sep_post = np.isin(sep_bfaces, map_lv_sep[post_nodes])
    sep_bmarker = np.zeros(len(sep_bfaces), dtype=int)
    sep_bmarker[np.all(sep_ant, axis=1)] = 1
    sep_bmarker[np.all(sep_post, axis=1)] = 2
    sep_bdata = np.vstack([sep_belems, sep_bfaces.T, sep_bmarker]).T

    # Lateral side
    lat_elems = np.where(lat_elem_marker)[0]
    lat_mesh, map_lv_sep, map_lat_lv = create_submesh(mmg_mesh, lat_elems)
    lat_belems, lat_bfaces = get_surface_mesh(lat_mesh)

    lat_ant = np.isin(lat_bfaces, map_lv_sep[ant_nodes])
    lat_post = np.isin(lat_bfaces, map_lv_sep[post_nodes])
    lat_bmarker = np.zeros(len(lat_bfaces), dtype=int)
    lat_bmarker[np.all(lat_ant, axis=1)] = 1
    lat_bmarker[np.all(lat_post, axis=1)] = 2
    lat_bdata = np.vstack([lat_belems, lat_bfaces.T, lat_bmarker]).T

    # Correcting maps
    sep_map_sep = np.where(map_sep_lv<len(xyz))[0]
    sep_map_lv = map_sep_lv[sep_map_sep]
    sep_map = [sep_map_lv, sep_map_sep, sep_map_elem]

    lat_map_sep = np.where(map_lat_lv<len(xyz))[0]
    lat_map_lv = map_lat_lv[lat_map_sep]
    lat_map = [lat_map_lv, lat_map_sep, lat_map_elem]

    return sep_mesh, sep_bdata, sep_map, lat_mesh, lat_bdata, lat_map



def mmg_create_rv_circ_bc2(rv_circ1, uvc):
    # Split mesh
    mmg_mesh, mmg_bdata = pymmg.mmg_isoline_meshing(uvc.rv_mesh, rv_circ1,
                                                    funcs_to_interpolate=['long_plane'], bdata=uvc.rv_bdata_og)
    xyz = uvc.rv_mesh.points
    mmg_xyz = mmg_mesh.points

    # Find elements at isoline 0
    div_nodes = np.arange(len(xyz), len(mmg_xyz), dtype=int)   # New nodes are div nodes

    # Getting apex distance
    nodes = uvc.rv_interface_nodes
    interface_xyz = xyz[nodes]

    rv_bc1_marker_post = nodes[uvc.rvlv_normal_post@(interface_xyz - uvc.xyz[uvc.sep_endo_apex_node]).T > 0]
    rv_bc1_marker_ant = nodes[uvc.rvlv_normal_ant@(interface_xyz - uvc.xyz[uvc.sep_endo_apex_node]).T > 0]

    bmarker = np.zeros(len(mmg_xyz))
    bmarker[rv_bc1_marker_ant] =  1  # post
    bmarker[rv_bc1_marker_post] =  2  # ant
    bmarker[div_nodes] = 3

    mmg_mesh.point_data['bc2'] = bmarker
    return mmg_mesh, bmarker, mmg_bdata



def mmg_create_ot_circ_bc2(ot_circ1, uvc):
    xyz = uvc.ot_mesh.points

    # Split mesh
    mmg_mesh = pymmg.mmg_isoline_meshing(uvc.ot_mesh, ot_circ1)
    mmg_xyz = mmg_mesh.points
    mmg_ien = mmg_mesh.cells[0].data

    mmg_circ2 = np.zeros(len(mmg_xyz))
    mmg_circ2[:len(xyz)] = uvc.ot_mesh.point_data['ot_circ1']

    div_nodes = np.arange(len(xyz), len(mmg_xyz), dtype=int)   # New nodes are div nodes

    # Find lateral/septum nodes
    dist_mv = uvc.septum_vector@(mmg_xyz[div_nodes]-uvc.valve_centroids['mv']).T
    dist_tv = uvc.septum_vector@(mmg_xyz[div_nodes]-uvc.valve_centroids['tv']).T
    lat_nodes = div_nodes[dist_mv<0]
    tv_nodes = div_nodes[dist_tv>0]
    sep_nodes = np.setdiff1d(div_nodes, np.union1d(lat_nodes, tv_nodes))

    # Duplicate nodes
    new_xyz = np.zeros([len(mmg_xyz)+len(lat_nodes), 3])
    new_xyz[0:len(mmg_xyz)] = mmg_xyz
    new_xyz[len(mmg_xyz):] = mmg_xyz[lat_nodes]

    map_dup = np.arange(len(mmg_xyz))
    map_dup[lat_nodes] = np.arange(len(lat_nodes)) + len(mmg_xyz)

    elems_in_dup = np.where(np.any(np.isin(mmg_ien, lat_nodes), axis=1))[0]
    elem_circ1 = np.mean(mmg_circ2[mmg_ien], axis=1)
    pos_elems = elems_in_dup[elem_circ1[elems_in_dup]>0]

    new_ien = np.copy(mmg_ien)
    new_ien[pos_elems] = map_dup[mmg_ien[pos_elems]]

    new_mesh = io.Mesh(new_xyz, {'tetra': new_ien})

    base_nodes = np.unique(uvc.ot_bdata[uvc.ot_bdata[:,-1] == uvc.patches['base'], 1:-1])

    bmarker = np.zeros(len(new_xyz))
    bmarker[sep_nodes] = 1
    bmarker[lat_nodes] = 2
    bmarker[tv_nodes] = 4
    bmarker[len(mmg_xyz):] = 3
    bmarker[base_nodes] = 5
    new_mesh.point_data['bc2'] = bmarker

    return new_mesh, bmarker


def mmg_create_ot_circ_bc3(ot_circ2, uvc):
    xyz = uvc.ot_mesh.points

    # I need to define a function that is 0 at circ = 1 and circ =1/3
    aux = np.sin(ot_circ2*np.pi*3)

    mmg_mesh = pymmg.mmg_isoline_meshing(uvc.ot_mesh, aux)
    mmg_xyz = mmg_mesh.points
    mmg_ien = mmg_mesh.cells[0].data

    mmg_circ2 = np.zeros(len(mmg_xyz))
    mmg_circ2[:len(xyz)] = ot_circ2

    div_nodes = np.arange(len(xyz), len(mmg_xyz), dtype=int)   # New nodes are div nodes
    tree = KDTree(xyz)
    _, closest_og_node = tree.query(mmg_xyz[div_nodes])
    mmg_circ2_aux = np.copy(mmg_circ2)
    mmg_circ2_aux[div_nodes] = ot_circ2[closest_og_node]

    ant_nodes = div_nodes[np.isclose(mmg_circ2_aux[div_nodes], 1/3, rtol=1e-1)]
    post_nodes = div_nodes[np.isclose(mmg_circ2_aux[div_nodes], -1/3, rtol=1e-1)]
    sep_nodes = div_nodes[np.abs(mmg_circ2_aux[div_nodes]) < 1e-1]
    lat_nodes = div_nodes[np.isclose(np.abs(mmg_circ2_aux[div_nodes]), 1, rtol=1e-1)]

    # Duplicate nodes
    new_xyz = np.zeros([len(mmg_xyz)+len(lat_nodes), 3])
    new_xyz[0:len(mmg_xyz)] = mmg_xyz
    new_xyz[len(mmg_xyz):] = mmg_xyz[lat_nodes]

    map_dup = np.arange(len(mmg_xyz))
    map_dup[lat_nodes] = np.arange(len(lat_nodes)) + len(mmg_xyz)

    elems_in_dup = np.where(np.any(np.isin(mmg_ien, lat_nodes), axis=1))[0]
    elem_circ1 = np.mean(mmg_circ2[mmg_ien], axis=1)
    pos_elems = elems_in_dup[elem_circ1[elems_in_dup]>0]

    new_ien = np.copy(mmg_ien)
    new_ien[pos_elems] = map_dup[mmg_ien[pos_elems]]

    new_mesh = io.Mesh(new_xyz, {'tetra': new_ien})
    base_nodes = np.unique(uvc.ot_bdata[uvc.ot_bdata[:,-1] == uvc.patches['base'], 1:-1])

    bmarker = np.zeros(len(new_xyz))
    bmarker[sep_nodes] = 1
    bmarker[lat_nodes] = 2
    bmarker[len(mmg_xyz):] = 3
    bmarker[ant_nodes] = 4
    bmarker[post_nodes] = 5
    bmarker[base_nodes] = 6
    new_mesh.point_data['bc2'] = bmarker

    return new_mesh, bmarker



def mmg_create_long_bc(uvc):
    long =  uvc.bv_mesh.point_data['long_plane']
    mmg_mesh = pymmg.mmg_isoline_meshing(uvc.bv_mesh,long, uvc.thresholds['long'])
    mmg_xyz = mmg_mesh.points

    xyz = uvc.bv_mesh.points
    new_nodes = np.arange(len(xyz), len(mmg_xyz), dtype=int)   # New nodes are div nodes
    bmarker = np.zeros(len(mmg_xyz))

    for i, v in enumerate(['av', 'mv', 'pv', 'tv']):
        nodes = np.unique(uvc.bv_bdata[uvc.bv_bdata[:,-1] == uvc.patches[v], 1:-1])
        nodes = nodes[long[nodes]>1.0]
        bmarker[nodes] = i+1

    bmarker[new_nodes] = 5

    return mmg_mesh, bmarker

def compute_aha_segments(mesh, bdata, patches, coord_system, aha_type = 'points', long_cut=0.95):
    xyz = mesh.points
    septum_vector, third_vector, long_axis_vector = coord_system.T

    long_plane = mesh.point_data['long_plane']

    # Find endo apex
    lv_sep_nodes = np.unique(bdata[bdata[:,-1]==patches['rv_septum'],1:-1])
    dist_endo = long_axis_vector@xyz[lv_sep_nodes].T
    lv_endo_apex_node = lv_sep_nodes[np.argmin(dist_endo)]
    base_node = np.argmin((long_plane-long_cut)**2)

    # Compute long coord
    dist = long_axis_vector@xyz.T
    long = (dist-dist[lv_endo_apex_node])/(dist[base_node]-dist[lv_endo_apex_node])

    # Define xyz to use in AHA
    if aha_type == 'points':
        xyz_aha = np.copy(xyz)
    elif aha_type == 'elems':
        ien = mesh.cells[0].data
        xyz_aha = np.mean(xyz[ien], axis=1)
        long = np.mean(long[ien], axis=1)


    # Project coordinates to local cardinal system
    proj_xyz = (xyz_aha-xyz[lv_endo_apex_node])@coord_system

    # Long axis division
    base_marker = (long>=2/3)*(long<=1)
    mid_marker = (long>=1/3)*(long<2/3)
    apex_marker = (long>=0)*(long<1/3)
    apex_apex_marker = (long<0)

    # Compute circ for easiness
    base_centroid = np.mean(proj_xyz[base_marker], axis=0)
    mid_centroid = np.mean(proj_xyz[mid_marker], axis=0)
    apex_centroid = np.mean(proj_xyz[apex_marker], axis=0)

    base_xyz = proj_xyz - base_centroid
    mid_xyz = proj_xyz - mid_centroid
    apex_xyz = proj_xyz - apex_centroid

    base_circ = -np.rad2deg(np.arctan2(base_xyz[:,1], base_xyz[:,0]))
    mid_circ = -np.rad2deg(np.arctan2(mid_xyz[:,1], mid_xyz[:,0]))
    apex_circ = -np.rad2deg(np.arctan2(apex_xyz[:,1], apex_xyz[:,0]))

    # Define AHA
    aha_region = np.zeros(len(xyz_aha))
    aha_region[(base_marker)*(base_circ>60)*(base_circ<=120)] = 4
    aha_region[(base_marker)*(base_circ>120)*(base_circ<=180)] = 5
    aha_region[(base_marker)*(base_circ>=-180)*(base_circ<=-120)] = 6
    aha_region[(base_marker)*(base_circ>-120)*(base_circ<=-60)] = 1
    aha_region[(base_marker)*(base_circ>-60)*(base_circ<=0)] = 2
    aha_region[(base_marker)*(base_circ>=0)*(base_circ<=60)] = 3

    aha_region[(mid_marker)*(mid_circ>60)*(mid_circ<=120)] = 10
    aha_region[(mid_marker)*(mid_circ>120)*(mid_circ<=180)] = 11
    aha_region[(mid_marker)*(mid_circ>=-180)*(mid_circ<=-120)] = 12
    aha_region[(mid_marker)*(mid_circ>-120)*(mid_circ<=-60)] = 7
    aha_region[(mid_marker)*(mid_circ>-60)*(mid_circ<=0)] = 8
    aha_region[(mid_marker)*(mid_circ>=0)*(mid_circ<=60)] = 9

    aha_region[(apex_marker)*(apex_circ>45)*(apex_circ<=135)] = 15
    aha_region[(apex_marker)*((apex_circ>135)+(apex_circ<=-135))] = 16
    aha_region[(apex_marker)*(apex_circ>-135)*(apex_circ<=-45)] = 13
    aha_region[(apex_marker)*(apex_circ>-45)*(apex_circ<=45)] = 14

    aha_region[apex_apex_marker] = 17

    return aha_region


def compute_simple_lv_aha_segments(mesh, bdata, patches, coord_system, aha_type='points', long_div=[0,1/3,2/3]):
    xyz = mesh.points
    septum_vector, third_vector, long_axis_vector = coord_system.T

    # Find endo apex
    lv_endo_nodes = np.unique(bdata[bdata[:,-1]==patches['endo'],1:-1])
    dist_endo = long_axis_vector@xyz[lv_endo_nodes].T
    endo_apex = lv_endo_nodes[np.argmin(dist_endo)]
    base_node = lv_endo_nodes[np.argmax(dist_endo)]

    # Compute long coord
    dist = long_axis_vector@xyz.T
    long = (dist-dist[endo_apex])/(dist[base_node]-dist[endo_apex])

    # Define xyz to use in AHA
    if aha_type == 'points':
        xyz_aha = np.copy(xyz)
    elif aha_type == 'elems':
        ien = mesh.cells[0].data
        xyz_aha = np.mean(xyz[ien], axis=1)
        long = np.mean(long[ien], axis=1)
    else:
        raise 'wrong AHA type. Only "points" or "elems" accepted'



    # Project coordinates to local cardinal system
    proj_xyz = (xyz_aha-xyz[endo_apex])@coord_system

    # Long axis division
    base_marker = (long>=long_div[2])*(long<=1)
    mid_marker = (long>=long_div[1])*(long<long_div[2])
    apex_marker = (long>=long_div[0])*(long<long_div[1])
    apex_apex_marker = (long<long_div[0])

    # Compute circ for easiness
    base_centroid = np.mean(proj_xyz[base_marker], axis=0)
    mid_centroid = np.mean(proj_xyz[mid_marker], axis=0)
    apex_centroid = np.mean(proj_xyz[apex_marker], axis=0)

    base_xyz = proj_xyz - base_centroid
    mid_xyz = proj_xyz - mid_centroid
    apex_xyz = proj_xyz - apex_centroid

    base_circ = np.rad2deg(np.arctan2(base_xyz[:,1], base_xyz[:,0]))
    mid_circ = np.rad2deg(np.arctan2(mid_xyz[:,1], mid_xyz[:,0]))
    apex_circ = np.rad2deg(np.arctan2(apex_xyz[:,1], apex_xyz[:,0]))

    # Define AHA
    aha_region = np.zeros(len(xyz_aha), dtype=int)
    aha_region[(base_marker)*(base_circ>60)*(base_circ<=120)] = 1
    aha_region[(base_marker)*(base_circ>120)*(base_circ<=180)] = 2
    aha_region[(base_marker)*(base_circ>=-180)*(base_circ<=-120)] = 3
    aha_region[(base_marker)*(base_circ>-120)*(base_circ<=-60)] = 4
    aha_region[(base_marker)*(base_circ>-60)*(base_circ<=0)] = 5
    aha_region[(base_marker)*(base_circ>=0)*(base_circ<=60)] = 6

    aha_region[(mid_marker)*(mid_circ>60)*(mid_circ<=120)] = 7
    aha_region[(mid_marker)*(mid_circ>120)*(mid_circ<=180)] = 8
    aha_region[(mid_marker)*(mid_circ>=-180)*(mid_circ<=-120)] = 9
    aha_region[(mid_marker)*(mid_circ>-120)*(mid_circ<=-60)] = 10
    aha_region[(mid_marker)*(mid_circ>-60)*(mid_circ<=0)] = 11
    aha_region[(mid_marker)*(mid_circ>=0)*(mid_circ<=60)] = 12

    aha_region[(apex_marker)*(apex_circ>45)*(apex_circ<=135)] = 13
    aha_region[(apex_marker)*((apex_circ>135)+(apex_circ<=-135))] = 14
    aha_region[(apex_marker)*(apex_circ>-135)*(apex_circ<=-45)] = 15
    aha_region[(apex_marker)*(apex_circ>-45)*(apex_circ<=45)] = 16

    aha_region[apex_apex_marker] = 17

    return aha_region


def compute_simple_lv_aha_long(mesh, bdata, patches, coord_system, long_aha, aha_type='points'):
    xyz = mesh.points
    septum_vector, third_vector, long_axis_vector = coord_system.T

    # Find endo apex
    lv_endo_nodes = np.unique(bdata[bdata[:,-1]==patches['endo'],1:-1])
    dist_endo = long_axis_vector@xyz[lv_endo_nodes].T
    endo_apex = lv_endo_nodes[np.argmin(dist_endo)]
    base_node = lv_endo_nodes[np.argmax(dist_endo)]

    # Compute long coord
    dist = long_axis_vector@xyz.T
    long = (dist-dist[endo_apex])/(dist[base_node]-dist[endo_apex])

    # Define xyz to use in AHA
    if aha_type == 'points':
        xyz_aha = np.copy(xyz)
    elif aha_type == 'elems':
        ien = mesh.cells[0].data
        xyz_aha = np.mean(xyz[ien], axis=1)
        long = np.mean(long[ien], axis=1)
    else:
        raise 'wrong AHA type. Only "points" or "elems" accepted'

    # Project coordinates to local cardinal system
    proj_xyz = (xyz_aha-xyz[endo_apex])@coord_system

    # Long axis division
    base_marker = long_aha == 3
    mid_marker = long_aha == 2
    apex_marker = long_aha == 1
    apex_apex_marker = long_aha == 0

    # Compute circ for easiness
    base_centroid = np.mean(proj_xyz[base_marker], axis=0)
    mid_centroid = np.mean(proj_xyz[mid_marker], axis=0)
    apex_centroid = np.mean(proj_xyz[apex_marker], axis=0)

    base_xyz = proj_xyz - base_centroid
    mid_xyz = proj_xyz - mid_centroid
    apex_xyz = proj_xyz - apex_centroid

    base_circ = np.rad2deg(np.arctan2(base_xyz[:,1], base_xyz[:,0]))
    mid_circ = np.rad2deg(np.arctan2(mid_xyz[:,1], mid_xyz[:,0]))
    apex_circ = np.rad2deg(np.arctan2(apex_xyz[:,1], apex_xyz[:,0]))

    # Define AHA
    aha_region = np.zeros(len(xyz_aha), dtype=int)
    aha_region[(base_marker)*(base_circ>60)*(base_circ<=120)] = 1
    aha_region[(base_marker)*(base_circ>120)*(base_circ<=180)] = 2
    aha_region[(base_marker)*(base_circ>=-180)*(base_circ<=-120)] = 3
    aha_region[(base_marker)*(base_circ>-120)*(base_circ<=-60)] = 4
    aha_region[(base_marker)*(base_circ>-60)*(base_circ<=0)] = 5
    aha_region[(base_marker)*(base_circ>=0)*(base_circ<=60)] = 6

    aha_region[(mid_marker)*(mid_circ>60)*(mid_circ<=120)] = 7
    aha_region[(mid_marker)*(mid_circ>120)*(mid_circ<=180)] = 8
    aha_region[(mid_marker)*(mid_circ>=-180)*(mid_circ<=-120)] = 9
    aha_region[(mid_marker)*(mid_circ>-120)*(mid_circ<=-60)] = 10
    aha_region[(mid_marker)*(mid_circ>-60)*(mid_circ<=0)] = 11
    aha_region[(mid_marker)*(mid_circ>=0)*(mid_circ<=60)] = 12

    aha_region[(apex_marker)*(apex_circ>45)*(apex_circ<=135)] = 13
    aha_region[(apex_marker)*((apex_circ>135)+(apex_circ<=-135))] = 14
    aha_region[(apex_marker)*(apex_circ>-135)*(apex_circ<=-45)] = 15
    aha_region[(apex_marker)*(apex_circ>-45)*(apex_circ<=45)] = 16

    aha_region[apex_apex_marker] = 17


    return aha_region