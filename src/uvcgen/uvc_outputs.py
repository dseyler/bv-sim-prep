#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 16:23:48 2023

@author: Javiera Jilberto Vallejos
"""

import numpy as np
import cheartio as chio

def export_coordinate_system(uvc, write_ch=True):
    """
    Parameters
    ----------
    uvc : UVC.
    write_ch : bool, optional
        If to write cheart output or not. The default is True.
    Returns
    -------
    Q : np.array
        3x3 array, each colum is a vector. sep-lat/ant-post/long axis.
    """
    v3 = uvc.long_axis_vector
    v1 = uvc.septum_vector
    v2 = np.cross(v3, v1)

    Q = np.array([v1,v2,v3]).T

    if write_ch:
        chio.write_dfile(uvc.out_folder + 'coordinate_system.FE', Q)

    return Q



def export_valve_info(uvc, write_ch=True):
    """
    Parameters
    ----------
    uvc : UVC.
    write_ch : bool, optional
        If to write cheart output or not. The default is True.
    Returns
    -------
    centroids : np.array
        3x4 array, each colum is an xyz position. av/mv/pv/tv.
    normals : np.array
        3x4 array, each colum is a normal vector. av/mv/pv/tv.
    """
    # Only include valves that exist
    valves = [v for v in ['av', 'mv', 'pv', 'tv'] if v in uvc.valve_centroids]
    centroids = []
    normals = []
    for v in valves:
        centroids.append(uvc.valve_centroids[v])
        normals.append(uvc.valve_normals[v])

    centroids = np.vstack(centroids).T
    normals = np.vstack(normals).T

    if write_ch:
        chio.write_dfile(uvc.out_folder + 'valve_normals.FE', normals)
        chio.write_dfile(uvc.out_folder + 'valve_centroids.FE', centroids)

    return centroids, normals


def export_point_data(uvc, fields, write_ch=True):
    for key in fields:
        chio.write_dfile(uvc.out_folder + key + '.FE', uvc.bv_mesh.point_data[key])


def export_ch_write_meshes(uvc, which='all'):
    if which == 'all':
        for key in uvc.meshes.keys():
            chio.write_mesh(uvc.out_folder + key + '_model', uvc.meshes[key].points, uvc.meshes[key].cells[0].data)
            chio.write_bfile(uvc.out_folder + key + '_model', uvc.bdatas[key])


def export_apex_nodes(uvc):
    nodes_dic = {'lv_sep_endo': uvc.bv_sep_apex_nodes[0],
                 'sep_epi':     uvc.bv_sep_apex_nodes[1],
                 'sep_endo':    uvc.bv_sep_apex_nodes[2]}
    np.save(uvc.out_folder + 'apex_nodes', nodes_dic)


# def write_data(self, method='dict'):   # TODO write to a P file
#     data = {'long_axis_vector': self.long_axis_vector,
#             'long_axis_length': self.long_axis_length,
#             'septum_vector': self.septum_vector,
#             'valve_centroids': self.valve_centroids,
#             'valve_normals': self.valve_normals,
#             'bv_apex_nodes': self.bv_sep_apex_nodes}

#     if method == 'dict':
#         import pickle
#         with open(self.out_folder + 'uvc_info.pkl', 'wb') as f:
#             pickle.dump(data, f)

def export_origin(uvc):
    chio.write_dfile(uvc.out_folder + 'origin.FE', uvc.origin)


def export_mappings(uvc, which, map_type='points'):
    if map_type == 'points':
        if which == 'bv2lv':
            chio.write_dfile(uvc.out_folder + 'bv2lv_map.FE', uvc.map_lv_bv, fmt='%i')
        if which == 'lv2bv':
            chio.write_dfile(uvc.out_folder + 'lv2bv_map.FE', uvc.map_bv_lv, fmt='%i')
    if map_type == 'elems':
        if which == 'lv2bv':
            chio.write_dfile(uvc.out_folder + 'lv2bv_elems_map.FE', uvc.map_bv_lv_elems, fmt='%i')


def export_info(uvc):
    info = {}

    # Valve info - only include valves that exist
    valves = [v for v in ['av', 'mv', 'pv', 'tv'] if v in uvc.valve_centroids]
    for v in valves:
        info[v + '_centroid'] = uvc.valve_centroids[v]
        info[v + '_normal'] = uvc.valve_normals[v]

    # N3 vector - use AV if exists, otherwise MV
    base_valve = 'av' if uvc.has_av else 'mv'
    n3 = np.cross(info['mv_normal'], info[base_valve + '_normal'])
    info['lv_N3'] = n3/np.linalg.norm(n3)
    n3 = np.cross(info['tv_normal'], info['pv_normal'])
    info['rv_N3'] = n3/np.linalg.norm(n3)

    # Apex info
    info['apex_lv_sep_endo'] = uvc.bv_sep_apex_nodes[0]
    info['apex_rv_sep_endo'] = uvc.bv_sep_apex_nodes[1]
    info['apex_sep_epi'] = uvc.bv_sep_apex_nodes[2]
    info['apex_lv_epi'] = uvc.bv_lv_epi_apex_node
    info['apex_lv_endo'] = uvc.bv_lv_endo_apex_node

    # Coordinate system
    info['long_axis_vector'] = uvc.long_axis_vector
    info['septum_vector'] = uvc.septum_vector

    np.save(uvc.out_folder + 'info', info, allow_pickle=True)

    return info

def export_cheart_inputs(uvc, which='all'):
    # Valve normals - only include valves that exist
    valves = [v for v in ['av', 'mv', 'pv', 'tv'] if v in uvc.valve_normals]
    for v in valves:
        chio.write_dfile(uvc.out_folder + v + '_normal.FE', uvc.valve_normals[v][None])
        print(v, uvc.valve_normals[v][None])
    # N3 vector - use AV if exists, otherwise MV
    base_valve = 'av' if uvc.has_av else 'mv'
    n3 = np.cross(uvc.valve_normals['mv'], uvc.valve_normals[base_valve])
    lv_n3 = n3/np.linalg.norm(n3)
    n3 = np.cross(uvc.valve_normals['tv'], uvc.valve_normals['pv'])
    rv_n3 = n3/np.linalg.norm(n3)

    chio.write_dfile(uvc.out_folder + 'N3_lv.FE', lv_n3[None])
    chio.write_dfile(uvc.out_folder + 'N3_rv.FE', rv_n3[None])
    print('N3lv', lv_n3[None])
    print('N3rv', rv_n3[None])

    for key, value in uvc.bv_mesh.point_data.items():
        chio.write_dfile(uvc.out_folder + key + '.FE', value)

    # Patch ids
    f = open(uvc.out_folder + 'boundaries.P', "w")
    for key in uvc.patches:
        f.write('#{}={}\n'.format(key, uvc.patches[key]))
    f.close()
