#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 15:32:08 2023

@author: Javiera Jilberto Vallejos
"""

import numpy as np
from collections import deque
import dolfinxio as dxio
from LaplaceProblem import LaplaceProblem, TrajectoryProblem
import uvcgen.UVC as uc
import meshio as io
from dolfinx.log import set_log_level, LogLevel
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.spatial import KDTree

set_log_level(LogLevel.WARNING)

def run_coord(mesh_, bdata_, bcs_marker, return_grad=False, diffusion=False):

    mesh, mt = dxio.read_meshio_mesh(mesh_, bdata_)
    LapSolver = LaplaceProblem(mesh, mt)

    if diffusion:
        # lap = LapSolver.solve_diffusion(bcs_marker)
        lap = LapSolver.solve(bcs_marker)
    else:
        lap = LapSolver.solve(bcs_marker)

    corr, _ = dxio.find_vtu_dx_mapping(mesh)
    if return_grad:
        grad = LapSolver.get_linear_gradient(lap)
        dxio.visualize_function('grad.xdmf', grad)
        grad = grad.x.petsc_vec.array.reshape([-1, 3])[corr]
        lap = lap.x.petsc_vec.array[corr]
        return lap, grad

    return lap.x.petsc_vec.array[corr]


def run_trajectory_coord(mesh_, bdata_, bcs_marker, vector=None):

    mesh, mt = dxio.read_meshio_mesh(mesh_, bdata_)
    TPSolver = TrajectoryProblem(mesh, mt)

    if vector is not None:
        lap = TPSolver.solve_with_vector(bcs_marker, vector)
    else:
        lap = TPSolver.solve(bcs_marker)

    corr, _ = dxio.find_vtu_dx_mapping(mesh)
    lap = lap.x.petsc_vec.array[corr]

    return lap


class UVCGen:
    def __init__(self, uvc, mmg=False):
        self.xyz_og = uvc.xyz
        self.bndry = uvc.patches
        self.mmg = mmg
        # Soft-Neumann control for longitudinal solve
        self.use_soft_long_base = True
        self.long_base_fraction = 0.2  # fraction of valve nodes kept as Dirichlet
        self.long_base_min_nodes = 12  # minimum number of nodes per valve kept

    def get_solver(self, method, mesh='bv'):
        if method == 'laplace':
            if mesh == 'bv':
                return self.LapSolver
            elif mesh == 'lv':
                return self.lv_LapSolver
            elif mesh == 'rv':
                return self.rv_LapSolver
            elif mesh == 'ot':
                return self.ot_LapSolver
        elif method == 'trajectory':
            if mesh == 'bv':
                return self.TPSolver
            elif mesh == 'lv':
                return self.lv_TPSolver
            elif mesh == 'rv':
                return self.rv_TPSolver
            elif mesh == 'ot':
                return self.ot_TPSolver
        else:
            raise ('Method unknown, please choose laplace or trajectory')

    def init_bv_mesh(self, uvc, og=False):
        if og:
            bdata = uvc.bv_bdata_og
        else:
            bdata = uvc.bv_bdata
        self.bv_mesh, self.bv_mt = dxio.read_meshio_mesh(
            uvc.bv_mesh, bdata)

        self.LapSolver = LaplaceProblem(self.bv_mesh, self.bv_mt)
        # self.TPSolver = TrajectoryProblem(self.bv_mesh, self.bv_mt)
        self.bv_corr, self.bv_icorr = dxio.find_vtu_dx_mapping(self.bv_mesh)

    def init_rv_mesh(self, uvc, og=False):
        if og:
            bdata = uvc.rv_bdata_og
        else:
            bdata = uvc.rv_bdata
        self.rv_mesh, self.rv_mt = dxio.read_meshio_mesh(
            uvc.rv_mesh, bdata)
        self.rv_LapSolver = LaplaceProblem(self.rv_mesh, self.rv_mt)
        # self.rv_TPSolver = TrajectoryProblem(self.rv_mesh, self.rv_mt)
        self.rv_corr, self.rv_icorr = dxio.find_vtu_dx_mapping(self.rv_mesh)

    def init_lv_mesh(self, uvc, og=False):
        if og:
            bdata = uvc.lv_bdata_og
        else:
            bdata = uvc.lv_bdata
        self.lv_mesh, self.lv_mt = dxio.read_meshio_mesh(
            uvc.lv_mesh, bdata)
        self.lv_LapSolver = LaplaceProblem(self.lv_mesh, self.lv_mt)
        # self.lv_TPSolver = TrajectoryProblem(self.lv_mesh, self.lv_mt)
        self.lv_corr, self.lv_icorr = dxio.find_vtu_dx_mapping(self.lv_mesh)

    def init_ot_mesh(self, uvc, og=False):
        if og:
            bdata = uvc.ot_bdata_og
        else:
            bdata = uvc.ot_bdata
        self.ot_mesh, self.ot_mt = dxio.read_meshio_mesh(
            uvc.ot_mesh, bdata)
        self.ot_LapSolver = LaplaceProblem(self.ot_mesh, self.ot_mt)
        # self.ot_TPSolver = TrajectoryProblem(self.ot_mesh, self.ot_mt)
        self.ot_corr, self.ot_icorr = dxio.find_vtu_dx_mapping(self.ot_mesh)


    def get_func_gradient(self, uvc, func, which, linear=False):
        if which == 'lv':
            self.init_lv_mesh(uvc)
            array = uvc.lv_mesh.point_data[func][self.lv_icorr]
            if linear:
                corr, _ = dxio.find_vtu_dx_mapping(self.lv_mesh)
            else:
                corr, _ = dxio.find_vtu_dx_mapping(self.lv_mesh, cells=True)
            g = self.lv_LapSolver.get_array_gradient(array, linear=linear)
            g = g.x.petsc_vec.array.reshape([-1,3])[corr]
        elif which == 'rv':
            self.init_rv_mesh(uvc)
            array = uvc.rv_mesh.point_data[func][self.rv_icorr]
            if linear:
                corr, _ = dxio.find_vtu_dx_mapping(self.rv_mesh)
            else:
                corr, _ = dxio.find_vtu_dx_mapping(self.rv_mesh, cells=True)
            g = self.rv_LapSolver.get_array_gradient(array, linear=linear)
            g = g.x.petsc_vec.array.reshape([-1,3])[corr]

        return g

    def run_septum(self, uvc):
        self.init_bv_mesh(uvc)

        # Run septum problem
        # Use AV if it exists, otherwise MV (already included)
        base_valve_patch = self.bndry.get('av', self.bndry['mv'])
        bcs_septum = {'face': {self.bndry['lv_endo']: 0,
                      self.bndry['mv']: 0,
                      base_valve_patch: 0,
                      self.bndry['rv_septum']: 0.5,
                      self.bndry['rv_endo']: 1,
                      self.bndry['tv']: 1,
                      self.bndry['pv']: 1}}

        if uvc.split_epi:
            bcs_septum['face'][self.bndry['lv_epi']] = 0
            bcs_septum['face'][self.bndry['rv_epi']] = 1

        septum = self.LapSolver.solve(bcs_septum)
        septum = septum.x.petsc_vec.array[self.bv_corr]
        uvc.bv_mesh.point_data['septum'] = septum

        return septum

    def correct_septum(self, uvc, septum0):
        self.init_bv_mesh(uvc)
        uvc.define_septum_bc(septum0)
        self.init_bv_mesh(uvc)

        # Use AV if it exists, otherwise MV (already included)
        base_valve_patch = self.bndry.get('av', self.bndry['mv'])
        bcs_septum = {'face': {self.bndry['lv_endo']: 0,
                      12: 0,
                      self.bndry['mv']: 0,
                      base_valve_patch: 0,
                      self.bndry['rv_septum']: 0.5,
                      self.bndry['rv_endo']: 1,
                      self.bndry['tv']: 1,
                      self.bndry['pv']: 1}}

        septum = self.LapSolver.solve(bcs_septum)
        uvc.bv_mesh.point_data['septum'] = septum

        return septum.x.petsc_vec.array[self.bv_corr]

    def _get_default_long_face_bcs(self, uvc):
        """
        Build the default Dirichlet BC dictionary (value 1) for the valve faces.
        """
        face_bcs = {}
        for patch_name in ('mv', 'pv', 'tv'):
            patch_id = self.bndry.get(patch_name)
            if patch_id is not None:
                face_bcs[patch_id] = 1.0
        if uvc.has_av:
            av_patch = self.bndry.get('av')
            if av_patch is not None:
                face_bcs[av_patch] = 1.0
        return face_bcs

    def _get_epi_node_set(self, uvc):
        """Return a set with all BV node indices belonging to epi patches."""
        epi_patch_ids = []
        for patch_name in ('lv_epi', 'rv_epi', 'epi'):
            pid = self.bndry.get(patch_name)
            if pid is not None:
                epi_patch_ids.append(pid)
        if not epi_patch_ids:
            return set()

        bdata = uvc.bv_bdata
        epi_nodes = []
        for pid in epi_patch_ids:
            faces = bdata[bdata[:, -1] == pid]
            if len(faces) == 0:
                continue
            epi_nodes.extend(np.unique(faces[:, 1:-1]).astype(int).tolist())
        return set(epi_nodes)

    def _collect_valve_dirichlet_nodes_excluding_epi(self, uvc):
        """
        Collect all valve surface nodes (mv/pv/tv[/av]) except those that touch
        an epicardial patch. These nodes receive Dirichlet=1 point BCs, while
        the excluded nodes are left natural (Neumann).
        """
        if not hasattr(uvc, 'bv_mesh'):
            return {}
        mesh_points = uvc.bv_mesh.points
        bdata = uvc.bv_bdata
        epi_nodes = self._get_epi_node_set(uvc)
        valve_names = ['mv', 'pv', 'tv']
        if uvc.has_av:
            valve_names.append('av')

        dirichlet_points = {}
        total = 0
        removed = 0
        for name in valve_names:
            patch_id = self.bndry.get(name)
            if patch_id is None:
                continue
            faces = bdata[bdata[:, -1] == patch_id]
            if len(faces) == 0:
                continue
            nodes = np.unique(faces[:, 1:-1]).astype(int)
            for nid in nodes:
                if nid in epi_nodes:
                    removed += 1
                    continue
                dirichlet_points[tuple(mesh_points[nid])] = 1.0
                total += 1
        print(f"  Valve Dirichlet nodes (excluding epi borders): kept {total}, "
              f"excluded {removed}")
        return dirichlet_points

    def _collect_valve_nodes_for_mesh(self, bdata, which):
        """
        Return numpy array of node indices on valve patches for the specified mesh.
        """
        valve_names = []
        if which in ('lv', 'bv'):
            valve_names.extend(['mv'])
            if 'av' in self.bndry:
                valve_names.append('av')
        if which in ('rv', 'bv'):
            valve_names.extend(['tv', 'pv'])

        nodes = []
        for name in valve_names:
            patch_id = self.bndry.get(name)
            if patch_id is None:
                continue
            faces = bdata[bdata[:, -1] == patch_id]
            if len(faces) == 0:
                continue
            nodes.extend(np.unique(faces[:, 1:-1]).astype(int).tolist())
        if not nodes:
            return np.array([], dtype=int)
        return np.unique(np.asarray(nodes, dtype=int))

    def run_longitudinal(self, uvc, method='laplace'):
        """
        Solve the longitudinal Laplace problem on the BV mesh using apex nodes
        (Dirichlet=0) and valve anchors (Dirichlet=1). When soft-base mode is
        enabled, only the valve nodes closest to the centroid are kept as
        Dirichlet constraints, leaving the remainder of the base to satisfy a
        natural (Neumann) boundary condition that eases the transition near the
        apex.
        """
        apex_nodes = getattr(uvc, 'bv_sep_apex_nodes', None)
        if apex_nodes is not None and len(apex_nodes) > 0:
            apex_nodes = np.atleast_1d(apex_nodes).astype(int)
            apex_coords = uvc.bv_mesh.points[apex_nodes]
            print("Apex nodes for longitudinal BCs (BV mesh indices and coordinates):")
            for nid, coord in zip(apex_nodes, apex_coords):
                print(f"  node {nid}: {coord}")
        else:
            apex_nodes = None

        bcs_point = {}
        if apex_nodes is not None and len(apex_nodes) > 0:
            for nid in apex_nodes:
                bcs_point[tuple(uvc.bv_mesh.points[nid])] = 0.0
        else:
            print("  Warning: No bv_sep_apex_nodes defined on UVC object; "
                  "BV longitudinal solve will not have apex point Dirichlet BCs.")

        face_bcs = {}
        valve_points = self._collect_valve_dirichlet_nodes_excluding_epi(uvc)
        if valve_points:
            bcs_point.update(valve_points)
        else:
            print("  Warning: Could not collect valve Dirichlet nodes; reverting to face BCs.")
            face_bcs = self._get_default_long_face_bcs(uvc)
            if not face_bcs:
                print("  Warning: No valve face BCs available; Laplace solve may be ill-posed.")

        bcs_marker = {'point': bcs_point}
        if face_bcs:
            bcs_marker['face'] = face_bcs

        self.init_bv_mesh(uvc)
        bv_solver = self.get_solver(method, 'bv')
        bv_long = bv_solver.solve(bcs_marker)
        bv_long = bv_long.x.petsc_vec.array[self.bv_corr]
        uvc.bv_mesh.point_data['long'] = bv_long

        if hasattr(uvc, 'lv_mesh'):
            uvc.lv_mesh.point_data['long'] = bv_long[uvc.map_lv_bv]
        if hasattr(uvc, 'rv_mesh'):
            uvc.rv_mesh.point_data['long'] = bv_long[uvc.map_rv_bv]

        return [bv_long]

    def correct_longitudinal(self, uvc):
        """
        Reparameterize the BV long field so that it increases linearly from the
        apex to the valves along the LV epicardial gradient curve, then map the
        corrected field to LV and RV meshes.
        """
        bv_mesh = uvc.bv_mesh
        bdata = uvc.bv_bdata
        if 'long' not in bv_mesh.point_data:
            print("  Warning: BV mesh missing 'long' field; skipping longitudinal correction.")
            return uvc.lv_mesh.point_data.get('long'), uvc.rv_mesh.point_data.get('long')

        bv_long = np.copy(bv_mesh.point_data['long'])
        zero_nodes = getattr(uvc, 'bv_sep_apex_nodes', None)
        if zero_nodes is None or len(zero_nodes) == 0:
            print("  Warning: No apex nodes available for BV longitudinal correction.")
            return uvc.lv_mesh.point_data.get('long'), uvc.rv_mesh.point_data.get('long')

        grad_curve = self._build_long_gradient_curve(bv_mesh, bdata, bv_long, zero_nodes, uvc)
        if grad_curve is None:
            print("  Warning: Could not build longitudinal gradient curve; skipping correction.")
            return uvc.lv_mesh.point_data.get('long'), uvc.rv_mesh.point_data.get('long')

        curve_nodes, grad_curve_field = grad_curve
        path_long = bv_long[curve_nodes]
        path_param = grad_curve_field[curve_nodes]

        order = np.argsort(path_long)
        path_long = path_long[order]
        path_param = path_param[order]

        unique_long, unique_indices = np.unique(path_long, return_index=True)
        unique_param = path_param[unique_indices]
        if len(unique_long) < 2:
            print("  Warning: Not enough unique long values along gradient curve; skipping correction.")
            return uvc.lv_mesh.point_data.get('long'), uvc.rv_mesh.point_data.get('long')

        norm_func = interp1d(unique_long, unique_param, fill_value='extrapolate')
        norm_long = norm_func(bv_long)
        norm_long = np.clip(norm_long, 0.0, 1.0)

        bv_mesh.point_data['long'] = norm_long
        bv_mesh.point_data['long_grad_curve'] = grad_curve_field

        if hasattr(uvc, 'lv_mesh'):
            uvc.lv_mesh.point_data['long'] = norm_long[uvc.map_lv_bv]
            uvc.lv_mesh.point_data['long_grad_curve'] = grad_curve_field[uvc.map_lv_bv]
        if hasattr(uvc, 'rv_mesh'):
            uvc.rv_mesh.point_data['long'] = norm_long[uvc.map_rv_bv]
            uvc.rv_mesh.point_data['long_grad_curve'] = grad_curve_field[uvc.map_rv_bv]

        return uvc.lv_mesh.point_data.get('long'), uvc.rv_mesh.point_data.get('long')

    def _build_long_gradient_curve(self, mesh, bdata, long_vals, zero_nodes, uvc,
                                   max_long=0.995, long_tol=1e-4):
        """
        Follow the gradient of the longitudinal field along the LV epicardial
        surface, starting from the apex zero nodes, until reaching the mitral
        valve or long == 1. Returns (ordered_nodes, curve_field) or None.
        """
        if zero_nodes is None or len(zero_nodes) == 0:
            return None

        npts = len(mesh.points)
        adjacency = self._build_node_adjacency(mesh)

        epi_patch = self.bndry.get('lv_epi', self.bndry.get('epi'))
        mv_patch = self.bndry.get('mv')
        if epi_patch is None or mv_patch is None:
            return None

        epi_nodes = np.unique(bdata[bdata[:, -1] == epi_patch, 1:-1]).astype(int)
        epi_mask = np.zeros(npts, dtype=bool)
        epi_mask[epi_nodes] = True

        mv_nodes = np.unique(bdata[bdata[:, -1] == mv_patch, 1:-1]).astype(int)
        if len(mv_nodes) == 0:
            return None
        mv_mask = np.zeros(npts, dtype=bool)
        mv_mask[mv_nodes] = True

        best_path = None
        best_long = -np.inf
        for start in np.atleast_1d(zero_nodes):
            if start < 0 or start >= npts:
                continue
            path = self._follow_long_gradient_path(int(start), adjacency, long_vals,
                                                   epi_mask, mv_mask, max_long, long_tol)
            if len(path) >= 2 and long_vals[path[-1]] > best_long:
                best_path = path
                best_long = long_vals[path[-1]]

        if best_path is None:
            return None

        ordered_nodes = np.array(best_path, dtype=int)
        points = mesh.points[ordered_nodes]
        seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
        dist = np.append(0.0, np.cumsum(seg))
        total = dist[-1]
        if total <= 0:
            return None
        param = dist / total

        curve_field = np.zeros(npts)
        curve_field[ordered_nodes] = param
        curve_field[np.atleast_1d(zero_nodes)] = 0.0

        return ordered_nodes, curve_field

    @staticmethod
    def _build_node_adjacency(mesh):
        cells = mesh.cells[0].data
        npts = len(mesh.points)
        adjacency = [set() for _ in range(npts)]
        for cell in cells:
            for i in range(len(cell)):
                ni = cell[i]
                for j in range(i + 1, len(cell)):
                    nj = cell[j]
                    adjacency[ni].add(nj)
                    adjacency[nj].add(ni)
        adjacency = [np.array(list(neigh), dtype=int) if len(neigh) > 0 else np.array([], dtype=int)
                     for neigh in adjacency]
        return adjacency

    def _follow_long_gradient_path(self, start, adjacency, long_vals, epi_mask, mv_mask,
                                   max_long, long_tol):
        path = []
        current = start
        visited = set()

        while True:
            path.append(current)
            visited.add(current)
            if mv_mask[current] or long_vals[current] >= max_long:
                break
            neighbors = adjacency[current]
            if neighbors.size == 0:
                break
            candidates = [nbr for nbr in neighbors
                          if (epi_mask[nbr] or mv_mask[nbr]) and
                          long_vals[nbr] > long_vals[current] + long_tol]
            if not candidates:
                candidates = [nbr for nbr in neighbors
                              if (epi_mask[nbr] or mv_mask[nbr]) and
                              long_vals[nbr] > long_vals[current]]
            if not candidates:
                break
            next_node = candidates[np.argmax(long_vals[candidates])]
            if next_node in visited:
                break
            current = next_node

        return path

    def _follow_surface_long_gradient_path(self, start, adjacency, long_vals, surface_mask,
                                           descending=False, long_tol=1e-4, max_steps=2000,
                                           fallback_point=None, points=None, debug_name=None):
        if start < 0 or start >= len(surface_mask) or not surface_mask[start]:
            if debug_name:
                print(f"    DEBUG {debug_name}: Invalid start node {start} (mask check failed)")
            return []
        path = []
        current = start
        visited = set()
        steps = 0
        while steps < max_steps:
            path.append(current)
            visited.add(current)
            neighbors = adjacency[current]
            allowed = [nbr for nbr in neighbors if surface_mask[nbr] and nbr not in visited]
            if not allowed:
                if debug_name and steps == 0:
                    print(f"    DEBUG {debug_name}: No allowed neighbors from start node {current}")
                break
            if descending:
                candidates = [nbr for nbr in allowed if long_vals[nbr] < long_vals[current] - long_tol]
                if not candidates:
                    candidates = [nbr for nbr in allowed if long_vals[nbr] < long_vals[current]]
                selector = np.argmin
            else:
                candidates = [nbr for nbr in allowed if long_vals[nbr] > long_vals[current] + long_tol]
                if not candidates:
                    candidates = [nbr for nbr in allowed if long_vals[nbr] > long_vals[current]]
                selector = np.argmax
            if not candidates:
                if fallback_point is not None and points is not None and allowed:
                    direction = points[current] - fallback_point
                    direction_norm = np.linalg.norm(direction)
                    if debug_name and steps == 0:
                        print(f"    DEBUG {debug_name}: Fallback attempt - "
                              f"current_pos={points[current]}, "
                              f"fallback_point={fallback_point}, "
                              f"direction_norm={direction_norm:.6e}")
                    if direction_norm < 1e-10 and len(path) >= 2:
                        direction = points[current] - points[path[-2]]
                        direction_norm = np.linalg.norm(direction)
                        if debug_name and steps == 0:
                            print(f"    DEBUG {debug_name}: Using path direction, norm={direction_norm:.6e}")
                    if direction_norm > 0:
                        dir_unit = direction / direction_norm
                        vecs = points[allowed] - points[current]
                        dots = vecs @ dir_unit
                        idx = np.argmax(dots)
                        max_dot = dots[idx]
                        if debug_name and steps == 0:
                            print(f"    DEBUG {debug_name}: Fallback dot products - "
                                  f"max_dot={max_dot:.6f}, "
                                  f"min_dot={np.min(dots):.6f}, "
                                  f"mean_dot={np.mean(dots):.6f}, "
                                  f"num_allowed={len(allowed)}, "
                                  f"dir_unit={dir_unit}")
                        if dots[idx] > 0:
                            next_node = int(allowed[idx])
                            if next_node in visited:
                                if debug_name:
                                    print(f"    DEBUG {debug_name}: Fallback selected visited node {next_node}")
                                break
                            if debug_name and steps == 0:
                                print(f"    DEBUG {debug_name}: Fallback SUCCESS - moving to node {next_node}")
                            current = next_node
                            steps += 1
                            continue
                        elif debug_name and steps == 0:
                            print(f"    DEBUG {debug_name}: Fallback FAILED - max_dot={max_dot:.6f} <= 0 "
                                  f"(all neighbors point back toward centroid)")
                    elif debug_name and steps == 0:
                        print(f"    DEBUG {debug_name}: Fallback FAILED - zero direction vector "
                              f"(current node is at or very close to fallback_point)")
                elif debug_name and steps == 0:
                    print(f"    DEBUG {debug_name}: Fallback not attempted - "
                          f"fallback_point={fallback_point is not None}, "
                          f"points={points is not None}, "
                          f"allowed={len(allowed) if allowed else 0}")
                if debug_name and steps == 0:
                    print(f"    DEBUG {debug_name}: No gradient candidates at step 0, "
                          f"current_long={long_vals[current]:.6f}, "
                          f"allowed_long_range=[{min(long_vals[n] for n in allowed):.6f}, "
                          f"{max(long_vals[n] for n in allowed):.6f}], "
                          f"fallback_used={fallback_point is not None}")
                break
            next_node = candidates[int(selector(long_vals[candidates]))]
            current = next_node
            steps += 1
        return path

    @staticmethod
    def _compute_curve_param(points, nodes):
        pts = points[nodes]
        if len(pts) < 2:
            return np.zeros(len(pts))
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        dist = np.append(0.0, np.cumsum(seg))
        total = dist[-1]
        if total <= 0:
            total = max(len(pts) - 1, 1)
            dist = np.linspace(0.0, total, len(pts))
        return dist / dist[-1]

    @staticmethod
    def _match_curve_samples(src_nodes, src_param, dst_nodes, dst_param):
        if len(dst_nodes) == 0:
            return np.array([], dtype=int)
        matched = []
        for val in src_param:
            idx = np.searchsorted(dst_param, val, side='left')
            if idx >= len(dst_param):
                idx = len(dst_param) - 1
            elif idx > 0:
                if abs(dst_param[idx - 1] - val) < abs(dst_param[idx] - val):
                    idx -= 1
            matched.append(dst_nodes[idx])
        return np.array(matched, dtype=int)

    @staticmethod
    def _shortest_path_between_nodes(adjacency, start, goal, allowed_mask, max_steps=2000):
        if start == goal:
            return [start]
        visited = set([start])
        parent = {start: None}
        queue = deque([start])
        steps = 0
        while queue and steps < max_steps:
            current = queue.popleft()
            for nbr in adjacency[current]:
                if not allowed_mask[nbr] or nbr in visited:
                    continue
                parent[nbr] = current
                if nbr == goal:
                    path = [nbr]
                    while parent[path[-1]] is not None:
                        path.append(parent[path[-1]])
                    path.append(start)
                    return list(reversed(path))
                visited.add(nbr)
                queue.append(nbr)
            steps += 1
        return [start, goal]

    def _build_ribbon_nodes(self, adjacency, pairs, allowed_mask):
        ribbon = set()
        for start, goal in pairs:
            path = self._shortest_path_between_nodes(adjacency, int(start), int(goal), allowed_mask)
            ribbon.update(path)
        return np.array(sorted(ribbon), dtype=int)

    def _compute_pv_split_plane_sets(self, uvc):
        rv_mesh = uvc.rv_mesh
        rv_bdata = uvc.rv_bdata
        rv_xyz = rv_mesh.points
        long_rv = rv_mesh.point_data.get('long')
        if long_rv is None:
            raise ValueError("RV mesh missing 'long' coordinate required for pv_circ.")

        pv_nodes, pv_centroid = self._get_patch_nodes_and_centroid(
            rv_bdata, self.bndry.get('pv'), rv_xyz, 'pv')
        tv_nodes, tv_centroid = self._get_patch_nodes_and_centroid(
            rv_bdata, self.bndry.get('tv'), rv_xyz, 'tv')

        min_long_idx = int(np.argmin(long_rv))
        min_long_val = float(long_rv[min_long_idx])
        min_long_point = rv_xyz[min_long_idx]

        line_vec = tv_centroid - pv_centroid
        line_len = np.linalg.norm(line_vec)
        if line_len < 1e-8:
            raise ValueError("PV-TV centroids are coincident; cannot define split axis.")
        line_dir = line_vec / line_len

        #plane_normal = np.cross(line_vec, min_long_point - pv_centroid)
        # Build plane normal orthogonal to both line_vec and the union PV/TV best-fit plane normal
        combined_nodes = np.unique(np.concatenate([pv_nodes, tv_nodes]))
        combined_coords = rv_xyz[combined_nodes]
        combined_centered = combined_coords - np.mean(combined_coords, axis=0)
        try:
            _, _, vh = np.linalg.svd(combined_centered, full_matrices=False)
            plane_fit_normal = vh[-1]
        except np.linalg.LinAlgError:
            print("Warning: Singular value decomposition failed for PV/TV combined plane normal. Using fallback.")
            plane_fit_normal = self._orthogonal_vector(line_dir)
        plane_fit_normal = self._normalize_vector(plane_fit_normal, 'pv/tv combined plane normal')

        plane_normal = np.cross(line_dir, plane_fit_normal)
        norm_normal = np.linalg.norm(plane_normal)
        if norm_normal < 1e-8:
            plane_normal = self._orthogonal_vector(line_dir)
            norm_normal = np.linalg.norm(plane_normal)
        plane_normal /= norm_normal

        rv_epi_nodes, _ = self._get_patch_nodes_and_centroid(
            rv_bdata, self.bndry.get('rv_epi', self.bndry.get('epi')), rv_xyz, 'rv_epi/epi')
        rv_endo_nodes, _ = self._get_patch_nodes_and_centroid(
            rv_bdata, self.bndry.get('rv_endo'), rv_xyz, 'rv_endo')
        candidate_nodes = np.unique(np.concatenate([rv_epi_nodes, rv_endo_nodes]))

        rel = rv_xyz[candidate_nodes] - pv_centroid
        projection = rel @ line_dir
        between_mask = (projection >= -1e-6) & (projection <= line_len + 1e-6)
        long_mask = long_rv[candidate_nodes] > 0.95 #may need to change this
        nodes_filtered = candidate_nodes[between_mask & long_mask]
        if len(nodes_filtered) == 0:
            raise ValueError("No RV nodes satisfy PV split-plane criteria.")

        signed_dist = (rv_xyz[nodes_filtered] - pv_centroid) @ plane_normal
        faces = rv_bdata[:, 1:-1].astype(int)
        tri_pts = rv_xyz[faces]
        edge_vecs = tri_pts[:, [1, 2, 0]] - tri_pts
        edge_lengths = np.linalg.norm(edge_vecs, axis=2)
        mean_edge_length = np.mean(edge_lengths)
        if mean_edge_length <= 0 or np.isnan(mean_edge_length):
            mean_edge_length = 1e-3
        close_mask = np.abs(signed_dist) <= mean_edge_length
        plane_nodes = nodes_filtered[close_mask]
        pos_nodes = nodes_filtered[signed_dist >= 0]
        neg_nodes = nodes_filtered[signed_dist < 0]
        if len(pos_nodes) == 0 or len(neg_nodes) == 0:
            raise ValueError("PV split-plane classification failed; both sides must have nodes.")

        pos_near_plane = nodes_filtered[(signed_dist >= 0) & close_mask]
        neg_near_plane = nodes_filtered[(signed_dist < 0) & close_mask]

        return {
            'pos_nodes': pos_nodes,
            'neg_nodes': neg_nodes,
            'plane_nodes': plane_nodes,
            'pos_near_plane': pos_near_plane,
            'neg_near_plane': neg_near_plane,
            'line_dir': line_dir,
            'plane_normal': plane_normal,
            'long_threshold': 0.5 * (1.0 + min_long_val)
        }

    def _select_surface_start_node(self, info, target_angle, surface_mask, points):
        pv_nodes = info.get('pv_nodes')
        pv_angles = info.get('angles')
        if pv_nodes is None or pv_angles is None or len(pv_nodes) == 0:
            return None
        diffs = np.abs(((pv_angles - target_angle + np.pi) % (2 * np.pi)) - np.pi)
        idx = int(np.argmin(diffs))
        candidate = pv_nodes[idx]
        if surface_mask[candidate]:
            return int(candidate)
        surface_nodes = np.where(surface_mask)[0]
        if surface_nodes.size == 0:
            return None
        target_point = points[candidate]
        nearest = surface_nodes[np.argmin(np.linalg.norm(points[surface_nodes] - target_point, axis=1))]
        return int(nearest)

    def _build_surface_gradient_curve(self, mesh, adjacency, long_vals, surface_mask, start_node,
                                      descending=True, long_tol=1e-4, fallback_point=None, debug_name=None):
        path = self._follow_surface_long_gradient_path(
            start_node, adjacency, long_vals, surface_mask,
            descending=descending, long_tol=long_tol,
            fallback_point=fallback_point, points=mesh.points, debug_name=debug_name)
        if len(path) < 2:
            if debug_name:
                print(f"    DEBUG {debug_name}: path length={len(path)}, start_node={start_node}, "
                      f"start_long={long_vals[start_node]:.6f}, "
                      f"surface_mask[start]={surface_mask[start_node] if start_node < len(surface_mask) else 'N/A'}")
                if start_node < len(adjacency):
                    neighbors = adjacency[start_node]
                    if len(neighbors) > 0:
                        nbr_on_surface = [n for n in neighbors if n < len(surface_mask) and surface_mask[n]]
                        nbr_long_vals = [long_vals[n] for n in nbr_on_surface if n < len(long_vals)]
                        print(f"    DEBUG {debug_name}: start has {len(neighbors)} neighbors, "
                              f"{len(nbr_on_surface)} on surface, "
                              f"neighbor long range: [{min(nbr_long_vals):.6f}, {max(nbr_long_vals):.6f}]" 
                              if nbr_long_vals else "no valid neighbor long values")
            return None
        nodes = np.array(path, dtype=int)
        param = self._compute_curve_param(mesh.points, nodes)
        return {'nodes': nodes, 'param': param}

    def _compute_pv_reference_frame(self, uvc):
        rv_mesh = uvc.rv_mesh
        rv_bdata = uvc.rv_bdata
        rv_xyz = rv_mesh.points

        try:
            pv_nodes, pv_centroid = self._get_patch_nodes_and_centroid(
                rv_bdata, self.bndry.get('pv'), rv_xyz, 'pv')
            tv_nodes, tv_centroid = self._get_patch_nodes_and_centroid(
                rv_bdata, self.bndry.get('tv'), rv_xyz, 'tv')
        except ValueError:
            return None

        pv_coords = rv_xyz[pv_nodes]
        centered = pv_coords - pv_centroid
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            axis = vh[-1]
        except np.linalg.LinAlgError:
            return None
        axis = self._normalize_vector(axis, 'pv axis')
        ref_vec = tv_centroid - pv_centroid
        ref_plane = ref_vec - np.dot(ref_vec, axis) * axis
        if np.linalg.norm(ref_plane) < 1e-8:
            ref_plane = self._orthogonal_vector(axis)
        ref_dir = self._normalize_vector(ref_plane, 'pv reference direction')
        ref_perp = np.cross(axis, ref_dir)
        ref_perp = self._normalize_vector(ref_perp, 'pv perpendicular direction')

        vecs = rv_xyz[pv_nodes] - pv_centroid
        axis_component = (vecs @ axis)[:, None] * axis
        vec_plane = vecs - axis_component
        x = vec_plane @ ref_dir
        y = vec_plane @ ref_perp
        theta = (np.arctan2(y, x) + 2 * np.pi) % (2 * np.pi)
        return {
            'pv_nodes': pv_nodes,
            'angles': theta,
            'pv_centroid': pv_centroid,
            'tv_centroid': tv_centroid,
            'axis': axis,
            'ref_dir': ref_dir,
            'ref_perp': ref_perp
        }

    def _compute_tv_reference_frame(self, uvc):
        rv_mesh = uvc.rv_mesh
        rv_bdata = uvc.rv_bdata
        rv_xyz = rv_mesh.points

        try:
            pv_nodes, pv_centroid = self._get_patch_nodes_and_centroid(
                rv_bdata, self.bndry.get('pv'), rv_xyz, 'pv')
            tv_nodes, tv_centroid = self._get_patch_nodes_and_centroid(
                rv_bdata, self.bndry.get('tv'), rv_xyz, 'tv')
        except ValueError:
            return None

        tv_coords = rv_xyz[tv_nodes]
        centered = tv_coords - tv_centroid
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            axis = vh[-1]
        except np.linalg.LinAlgError:
            return None
        axis = self._normalize_vector(axis, 'tv axis')
        ref_vec = pv_centroid - tv_centroid
        ref_plane = ref_vec - np.dot(ref_vec, axis) * axis
        if np.linalg.norm(ref_plane) < 1e-8:
            ref_plane = self._orthogonal_vector(axis)
        ref_dir = self._normalize_vector(ref_plane, 'tv reference direction')
        ref_perp = np.cross(axis, ref_dir)
        ref_perp = self._normalize_vector(ref_perp, 'tv perpendicular direction')

        vecs = rv_xyz[tv_nodes] - tv_centroid
        axis_component = (vecs @ axis)[:, None] * axis
        vec_plane = vecs - axis_component
        x = vec_plane @ ref_dir
        y = vec_plane @ ref_perp
        theta = (np.arctan2(y, x) + 2 * np.pi) % (2 * np.pi)
        return {
            'tv_nodes': tv_nodes,
            'angles': theta,
            'pv_centroid': pv_centroid,
            'tv_centroid': tv_centroid,
            'axis': axis,
            'ref_dir': ref_dir,
            'ref_perp': ref_perp
        }

    def _prepare_pv_curve_data(self, uvc):
        rv_mesh = uvc.rv_mesh
        rv_bdata = uvc.rv_bdata
        rv_long = rv_mesh.point_data.get('long')
        if rv_long is None:
            print("  Warning: RV mesh missing 'long' field; cannot build PV curves.")
            return None

        info = self._compute_pv_reference_frame(uvc)
        if info is None:
            print("  Warning: Unable to compute PV reference frame.")
            return None

        epi_patch = self.bndry.get('rv_epi', self.bndry.get('epi'))
        endo_patch = self.bndry.get('rv_endo')
        if epi_patch is None or endo_patch is None:
            print("  Warning: Missing RV epi/endo patches; cannot build PV curves.")
            return None

        epi_nodes = np.unique(rv_bdata[rv_bdata[:, -1] == epi_patch, 1:-1]).astype(int)
        endo_nodes = np.unique(rv_bdata[rv_bdata[:, -1] == endo_patch, 1:-1]).astype(int)
        if len(epi_nodes) == 0 or len(endo_nodes) == 0:
            print("  Warning: RV epi/endo nodes not found for PV curves.")
            return None

        npts = len(rv_mesh.points)
        epi_mask = np.zeros(npts, dtype=bool)
        epi_mask[epi_nodes] = True
        endo_mask = np.zeros(npts, dtype=bool)
        endo_mask[endo_nodes] = True
        allowed_mask = np.ones(npts, dtype=bool)

        adjacency = self._build_node_adjacency(rv_mesh)

        angle_specs = [
            ('two_thirds_pi', 2.0 * np.pi / 3.0),
            ('pi', np.pi),
            ('four_thirds_pi', 4.0 * np.pi / 3.0)
        ]

        curves = []
        for name, angle in angle_specs:
            epi_start = self._select_surface_start_node(info, angle, epi_mask, rv_mesh.points)
            if epi_start is None:
                print(f"  Warning: Could not find epi start node for PV curve {name}.")
                continue
            print(f"  Building PV curve {name}: epi_start={epi_start}, "
                  f"epi_start_long={rv_long[epi_start]:.6f}")
            epi_curve = self._build_surface_gradient_curve(
                rv_mesh, adjacency, rv_long, epi_mask, epi_start,
                descending=True, fallback_point=info['pv_centroid'], debug_name=f"{name}_epi")
            if epi_curve is None:
                print(f"  Warning: Failed to build epi gradient curve for {name}.")
                continue
            dirichlet = float(abs(1.0 - (angle / np.pi)))
            curves.append({
                'name': name,
                'angle': angle,
                'dirichlet': dirichlet,
                'epi_nodes': epi_curve['nodes']
            })

        if not curves:
            return None
        return curves

    def _build_pv_ribbons_from_epi_curves(self, uvc, curves):
        """Build ribbon nodes from epi curves based on distance and transmural projection."""
        rv_mesh = uvc.rv_mesh
        rv_bdata = uvc.rv_bdata
        rv_xyz = rv_mesh.points
        npts = len(rv_xyz)

        # Compute mean edge length from RV mesh triangles
        faces = rv_bdata[:, 1:-1].astype(int)
        tri_pts = rv_xyz[faces]
        edge_vecs = tri_pts[:, [1, 2, 0]] - tri_pts
        edge_lengths = np.linalg.norm(edge_vecs, axis=2)
        mean_edge_length = np.mean(edge_lengths)
        if mean_edge_length <= 0 or np.isnan(mean_edge_length):
            mean_edge_length = 1e-3
        print(f"  Mean edge length: {mean_edge_length:.6f}")

        # Get transmural gradient on RV mesh
        rv_trans_grad = self.get_func_gradient(uvc, 'trans', 'rv', linear=True)
        
        # Collect all epi curve points and their transmural gradients
        all_epi_points = []
        all_epi_grads = []
        all_epi_indices = []
        curve_id_lookup = []
        
        for idx, curve in enumerate(curves):
            epi_nodes = curve['epi_nodes']
            all_epi_points.append(rv_xyz[epi_nodes])
            all_epi_grads.append(rv_trans_grad[epi_nodes])
            all_epi_indices.append(epi_nodes)
            curve_id_lookup.extend([idx] * len(epi_nodes))
        
        if len(all_epi_points) == 0:
            return {'nodes': np.array([], dtype=int), 'curve_indices': np.array([], dtype=int)}
        
        # Flatten to get all epi curve points
        epi_curve_points = np.concatenate(all_epi_points, axis=0)
        epi_curve_grads = np.concatenate(all_epi_grads, axis=0)
        curve_id_lookup = np.array(curve_id_lookup, dtype=int)
        
        # Normalize transmural gradients
        grad_norms = np.linalg.norm(epi_curve_grads, axis=1)
        grad_norms[grad_norms < 1e-10] = 1.0  # Avoid division by zero
        epi_curve_grads_norm = epi_curve_grads / grad_norms[:, None]
        
        # Build KDTree for fast nearest neighbor search
        from scipy.spatial import KDTree
        epi_tree = KDTree(epi_curve_points)
        
        # For each RV point, check if it qualifies for ribbon
        ribbon_nodes = []
        ribbon_curve_idx = []
        distance_threshold = 5.0 * mean_edge_length
        perp_threshold = 0.5 * mean_edge_length
        
        # Process in batches to avoid memory issues
        batch_size = 10000
        for i in range(0, npts, batch_size):
            batch_end = min(i + batch_size, npts)
            batch_points = rv_xyz[i:batch_end]
            
            # Find nearest epi curve point for each batch point
            dists, nn_idx = epi_tree.query(batch_points, k=1)
            dists = np.atleast_1d(dists)
            nn_idx = np.atleast_1d(nn_idx).flatten()
            
            # Check distance threshold
            within_dist = dists < distance_threshold
            
            if np.any(within_dist):
                # For points within distance, compute perpendicular distance
                valid_mask = within_dist
                valid_batch_idx = np.where(valid_mask)[0]
                valid_global_idx = i + valid_batch_idx
                valid_nn_idx = nn_idx[valid_batch_idx]
                
                # Get corresponding epi curve point and gradient
                epi_pts = epi_curve_points[valid_nn_idx]
                epi_grads = epi_curve_grads_norm[valid_nn_idx]
                cand_pts = batch_points[valid_batch_idx]
                
                # Compute vector from epi point to candidate point
                vecs = cand_pts - epi_pts
                
                # Project onto transmural gradient direction
                proj_lengths = np.sum(vecs * epi_grads, axis=1)
                proj_vecs = proj_lengths[:, None] * epi_grads
                
                # Compute perpendicular component
                perp_vecs = vecs - proj_vecs
                perp_dists = np.linalg.norm(perp_vecs, axis=1)
                
                # Check perpendicular distance threshold
                within_perp = perp_dists < perp_threshold
                ribbon_nodes.extend(valid_global_idx[within_perp])
                if np.any(within_perp):
                    nn_curve_idx = curve_id_lookup[valid_nn_idx]
                    ribbon_curve_idx.extend(nn_curve_idx[within_perp])
        
        ribbon_nodes = np.array(ribbon_nodes, dtype=int)
        ribbon_curve_idx = np.array(ribbon_curve_idx, dtype=int)
        print(f"  Ribbon nodes: {len(ribbon_nodes)}")
        return {'nodes': ribbon_nodes, 'curve_indices': ribbon_curve_idx}

    def run_fast_longitudinal(self, uvc, method='laplace'):
        bcs_point = {}
        for i in range(len(uvc.lv_zero_nodes)):
            bcs_point[tuple(uvc.lv_mesh.points[uvc.lv_zero_nodes[i]])] = 0
        # Use AV if it exists, otherwise MV (already included)
        face_bcs = {self.bndry['mv']: 1.0,
                   self.bndry['pv']: 1.0, self.bndry['tv']: 1.0}
        if uvc.has_av:
            face_bcs[self.bndry['av']] = 1.0
        bcs_marker = {'point': bcs_point, 'face': face_bcs}
        self.init_lv_mesh(uvc)
        lv_solver = self.get_solver(method, 'lv')
        lv_long = lv_solver.solve(bcs_marker)
        lv_long = lv_long.x.petsc_vec.array[self.lv_corr]
        uvc.lv_mesh.point_data['long'] = lv_long

        # Correct
        bdata = uvc.lv_bdata_og
        epi_nodes = np.unique(bdata[bdata[:,-1]==self.bndry['lv_epi'],1:-1])
        rvlv_ant_nodes = np.unique(bdata[bdata[:,-1]==self.bndry['rvlv_ant'],1:-1])
        rvlv_post_nodes = np.unique(bdata[bdata[:,-1]==self.bndry['rvlv_post'],1:-1])
        epi_nodes = np.union1d(epi_nodes, rvlv_ant_nodes)
        epi_nodes = np.union1d(epi_nodes, rvlv_post_nodes)
        lat_nodes = np.unique(bdata[bdata[:,-1]==self.bndry['lat0'],1:-1])
        lat_nodes = np.union1d(lat_nodes, uvc.lv_zero_nodes)

        long_nodes = np.intersect1d(epi_nodes, lat_nodes)

        # Compute cartesian distance from node to node and correct long
        order = np.argsort(lv_long[long_nodes])
        points_long = uvc.lv_mesh.points[long_nodes[order]]
        dist = np.linalg.norm(np.diff(points_long, axis=0), axis=1)
        dist = np.append(0, np.cumsum(dist))
        norm_dist = dist/dist[-1]

        norm_func = interp1d(lv_long[long_nodes[order]], norm_dist)
        norm_long = norm_func(lv_long)
        uvc.lv_mesh.point_data['long'] = norm_long[:len(uvc.lv_mesh.points)]

        return lv_long, None



    def run_transmural(self, uvc, method='laplace', which='all'):
        ret = []
        if which == 'all' or which =='lv':
            # Run LV transmural problem
            self.init_lv_mesh(uvc, og=True)
            lv_solver = self.get_solver(method, 'lv')
            bcs_lv = {'face': {self.bndry['lv_endo']: 0,
                      self.bndry['rv_septum']: 1,
                      self.bndry['rv_endo']: 1}}
            if 'rv_lv_junction' in self.bndry:
                bcs_lv['face'][self.bndry['rv_lv_junction']] = 1
            else:
                bcs_lv['face'][self.bndry['rvlv_ant']] = 1
                bcs_lv['face'][self.bndry['rvlv_post']] = 1

            if 'lv_apex' in self.bndry:
                bcs_lv['face'][self.bndry['lv_apex']] = 0
            if uvc.split_epi:
                bcs_lv['face'][self.bndry['lv_epi']] = 1
            else:
                bcs_lv['face'][self.bndry['epi']] = 1

            trans_lv = lv_solver.solve(bcs_lv)
            trans_lv = trans_lv.x.petsc_vec.array[self.lv_corr]
            uvc.lv_mesh.point_data['trans'] = trans_lv

            ret += [trans_lv]

        if which == 'all' or which =='rv':
            # Run RV transmural problem
            self.init_rv_mesh(uvc, og=True)
            rv_solver = self.get_solver(method, 'rv')
            bcs_rv = {'face': {self.bndry['rv_endo']: 0}}
            if 'rv_apex' in self.bndry:
                bcs_rv['face'][self.bndry['rv_apex']] = 0
            if uvc.split_epi:
                bcs_rv['face'][self.bndry['rv_epi']] = 1
            else:
                bcs_rv['face'][self.bndry['epi']] = 1
            trans_rv = rv_solver.solve(bcs_rv)
            trans_rv = trans_rv.x.petsc_vec.array[self.rv_corr]
            uvc.rv_mesh.point_data['trans'] = trans_rv

            ret += [trans_rv]

        if which == 'all':
            # Run auxiliar problem for epi trans
            self.init_bv_mesh(uvc, og=True)
            bv_solver = self.get_solver(method, 'bv')

            bcs_bv = {'face': {self.bndry['lv_epi']: 0.0,
                                    self.bndry['rv_epi']: 0.0,
                                    self.bndry['lv_endo']: 1.0,
                                    self.bndry['rv_endo']: -1.0,
                                    self.bndry['rv_septum']: -1.0}}

            trans_bv = bv_solver.solve(bcs_bv)
            trans_bv = trans_bv.x.petsc_vec.array[self.bv_corr]
            uvc.bv_mesh.point_data['epitrans'] = trans_bv

            ret += [trans_bv]

        return ret

    @staticmethod
    def point_to_face_marker(point_marker, new_value, bdata=None, new_faces=False, mesh=None):

        if not new_faces:
            mark = np.isin(bdata[:, 1:-1], point_marker)
            face_marker = np.where(np.all(mark, axis=1))
            if bdata is None:
                raise ('if not adding faces, you need to provide bdata')
            bdata[face_marker, -1] = new_value
        else:
            if mesh is None:
                raise ('if new faces you need to pass the mesh')

            ien = mesh.cells[0].data
            arr = np.array([[0,1,2],[1,2,3],[2,3,0],[0,3,1]])
            tet_elem = np.repeat(np.arange(len(ien)), 4)
            faces = np.vstack(ien[:,arr])

            mark = np.isin(faces, point_marker)
            bfaces = np.where(np.sum(mark, axis=1) == 3)[0]

            new_faces = faces[bfaces]

            if bdata is None:
                new_bdata = np.vstack([tet_elem[bfaces],
                                       new_faces.T,
                                       np.ones(len(bfaces), dtype=int)*new_value]).T
                bdata = new_bdata
            else:
                # If bdata exists I always want to keep the old bdata
                og_faces = bdata[:,1:-1]
                new_og_faces = np.vstack([og_faces, new_faces])
                sort_new_og_faces = np.sort(new_og_faces, axis=1)
                new_og_elems = np.append(bdata[:,0], tet_elem[bfaces])
                sort_new_og_faces = np.vstack([new_og_elems, sort_new_og_faces.T]).T
                un, inv = np.unique(sort_new_og_faces,
                                    axis=0, return_inverse=True)

                ind = inv[inv>len(bdata)-1]
                new_bdata = np.hstack([new_og_elems[ind][:, None], new_og_faces[ind],
                                        np.ones([len(ind), 1], dtype=int)*new_value])
                bdata = np.vstack([bdata, new_bdata])

        return bdata

    def run_lv_circumferential1(self, uvc, method):
        uvc.create_lv_circ_bc1()

        # First, we need to define the point bc as face bc
        bc_point = uvc.lv_mesh.point_data['bc1']
        ant_marker = np.where(bc_point == 1)[0]
        pos_marker = np.where(bc_point == -1)[0]

        self.bndry['rv_lv_ant'] = 21
        self.bndry['rv_lv_post'] = 22
        uvc.lv_bdata = self.point_to_face_marker(
            ant_marker, self.bndry['rv_lv_ant'], uvc.lv_bdata)
        uvc.lv_bdata = self.point_to_face_marker(
            pos_marker, self.bndry['rv_lv_post'], uvc.lv_bdata)

        self.init_rv_mesh(uvc)
        self.init_lv_mesh(uvc)

        # Run LV circumferential problem
        bcs_lv = {'face': {self.bndry['rv_lv_ant']: 1,
                  self.bndry['rv_lv_post']: -1}}
        lv_solver = self.get_solver(method, 'lv')
        lv_circ1 = lv_solver.solve(bcs_lv)
        lv_circ1 = lv_circ1.x.petsc_vec.array[self.lv_corr]
        uvc.lv_mesh.point_data['lv_circ1'] = lv_circ1

        return lv_circ1

    def run_lv_circumferential2(self, lv_circ1, uvc, method):
        if self.mmg:
            mesh, bc = uc.mmg_create_lv_circ_bc2(lv_circ1, uvc)
            sep_nodes = np.where(bc == 1)[0]
            mpi_nodes = np.where(bc == 2)[0]
            bdata = self.point_to_face_marker(
                sep_nodes, 1, new_faces=True, mesh=mesh)
            bdata = self.point_to_face_marker(
                mpi_nodes, 2, bdata, new_faces=True, mesh=mesh)

            # Need to figure out a better way to map...
            circ = run_coord(mesh, bdata, {'face': {1: 0, 2: 1}})

            # mesh.point_data['circ'] = circ
            # io.write('check.vtu', mesh)
            lv_circ2 = circ[:len(uvc.lv_mesh.points)]


        else:
            uvc.create_lv_circ_bc2(lv_circ1)

            bc_point = uvc.lv_mesh.point_data['bc2']
            lat_marker_neg = uvc.lv_bc_mpi
            sep_marker_neg = uvc.lv_bc_0n
            lat_marker_pls = uvc.lv_bc_pi
            sep_marker_pls = uvc.lv_bc_0p

            self.bndry['lat_pi'] = 23
            self.bndry['lat_mpi'] = 24
            self.bndry['sep_0n'] = 25
            self.bndry['sep_0p'] = 26
            uvc.lv_bdata = self.point_to_face_marker(lat_marker_pls,
                                                     self.bndry['lat_pi'], uvc.lv_bdata, new_faces=True, mesh=uvc.lv_mesh)
            uvc.lv_bdata = self.point_to_face_marker(lat_marker_neg,
                                                     self.bndry['lat_mpi'], uvc.lv_bdata, new_faces=True, mesh=uvc.lv_mesh)
            uvc.lv_bdata = self.point_to_face_marker(sep_marker_pls,
                                                     self.bndry['sep_0p'], uvc.lv_bdata, new_faces=True, mesh=uvc.lv_mesh)
            uvc.lv_bdata = self.point_to_face_marker(sep_marker_neg,
                                                     self.bndry['sep_0n'], uvc.lv_bdata, new_faces=True, mesh=uvc.lv_mesh)

            self.init_rv_mesh(uvc)
            self.init_lv_mesh(uvc)

            bcs_lv = {'function': {self.bndry['lat_pi']: np.ones(len(bc_point)),
                      self.bndry['lat_mpi']: np.ones(len(bc_point)),
                      self.bndry['sep_0n']: np.zeros(len(bc_point))}}
            lv_solver = self.get_solver(method, 'lv')
            lv_circ2 = lv_solver.solve(bcs_lv)
            lv_circ2_1 = lv_circ2.x.petsc_vec.array[self.lv_corr]

            bcs_lv = {'function': {self.bndry['lat_pi']: np.ones(len(bc_point)),
                      self.bndry['lat_mpi']: np.ones(len(bc_point)),
                      self.bndry['sep_0p']: np.zeros(len(bc_point))}}
            lv_solver = self.get_solver(method, 'lv')
            lv_circ2 = lv_solver.solve(bcs_lv)
            lv_circ2_2 = lv_circ2.x.petsc_vec.array[self.lv_corr]

            lv_circ2 = (lv_circ2_1 + lv_circ2_2)/2
            lv_circ2 = uvc.correct_lv_circ2(lv_circ2)

        uvc.lv_mesh.point_data['lv_circ2'] = lv_circ2

        return lv_circ2

    def run_lv_circumferential3(self, lv_circ2, uvc, method):
        uvc.create_lv_circ_bc3(lv_circ2)
        bc_point = uvc.lv_mesh.point_data['bc3']
        sep_marker = np.where(bc_point == 1)[0]
        self.bndry['lv_septum'] = 27
        uvc.lv_bdata = self.point_to_face_marker(
            sep_marker, self.bndry['lv_septum'], uvc.lv_bdata)

        self.init_rv_mesh(uvc)
        self.init_lv_mesh(uvc)
        lv_solver = self.get_solver(method, 'lv')
        bcs_lv = {'face': {self.bndry['lv_septum']: 1,
                  self.bndry['rv_septum']: 1,
                  self.bndry['rv_endo']: 1}}

        if uvc.split_epi:
            bcs_lv['face'][self.bndry['lv_epi']] = -1
            bcs_lv['face'][self.bndry['rv_epi']] = -1
        else:
            bcs_lv['face'][self.bndry['epi']] = -1

        lv_circ3 = lv_solver.solve(bcs_lv)
        lv_circ3 = lv_circ3.x.petsc_vec.array[self.lv_corr]
        uvc.lv_mesh.point_data['lv_circ3'] = lv_circ3

        return lv_circ3

    def run_lv_circumferential4(self, lv_circ3, uvc, method, correct=True):
        if self.mmg:
            mesh, bc, mmg_bdata = uc.mmg_create_lv_circ_bc4(lv_circ3, uvc)
            sep_nodes = np.where(bc == 1)[0]
            mpi_nodes = np.where(bc == 2)[0]
            ant_nodes = np.where(bc == 4)[0]
            post_nodes = np.where(bc == 5)[0]
            zero_nodes = np.where(bc == 10)[0]

            bdata = self.point_to_face_marker(
                sep_nodes, 1, new_faces=True, mesh=mesh)
            bdata = self.point_to_face_marker(
                mpi_nodes, 2, bdata, new_faces=True, mesh=mesh)
            bdata = self.point_to_face_marker(
                ant_nodes, 4, bdata, new_faces=True, mesh=mesh)
            bdata = self.point_to_face_marker(
                post_nodes, 5, bdata, new_faces=True, mesh=mesh)

            bcs_point = {}
            for i in range(len(zero_nodes)):
                bcs_point[tuple(mesh.points[zero_nodes[i]])] = 0

            bcs_lv = {'face': {1: 0, 2: 1, 4: 1/3, 5: 1/3},
                      'point': bcs_point}

            circ = run_coord(mesh, bdata, bcs_lv)
            lv_circ4 = np.abs(circ[:len(uvc.lv_mesh.points)])
            uvc.lv_mesh.point_data['circ_aux'] = lv_circ4
            lv_circ4 = uvc.correct_lv_circ4(lv_circ4)*np.pi

            # Save MMG mesh
            self.lv_mmg_mesh = mesh
            self.lv_mmg_bdata = mmg_bdata
            self.lv_mmg_bc = bc

        else:
            # Last problem
            uvc.create_lv_circ_bc4(lv_circ3)
            bc_point = uvc.lv_mesh.point_data['bc4_n']
            ant_marker = np.where(bc_point == 1)[0]
            pos_marker = np.where(bc_point == -1)[0]
            self.bndry['circ_ant'] = 28
            self.bndry['circ_post'] = 29
            uvc.lv_bdata = self.point_to_face_marker(ant_marker,  self.bndry['circ_ant'],
                                                     uvc.lv_bdata, new_faces=True, mesh=uvc.lv_mesh)
            uvc.lv_bdata = self.point_to_face_marker(pos_marker, self.bndry['circ_post'],
                                                     uvc.lv_bdata, new_faces=True, mesh=uvc.lv_mesh)

            self.init_lv_mesh(uvc)
            lv_solver = self.get_solver(method, 'lv')
            bcs_lv = {'face': {self.bndry['circ_ant']: 1,
                      self.bndry['circ_post']: -1}}
            lv_circ4 = lv_solver.solve(bcs_lv)
            lv_circ4_1 = lv_circ4.x.petsc_vec.array[self.lv_corr]

            bc_point = uvc.lv_mesh.point_data['bc4_p']
            ant_marker = np.where(bc_point == 1)[0]
            pos_marker = np.where(bc_point == -1)[0]
            self.bndry['circ_ant'] = 28
            self.bndry['circ_post'] = 29
            uvc.lv_bdata = self.point_to_face_marker(ant_marker, self.bndry['circ_ant'],
                                                     uvc.lv_bdata, new_faces=True, mesh=uvc.lv_mesh)
            uvc.lv_bdata = self.point_to_face_marker(pos_marker, self.bndry['circ_post'],
                                                     uvc.lv_bdata, new_faces=True, mesh=uvc.lv_mesh)

            self.init_lv_mesh(uvc)
            lv_solver = self.get_solver(method, 'lv')
            bcs_lv = {'face': {self.bndry['circ_ant']: 1,
                      self.bndry['circ_post']: -1}}
            lv_circ4 = lv_solver.solve(bcs_lv)
            lv_circ4_2 = lv_circ4.x.petsc_vec.array[self.lv_corr]

            lv_circ4 = (lv_circ4_1 + lv_circ4_2)/2
            uvc.create_lv_circ_bc5(lv_circ4)
            lv_circ4 = uvc.correct_lv_circ4(lv_circ4)*np.pi

        uvc.lv_mesh.point_data['lv_circ4'] = lv_circ4

        return lv_circ4

    def run_lv_circumferential5(self, lv_circ4, uvc, method):

        if self.mmg:

            circ_aux = np.zeros(len(uvc.lv_mesh.points))
            lv_circ5 = np.zeros(len(uvc.lv_mesh.points))
            sep_mesh, sep_bdata, sep_map, lat_mesh, lat_bdata, lat_map = uc.mmg_create_lv_circ_bc5(lv_circ4, uvc)

            if method == 'trajectory':
                self.get_local_vectors(uvc)
                eC = uvc.lv_mesh.cell_data['eC'][0]

                # sep_eC
                sep_eC = eC[sep_map[2]]
                sep_circ, corr = run_trajectory_coord(sep_mesh, sep_bdata, {'face': {1: 1, 2: 0}}, vector=sep_eC)
                sep_circ = sep_circ[corr]

                lat_eC = eC[lat_map[2]]
                lat_circ, corr = run_trajectory_coord(lat_mesh, lat_bdata, {'face': {1: 1, 2: 0}}, vector=lat_eC)
                lat_circ = lat_circ[corr]

            else:
                sep_circ = run_coord(sep_mesh, sep_bdata, {'face': {1: 1, 2: 0}})
                lat_circ = run_coord(lat_mesh, lat_bdata, {'face': {1: 1, 2: 0}})

            circ_aux[sep_map[0]] = sep_circ[sep_map[1]]
            circ_aux[lat_map[0]] = lat_circ[lat_map[1]]
            circ_aux = (circ_aux*2-1)
            uvc.lv_mesh.point_data['circ_aux'] = circ_aux

            sep_circ = (sep_circ*2-1)/3
            lat_circ_aux = lat_circ*2-1
            lat_circ = (2/3-np.abs(lat_circ_aux*2/3)) + 1/3
            lat_circ = lat_circ*np.sign(lat_circ_aux)
            lv_circ5[sep_map[0]] = sep_circ[sep_map[1]]
            lv_circ5[lat_map[0]] = lat_circ[lat_map[1]]

            lv_circ5 *= np.pi
            uvc.lv_mesh.point_data['lv_circ5'] = lv_circ5

        else:
            return lv_circ4

        return lv_circ5

    def run_rv_circumferential1(self, uvc, method):
        uvc.create_rv_circ_bc()

        # First, we need to define the point bc as face bc
        bc_point = uvc.rv_mesh.point_data['bc1']
        ant_marker = np.where(bc_point == 1)[0]
        pos_marker = np.where(bc_point == -1)[0]

        uvc.rv_bdata = self.point_to_face_marker(
            ant_marker, self.bndry['rv_lv_ant'], uvc.rv_bdata)
        uvc.rv_bdata = self.point_to_face_marker(
            pos_marker, self.bndry['rv_lv_post'], uvc.rv_bdata)

        self.init_rv_mesh(uvc)

        # Build Dirichlet BCs on RV nodes by copying LV circ values at the junction
        lv_circ = uvc.lv_mesh.point_data.get('circ')
        if lv_circ is None:
            raise ValueError("LV circumferential field ('circ') not available when solving RV circ.")

        junction_patch = self.bndry.get('rv_lv_junction')
        ant_patch = self.bndry.get('rv_lv_ant')
        post_patch = self.bndry.get('rv_lv_post')
        print(f"junction_patch: {junction_patch}")
        if junction_patch is None:
            raise ValueError("rv_lv_junction patch ID not defined for RV circumferential BCs.")

        rv_lv_junction_faces = uvc.lv_bdata[uvc.lv_bdata[:, -1] == junction_patch]
        rv_lv_ant_faces = uvc.lv_bdata[uvc.lv_bdata[:, -1] == ant_patch]
        rv_lv_post_faces = uvc.lv_bdata[uvc.lv_bdata[:, -1] == post_patch]
        print(f"number of rv_junction_faces: {len(rv_lv_junction_faces)}")
        print(f"number of rv_lv_ant_faces: {len(rv_lv_ant_faces)}")
        print(f"number of rv_lv_post_faces: {len(rv_lv_post_faces)}")
        rv_lv_junction_nodes = np.unique(rv_lv_junction_faces[:, 1:-1]).astype(int)
        rv_lv_ant_nodes = np.unique(rv_lv_ant_faces[:, 1:-1]).astype(int)
        rv_lv_post_nodes = np.unique(rv_lv_post_faces[:, 1:-1]).astype(int)
        print(f"number of rv_lv_junction_nodes: {len(rv_lv_junction_nodes)}")
        print(f"number of rv_lv_ant_nodes: {len(rv_lv_ant_nodes)}")
        print(f"number of rv_lv_post_nodes: {len(rv_lv_post_nodes)}")

        combined_lv_nodes = np.unique(
            np.concatenate([rv_lv_junction_nodes, rv_lv_ant_nodes, rv_lv_post_nodes])
        )

        bcs_point = {}
        for lv_node in combined_lv_nodes:
            bv_node = uvc.map_lv_bv[lv_node]
            rv_node = uvc.map_bv_rv[bv_node]
            #print(f"lv_node: {lv_node}, bv_node: {bv_node}, rv_node: {rv_node}")
            if lv_node >= 0:
                bcs_point[tuple(uvc.rv_mesh.points[rv_node])] = lv_circ[lv_node]

        rv_solver = self.get_solver(method, 'rv')
        bcs_rv = {'point': bcs_point,
                  'face': {self.bndry['tv']: -1,
                           self.bndry['pv']: 1}}
        rv_circ1 = rv_solver.solve(bcs_rv)
        rv_circ1 = rv_circ1.x.petsc_vec.array[self.rv_corr]
        uvc.rv_mesh.point_data['rv_circ1'] = rv_circ1
        uvc.rv_mesh.point_data['circ'] = rv_circ1

        return rv_circ1

    def run_pv_circumferential(self, uvc, method):
        self.init_rv_mesh(uvc)
        pv_circ_base = self.run_pv_circ_base(uvc, method)
        pv_circ_sign = self.run_pv_circ_sign(uvc, method)
        sign_mask = np.sign(pv_circ_sign)
        sign_mask[sign_mask == 0.0] = 1.0
        pv_circ = pv_circ_base * sign_mask * np.pi

        uvc.rv_mesh.point_data['pv_circ_base'] = pv_circ_base
        uvc.rv_mesh.point_data['pv_circ_sign'] = pv_circ_sign
        uvc.rv_mesh.point_data['pv_circ'] = pv_circ

        return pv_circ_base, pv_circ_sign, pv_circ

    def run_pv_circ_base(self, uvc, method):
        rv_mesh = uvc.rv_mesh
        rv_bdata = uvc.rv_bdata
        rv_xyz = rv_mesh.points

        custom_curves = None
        split_sets = None
        bcs_point = {}

        try:
            custom_curves = self._prepare_pv_curve_data(uvc)
            if custom_curves:
                split_sets = self._compute_pv_split_plane_sets(uvc)
        except ValueError as exc:
            print(f"  Warning: PV curve preparation failed ({exc}); using fallback BCs.")
            custom_curves = None
            split_sets = None

        if custom_curves:
            uvc.pv_curve_data = custom_curves
            uvc.pv_split_sets = split_sets
            
            # Build ribbons from epi curves
            ribbon_data = self._build_pv_ribbons_from_epi_curves(uvc, custom_curves)
            ribbon_nodes = ribbon_data['nodes']
            ribbon_curve_idx = ribbon_data['curve_indices']
            uvc.pv_ribbon_nodes = ribbon_nodes  # Store for use in run_pv_circ_sign
            uvc.pv_ribbon_curve_idx = ribbon_curve_idx
            
            npts = len(rv_xyz)
            
            # Combined field for all epi curves (1 if on any curve, 0 otherwise)
            epi_curve_combined = np.zeros(npts, dtype=float)
            
            for curve in custom_curves:
                # Mark epi curve nodes
                epi_curve_combined[curve['epi_nodes']] = 1.0
            
            if len(ribbon_nodes) > 0:
                for nid, curve_idx in zip(ribbon_nodes, ribbon_curve_idx):
                    value = custom_curves[int(curve_idx)]['dirichlet']
                    bcs_point[tuple(rv_xyz[int(nid)])] = float(value)

            # Combined field for all ribbons (1 if in any ribbon, 0 otherwise)
            ribbon_combined = np.zeros(npts, dtype=float)
            ribbon_combined[ribbon_nodes] = 1.0
            
            # Save combined fields
            rv_mesh.point_data['pv_curve_epi'] = epi_curve_combined
            rv_mesh.point_data['pv_curve_ribbon'] = ribbon_combined
            
            print(f"  PV curves: {sum(len(c['epi_nodes']) for c in custom_curves)} total epi nodes, "
                  f"{len(ribbon_nodes)} ribbon nodes")

            if split_sets and len(split_sets['plane_nodes']) > 0:
                for nid in split_sets['plane_nodes']:
                    bcs_point[tuple(rv_xyz[nid])] = 1.0
                split_mask = np.zeros(npts, dtype=float)
                split_mask[split_sets['plane_nodes']] = 1.0
                rv_mesh.point_data['pv_circ_split_plane'] = split_mask
        else:
            uvc.pv_curve_data = None
            uvc.pv_split_sets = None
            uvc.pv_ribbon_nodes = None
            uvc.pv_ribbon_curve_idx = None
            pv_nodes, pv_centroid = self._get_patch_nodes_and_centroid(
                rv_bdata, self.bndry.get('pv'), rv_xyz, 'pv')
            tv_nodes, tv_centroid = self._get_patch_nodes_and_centroid(
                rv_bdata, self.bndry.get('tv'), rv_xyz, 'tv')

            pv_coords = rv_xyz[pv_nodes]
            centered = pv_coords - pv_centroid
            try:
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
                axis = vh[-1]
            except np.linalg.LinAlgError as exc:
                raise ValueError("Unable to compute PV axis (SVD failed).") from exc
            axis = self._normalize_vector(axis, 'pv axis')
            ref_vec = tv_centroid - pv_centroid
            ref_plane = ref_vec - np.dot(ref_vec, axis) * axis
            if np.linalg.norm(ref_plane) < 1e-8:
                ref_plane = self._orthogonal_vector(axis)
            ref_dir = self._normalize_vector(ref_plane, 'pv reference direction')
            ref_perp = np.cross(axis, ref_dir)
            ref_perp = self._normalize_vector(ref_perp, 'pv perpendicular direction')

            vecs = rv_xyz[pv_nodes] - pv_centroid
            axis_component = (vecs @ axis)[:, None] * axis
            vec_plane = vecs - axis_component
            x = vec_plane @ ref_dir
            y = vec_plane @ ref_perp
            theta = (np.arctan2(y, x) + 2 * np.pi) % (2 * np.pi)
            dirichlet_vals = abs(1 - (theta / np.pi))

            for node, value in zip(pv_nodes, dirichlet_vals):
                bcs_point[tuple(rv_xyz[node])] = float(value)

        rv_solver = self.get_solver(method, 'rv')
        sol = rv_solver.solve({'point': bcs_point})
        pv_base = sol.x.petsc_vec.array[self.rv_corr]
        return pv_base

    def run_pv_circ_sign(self, uvc, method):
        rv_mesh = uvc.rv_mesh
        rv_xyz = rv_mesh.points

        split_sets = getattr(uvc, 'pv_split_sets', None)
        if split_sets is None:
            split_sets = self._compute_pv_split_plane_sets(uvc)
        pos_nodes = split_sets['pos_near_plane']
        neg_nodes = split_sets['neg_near_plane']
        if pos_nodes is None or len(pos_nodes) == 0:
            pos_nodes = split_sets['pos_nodes']
        if neg_nodes is None or len(neg_nodes) == 0:
            neg_nodes = split_sets['neg_nodes']

        bcs_point = {}
        for node in pos_nodes:
            bcs_point[tuple(rv_xyz[node])] = 1.0
        for node in neg_nodes:
            bcs_point[tuple(rv_xyz[node])] = -1.0

        # Add zero BC only for ribbon nodes belonging to the pi-angle curve
        curve_data = getattr(uvc, 'pv_curve_data', None)
        ribbon_nodes = getattr(uvc, 'pv_ribbon_nodes', None)
        ribbon_curve_idx = getattr(uvc, 'pv_ribbon_curve_idx', None)
        if curve_data is not None and ribbon_nodes is not None and ribbon_curve_idx is not None:
            pi_indices = [idx for idx, curve in enumerate(curve_data) if abs(curve['angle'] - np.pi) < 1e-6]
            if pi_indices:
                pi_idx = pi_indices[0]
                pi_mask = ribbon_curve_idx == pi_idx
                for node in ribbon_nodes[pi_mask]:
                    bcs_point[tuple(rv_xyz[int(node)])] = 0.0

        rv_solver = self.get_solver(method, 'rv')
        sol = rv_solver.solve({'point': bcs_point})
        pv_sign_field = sol.x.petsc_vec.array[self.rv_corr]
        return pv_sign_field

    @staticmethod
    def _normalize_vector(vec, name):
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            raise ValueError(f"Cannot normalize zero vector while computing {name}.")
        return vec / norm

    @staticmethod
    def _orthogonal_vector(vec):
        ref = np.array([1.0, 0.0, 0.0])
        if np.allclose(np.abs(np.dot(vec, ref)), np.linalg.norm(vec), atol=1e-8):
            ref = np.array([0.0, 1.0, 0.0])
        ortho = np.cross(vec, ref)
        norm = np.linalg.norm(ortho)
        if norm < 1e-10:
            ref = np.array([0.0, 0.0, 1.0])
            ortho = np.cross(vec, ref)
            norm = np.linalg.norm(ortho)
        return ortho / norm

    @staticmethod
    def _pv_dirichlet_from_angle(theta):
        theta_norm = np.mod(theta, 2 * np.pi)
        values = np.zeros_like(theta_norm)
        half_pi = np.pi / 2

        first_mask = theta_norm <= np.pi
        theta_first = theta_norm[first_mask]
        values[first_mask] = np.clip(1.0 - np.abs(theta_first - half_pi) / half_pi, 0.0, 1.0)

        theta_second = theta_norm[~first_mask]
        values[~first_mask] = np.clip((theta_second - np.pi) / np.pi, 0.0, 1.0)
        return values

    @staticmethod
    def _get_patch_nodes_and_centroid(bdata, patch_id, coords, patch_name):
        if patch_id is None:
            raise ValueError(f"Patch '{patch_name}' is not defined in boundaries.")
        faces = bdata[bdata[:, -1] == patch_id]
        if len(faces) == 0:
            raise ValueError(f"No faces found for patch '{patch_name}'.")
        nodes = np.unique(faces[:, 1:-1]).astype(int)
        centroid = np.mean(coords[nodes], axis=0)
        return nodes, centroid

    def run_rv_circumferential2(self, rv_circ1, uvc, method):

        if self.mmg:
            mesh, bc, mmg_bdata = uc.mmg_create_rv_circ_bc2(rv_circ1, uvc)

            ant_nodes = np.where(bc == 1)[0]
            post_nodes = np.where(bc == 2)[0]
            zero_nodes = np.where(bc == 3)[0]
            bdata = self.point_to_face_marker(
                ant_nodes, 1, new_faces=True, mesh=mesh)
            bdata = self.point_to_face_marker(
                post_nodes, 2, bdata, new_faces=True, mesh=mesh)
            bdata = self.point_to_face_marker(
                zero_nodes, 3, bdata, new_faces=True, mesh=mesh)

            # Need to figure out a better way to map...
            circ = run_coord(mesh, bdata, {'face': {1: -1, 2: 1, 3: 0}})
            rv_circ2 = circ[:len(uvc.rv_mesh.points)]
            uvc.rv_mesh.point_data['rv_circ2'] = rv_circ2

            self.rv_mmg_mesh = mesh
            self.rv_mmg_bdata = mmg_bdata
            self.rv_bc = bc


        else:
            raise('Not implemented')

        return rv_circ2


    def run_circumferential(self, uvc, method='laplace'):
        # PV circumferential (run first for debugging)
        print('Computing PV Circumferential')
        pv_circ_base, pv_circ_sign, pv_circ = self.run_pv_circumferential(uvc, method)

        # LV side
        print('Computing LV Circumferential')
        lv_circ1 = self.run_lv_circumferential1(uvc, 'laplace')

        # For the second problem we define surfaces at 0 and pi
        lv_circ2 = self.run_lv_circumferential2(lv_circ1, uvc, 'laplace')

        # # Third problem
        lv_circ3 = self.run_lv_circumferential3(lv_circ2, uvc, 'laplace')

        # # Fourth problem
        lv_circ = self.run_lv_circumferential4(lv_circ3, uvc, 'laplace')

        # # Fifth problem
        # lv_circ = self.run_lv_circumferential5(lv_circ4, uvc, method)
        uvc.lv_mesh.point_data['circ'] = lv_circ

        # # RV side
        print('Computing RV Circumferential')
        rv_circ1 = self.run_rv_circumferential1(uvc, method)

        return lv_circ, rv_circ1, pv_circ

    def run_rv_lateral(self, uvc, method='laplace'):
        """
        Compute rv_lat coordinate by solving Laplace equation on RV mesh.
        Boundary conditions:
        - TV and PV faces closest to MV centroid: 0
        - rv_lv_junction: 0
        - TV and PV faces farthest from MV centroid: +1
        """
        # Get MV centroid
        if 'mv' not in uvc.valve_centroids:
            raise ValueError("MV (mitral valve) centroid not found. Cannot compute rv_lat.")
        mv_centroid = uvc.valve_centroids['mv']
        
        # Get RV mesh coordinates
        rv_xyz = uvc.rv_mesh.points
        
        # Get TV and PV faces from RV boundary data
        # bdata format: [element_idx, node0, node1, node2, patch_id]
        tv_faces = uvc.rv_bdata[uvc.rv_bdata[:, -1] == self.bndry['tv']]
        pv_faces = uvc.rv_bdata[uvc.rv_bdata[:, -1] == self.bndry['pv']]
        
        if len(tv_faces) == 0:
            raise ValueError("No TV (tricuspid valve) faces found in RV boundary data")
        if len(pv_faces) == 0:
            raise ValueError("No PV (pulmonary valve) faces found in RV boundary data")
        
        # Compute face centroids for TV and PV
        # Face nodes are in columns 1:-1 (0-based indexing after read_bfile)
        tv_face_nodes = tv_faces[:, 1:-1].astype(int)
        pv_face_nodes = pv_faces[:, 1:-1].astype(int)
        
        tv_face_centroids = np.mean(rv_xyz[tv_face_nodes], axis=1)
        pv_face_centroids = np.mean(rv_xyz[pv_face_nodes], axis=1)
        
        # Compute distances from MV centroid to face centroids
        tv_distances = np.linalg.norm(tv_face_centroids - mv_centroid, axis=1)
        pv_distances = np.linalg.norm(pv_face_centroids - mv_centroid, axis=1)
        
        # Find closest and farthest faces
        tv_closest_idx = np.argmin(tv_distances)
        tv_farthest_idx = np.argmax(tv_distances)
        pv_closest_idx = np.argmin(pv_distances)
        pv_farthest_idx = np.argmax(pv_distances)
        
        # Get rv_lv_junction faces
        if 'rv_lv_junction' in self.bndry:
            rv_lv_junction_faces = uvc.rv_bdata[uvc.rv_bdata[:, -1] == self.bndry['rv_lv_junction']]
        elif 'rvlv_ant' in self.bndry and 'rvlv_post' in self.bndry:
            # If no rv_lv_junction patch, use rvlv_ant and rvlv_post
            rvlv_ant_faces = uvc.rv_bdata[uvc.rv_bdata[:, -1] == self.bndry['rvlv_ant']]
            rvlv_post_faces = uvc.rv_bdata[uvc.rv_bdata[:, -1] == self.bndry['rvlv_post']]
            rv_lv_junction_faces = np.vstack([rvlv_ant_faces, rvlv_post_faces])
        elif 'rv_lv_ant' in self.bndry and 'rv_lv_post' in self.bndry:
            # Use rv_lv_ant and rv_lv_post (created during circumferential computation)
            rv_lv_ant_faces = uvc.rv_bdata[uvc.rv_bdata[:, -1] == self.bndry['rv_lv_ant']]
            rv_lv_post_faces = uvc.rv_bdata[uvc.rv_bdata[:, -1] == self.bndry['rv_lv_post']]
            rv_lv_junction_faces = np.vstack([rv_lv_ant_faces, rv_lv_post_faces])
        else:
            raise ValueError("Cannot find rv_lv_junction faces. Need either rv_lv_junction, rvlv_ant/rvlv_post, or rv_lv_ant/rv_lv_post patches.")
        
        # Create new patch IDs for the BCs
        if 'rv_lat_zero' not in self.bndry:
            self.bndry['rv_lat_zero'] = 23
        if 'rv_lat_one' not in self.bndry:
            self.bndry['rv_lat_one'] = 24
        
        # Create a copy of rv_bdata to modify
        rv_bdata_new = uvc.rv_bdata.copy()
        
        # Get indices of TV and PV faces in the original rv_bdata
        tv_face_indices = np.where(uvc.rv_bdata[:, -1] == self.bndry['tv'])[0]
        pv_face_indices = np.where(uvc.rv_bdata[:, -1] == self.bndry['pv'])[0]
        
        # Get indices of closest and farthest faces
        tv_closest_face_idx = tv_face_indices[tv_closest_idx]
        tv_farthest_face_idx = tv_face_indices[tv_farthest_idx]
        pv_closest_face_idx = pv_face_indices[pv_closest_idx]
        pv_farthest_face_idx = pv_face_indices[pv_farthest_idx]
        
        # Get rv_lv_junction face indices
        if 'rv_lv_junction' in self.bndry:
            rv_lv_junction_face_indices = np.where(uvc.rv_bdata[:, -1] == self.bndry['rv_lv_junction'])[0]
        elif 'rvlv_ant' in self.bndry and 'rvlv_post' in self.bndry:
            rv_lv_junction_face_indices = np.where((uvc.rv_bdata[:, -1] == self.bndry['rvlv_ant']) | 
                                                   (uvc.rv_bdata[:, -1] == self.bndry['rvlv_post']))[0]
        elif 'rv_lv_ant' in self.bndry and 'rv_lv_post' in self.bndry:
            rv_lv_junction_face_indices = np.where((uvc.rv_bdata[:, -1] == self.bndry['rv_lv_ant']) | 
                                                   (uvc.rv_bdata[:, -1] == self.bndry['rv_lv_post']))[0]
        else:
            rv_lv_junction_face_indices = np.array([], dtype=int)
        
        # Assign BC = 0 to closest TV and PV faces, and rv_lv_junction faces
        zero_face_indices = np.concatenate([[tv_closest_face_idx], [pv_closest_face_idx], rv_lv_junction_face_indices])
        rv_bdata_new[zero_face_indices, -1] = self.bndry['rv_lat_zero']
        
        # Assign BC = +1 to farthest TV and PV faces
        one_face_indices = np.array([tv_farthest_face_idx, pv_farthest_face_idx])
        rv_bdata_new[one_face_indices, -1] = self.bndry['rv_lat_one']
        
        # Update rv_bdata
        uvc.rv_bdata = rv_bdata_new
        
        # Initialize RV mesh solver
        self.init_rv_mesh(uvc)
        
        # Run Laplace problem
        rv_solver = self.get_solver(method, 'rv')
        bcs_rv = {'face': {self.bndry['rv_lat_zero']: 0,
                           self.bndry['rv_lat_one']: 1}}
        
        rv_lat = rv_solver.solve(bcs_rv)
        rv_lat = rv_lat.x.petsc_vec.array[self.rv_corr]
        
        # Store in RV mesh
        uvc.rv_mesh.point_data['rv_lat'] = rv_lat
        
        return rv_lat

    def run_rv_lateral_angular(self, uvc):
        """
        Compute rv_lat coordinate using angular method.
        
        Defines two rotation axes:
        - TV_axis: line through RV epi apex and TV centroid
        - PV_axis: line through RV epi apex and PV centroid
        
        Finds closest TV/PV node pair (TV_0 and PV_0).
        Computes angular coordinates about each axis in perpendicular planes,
        then blends them using rv_circ1 as weight.
        """
        # Check prerequisites
        if 'rv_circ1' not in uvc.rv_mesh.point_data:
            raise ValueError("rv_circ1 must be computed before rv_lat. Call run_rv_circumferential1() first.")
        
        if not hasattr(uvc, 'rv_sep_epi_apex_node'):
            raise ValueError("RV epi apex node not defined. Call define_apex_nodes() first.")
        
        # Get RV epi apex
        rv_epi_apex_node = uvc.rv_sep_epi_apex_node
        rv_epi_apex = uvc.rv_mesh.points[rv_epi_apex_node]
        
        # Get TV and PV centroids
        if 'tv' not in uvc.valve_centroids:
            raise ValueError("TV (tricuspid valve) centroid not found.")
        if 'pv' not in uvc.valve_centroids:
            raise ValueError("PV (pulmonary valve) centroid not found.")
        
        tv_centroid = uvc.valve_centroids['tv']
        pv_centroid = uvc.valve_centroids['pv']
        
        # Define axes (unit vectors from RV epi apex to valve centroids)
        tv_axis = tv_centroid - rv_epi_apex
        tv_axis = tv_axis / np.linalg.norm(tv_axis)
        
        pv_axis = pv_centroid - rv_epi_apex
        pv_axis = pv_axis / np.linalg.norm(pv_axis)
        
        # Get TV and PV nodes from RV boundary data
        tv_nodes = np.unique(uvc.rv_bdata[uvc.rv_bdata[:, -1] == self.bndry['tv'], 1:-1])
        pv_nodes = np.unique(uvc.rv_bdata[uvc.rv_bdata[:, -1] == self.bndry['pv'], 1:-1])
        
        if len(tv_nodes) == 0:
            raise ValueError("No TV (tricuspid valve) nodes found in RV boundary data")
        if len(pv_nodes) == 0:
            raise ValueError("No PV (pulmonary valve) nodes found in RV boundary data")
        
        # Get RV mesh coordinates
        rv_xyz = uvc.rv_mesh.points
        
        # Find TV_0: TV node closest to any PV node
        tv_coords = rv_xyz[tv_nodes]
        pv_coords = rv_xyz[pv_nodes]
        
        # Compute all pairwise distances between TV and PV nodes
        tv_to_pv_distances = np.linalg.norm(tv_coords[:, np.newaxis] - pv_coords, axis=2)
        tv_0_idx, pv_0_corr_idx = np.unravel_index(np.argmin(tv_to_pv_distances), tv_to_pv_distances.shape)
        tv_0_node = tv_nodes[tv_0_idx]
        tv_0 = rv_xyz[tv_0_node]
        
        # Find PV_0: PV node closest to any TV node
        pv_0_node = pv_nodes[pv_0_corr_idx]
        pv_0 = rv_xyz[pv_0_node]

        #compute average of tv_0 and pv_0
        tv_pv_bridge = (tv_0 + pv_0) / 2
        
        # Get rv_circ1 values
        rv_circ1 = uvc.rv_mesh.point_data['rv_circ1']
        
        # Vectorize: compute for all points at once
        # Vectors from RV epi apex to all points
        vecs_to_points = rv_xyz - rv_epi_apex
        
        # Project all points onto planes perpendicular to axes
        # TV axis: remove component along TV_axis
        dot_tv = np.dot(vecs_to_points, tv_axis)[:, np.newaxis]
        proj_tv = vecs_to_points - dot_tv * tv_axis
        
        # PV axis: remove component along PV_axis
        dot_pv = np.dot(vecs_to_points, pv_axis)[:, np.newaxis]
        proj_pv = vecs_to_points - dot_pv * pv_axis
        
        # Normalize projections
        proj_tv_norms = np.linalg.norm(proj_tv, axis=1)
        proj_pv_norms = np.linalg.norm(proj_pv, axis=1)
        
        # Avoid division by zero (points on axis)
        proj_tv_norms[proj_tv_norms < 1e-10] = 1e-10
        proj_pv_norms[proj_pv_norms < 1e-10] = 1e-10
        
        proj_tv_normalized = proj_tv / proj_tv_norms[:, np.newaxis]
        proj_pv_normalized = proj_pv / proj_pv_norms[:, np.newaxis]
        
        # Compute angles using atan2 for better accuracy
        # For TV axis (counterclockwise)
        # Project TV_0 onto the perpendicular plane to get its normalized direction
        tv_0_vec = rv_epi_apex - tv_pv_bridge
        tv_0_proj = tv_0_vec - np.dot(tv_0_vec, tv_axis) * tv_axis
        tv_0_proj_norm = np.linalg.norm(tv_0_proj)
        if tv_0_proj_norm < 1e-10:
            raise ValueError("TV_0 is on the TV axis")
        tv_0_proj_normalized = tv_0_proj / tv_0_proj_norm
        
        cos_tv = np.clip(np.sum(proj_tv_normalized * tv_0_proj_normalized, axis=1), -1, 1)
        # Compute cross product for each point: cross(tv_0_proj_normalized, proj_tv_normalized[i])
        tv_0_proj_expanded = np.tile(tv_0_proj_normalized, (len(proj_tv_normalized), 1))
        cross_tv = np.cross(tv_0_proj_expanded, proj_tv_normalized)
        sin_tv = np.sum(cross_tv * tv_axis, axis=1)  # Dot product component along axis
        angle_tv = np.arctan2(sin_tv, cos_tv)
        
        # For PV axis (clockwise): negate the angle
        # Project PV_0 onto the perpendicular plane to get its normalized direction
        pv_0_vec = rv_epi_apex - tv_pv_bridge
        pv_0_proj = pv_0_vec - np.dot(pv_0_vec, pv_axis) * pv_axis
        pv_0_proj_norm = np.linalg.norm(pv_0_proj)
        if pv_0_proj_norm < 1e-10:
            raise ValueError("PV_0 is on the PV axis")
        pv_0_proj_normalized = pv_0_proj / pv_0_proj_norm
        
        cos_pv = np.clip(np.sum(proj_pv_normalized * pv_0_proj_normalized, axis=1), -1, 1)
        pv_0_proj_expanded = np.tile(pv_0_proj_normalized, (len(proj_pv_normalized), 1))
        cross_pv = np.cross(pv_0_proj_expanded, proj_pv_normalized)
        sin_pv = -np.sum(cross_pv * pv_axis, axis=1)  # Negate for clockwise rotation
        angle_pv = np.arctan2(sin_pv, cos_pv)
        
        # Scale factor: tanh(k * 0.5) should be close to 1
        # k = 6 gives tanh(3) ≈ 0.995, providing good saturation at boundaries
        k = 6.0
        
        # Apply tanh to rv_circ1 (tanh ranges from -1 to 1)
        tanh_val = np.tanh(k * rv_circ1)
        
    
        weight_pv = (tanh_val + 1.0) / 2.0
        weight_tv = 1.0 - weight_pv
        
        # Clamp to ensure no blending outside [-0.5, 0.5]
        weight_tv = np.where(rv_circ1 <= -0.5, 1.0,
                    np.where(rv_circ1 >= 0.5, 0.0, weight_tv))
        weight_pv = np.where(rv_circ1 <= -0.5, 0.0,
                    np.where(rv_circ1 >= 0.5, 1.0, weight_pv))
        
        # Blend angles
        rv_lat = weight_tv * angle_tv + weight_pv * angle_pv
        
        # Store in RV mesh
        uvc.rv_mesh.point_data['rv_lat'] = rv_lat
        uvc.rv_mesh.point_data['rv_lat_tv'] = angle_tv  # Unblended angle about TV axis
        uvc.rv_mesh.point_data['rv_lat_pv'] = angle_pv  # Unblended angle about PV axis
        
        return rv_lat



    def run_fast_circumferential(self, uvc, method='laplace'):
        self.init_lv_mesh(uvc)
        lv_solver = self.get_solver(method, 'lv')
        bcs_point = {}
        for i in range(len(uvc.lv_zero_nodes)):
            bcs_point[tuple(uvc.lv_mesh.points[uvc.lv_zero_nodes[i]])] = 0
        bcs_lv = {'face': {self.bndry['sep0']: 0,
                  self.bndry['lat0']: 1,
                  self.bndry['sep_ant']: 1/3,
                  self.bndry['sep_post']: 1/3},
                  'point': bcs_point,}

        lv_circ = lv_solver.solve(bcs_lv)
        lv_circ = lv_circ.x.petsc_vec.array[self.lv_corr]
        lv_circ = uvc.correct_lv_circ_by_subdomain(lv_circ)

        uvc.lv_mesh.point_data['circ'] = lv_circ*np.pi

        # TODO need to generate rv boundaries so it doesn't take the base elems.
        self.init_rv_mesh(uvc)
        rv_solver = self.get_solver(method, 'rv')
        bcs_rv = {'face': {self.bndry['rvlv_ant']: 1/3,
                  self.bndry['rvlv_post']: -1/3}}
        rv_circ = rv_solver.solve(bcs_rv)
        rv_circ = rv_circ.x.petsc_vec.array[self.rv_corr]
        uvc.rv_mesh.point_data['circ'] = rv_circ*np.pi

        return lv_circ, rv_circ

    def run_rv_lv_marker(self, uvc, method = 'laplace'):
        self.init_lv_mesh(uvc, og=True)
        lv_solver = self.get_solver(method, 'lv')

        bcs_lv = {'face': {self.bndry['lv_endo']: 0 }}
        if 'rv_lv_junction' in self.bndry:
            bcs_lv['face'][self.bndry['rv_lv_junction']] = 1
        else:
            bcs_lv['face'][self.bndry['rvlv_ant']] = 1
            bcs_lv['face'][self.bndry['rvlv_post']] = 1
        lv_rvlv = lv_solver.solve(bcs_lv)
        lv_rvlv = lv_rvlv.x.petsc_vec.array[self.lv_corr]
        uvc.lv_mesh.point_data['lv_rvlv'] = lv_rvlv

        return lv_rvlv


    def save_mmg_boundaries(self):
        mesh, bc = self.lv_mmg_mesh, self.lv_mmg_bc
        sep_nodes = np.where(bc == 1)[0]
        mpi_nodes = np.where(bc == 2)[0]
        ant_nodes = np.where(bc == 4)[0]
        post_nodes = np.where(bc == 5)[0]
        zero_nodes = np.where(bc == 10)[0]

        div_nodes = np.concatenate([sep_nodes, mpi_nodes, zero_nodes])

        bdata = self.point_to_face_marker(
            div_nodes, 1, new_faces=True, mesh=mesh)

        surf0 = io.Mesh(mesh.points, {'triangle': bdata[:,1:-1]})

        io.write('zero.stl', surf0)


        div_nodes = np.concatenate([ant_nodes, post_nodes, zero_nodes])
        bdata = self.point_to_face_marker(
            div_nodes, 1, new_faces=True, mesh=mesh)

        surf0 = io.Mesh(mesh.points, {'triangle': bdata[:,1:-1]})

        io.write('sep.stl', surf0)


    def run_ot_circumferential1(self, uvc, method='laplace'):
        self.init_ot_mesh(uvc)
        uvc.create_ot_circ_bc1()

        bcs_marker = {'function': {self.bndry['base']:
                                   uvc.ot_mesh.point_data['bc1'][self.ot_icorr]}}

        ot_solver = self.get_solver(method, 'ot')
        ot_circ = ot_solver.solve(bcs_marker)
        ot_circ = ot_circ.x.petsc_vec.array[self.ot_corr]
        uvc.ot_mesh.point_data['ot_circ1'] = ot_circ

        return ot_circ

    def run_ot_circumferential2(self, ot_circ1, uvc, method='laplace', correct=True):

        if self.mmg:
            mesh, bc = uc.mmg_create_ot_circ_bc2(ot_circ1, uvc)
            sep_nodes = np.where(bc == 1)[0]
            mpi_nodes = np.where(bc == 2)[0]
            pi_nodes = np.where(bc == 3)[0]
            tv_nodes = np.where(bc == 4)[0]
            base_nodes = np.where(bc == 5)[0]
            bdata = self.point_to_face_marker(
                sep_nodes, 1, new_faces=True, mesh=mesh)
            bdata = self.point_to_face_marker(
                mpi_nodes, 2, bdata, new_faces=True, mesh=mesh)
            bdata = self.point_to_face_marker(
                pi_nodes, 3, bdata, new_faces=True, mesh=mesh)
            bdata = self.point_to_face_marker(
                tv_nodes, 4, bdata, new_faces=True, mesh=mesh)
            bdata = self.point_to_face_marker(
                base_nodes, self.bndry['base'], bdata, new_faces=True, mesh=mesh)

            # Need to figure out a better way to map...
            bc1 = np.zeros(len(mesh.points))
            bc1[:len(uvc.ot_mesh.points)] = uvc.ot_mesh.point_data['bc1']/np.pi
            mesh.point_data['bc1'] = bc1
            mesh_, mt = dxio.read_meshio_mesh(mesh, bdata)
            _, icorr = dxio.find_vtu_dx_mapping(mesh_)
            bc1 = bc1[icorr]

            circ = run_coord(mesh, bdata, {'face': {1: 0, 2: -1, 3: 1, 4: 0},
                                           'function': {self.bndry['base']:
                                                                      bc1}})
            ot_circ2i = circ[:len(uvc.ot_mesh.points)]
            uvc.ot_mesh.point_data['ot_circ2i'] = ot_circ2i

            mesh.point_data['ot_circ2i'] = circ

            mesh, bc = uc.mmg_create_ot_circ_bc3(ot_circ2i, uvc)
            sep_nodes = np.where(bc == 1)[0]
            mpi_nodes = np.where(bc == 2)[0]
            pi_nodes = np.where(bc == 3)[0]
            ant_nodes = np.where(bc == 4)[0]
            post_nodes = np.where(bc == 5)[0]
            base_nodes = np.where(bc == 6)[0]
            bdata = self.point_to_face_marker(
                sep_nodes, 1, new_faces=True, mesh=mesh)
            bdata = self.point_to_face_marker(
                mpi_nodes, 2, bdata, new_faces=True, mesh=mesh)
            bdata = self.point_to_face_marker(
                pi_nodes, 3, bdata, new_faces=True, mesh=mesh)
            bdata = self.point_to_face_marker(
                ant_nodes, 4, bdata, new_faces=True, mesh=mesh)
            bdata = self.point_to_face_marker(
                post_nodes, 5, bdata, new_faces=True, mesh=mesh)
            bdata = self.point_to_face_marker(
                base_nodes, self.bndry['base'], bdata, new_faces=True, mesh=mesh)

            # Need to figure out a better way to map...
            bc1 = np.zeros(len(mesh.points))
            bc1[:len(uvc.ot_mesh.points)] = uvc.ot_mesh.point_data['bc1']/np.pi
            mesh.point_data['bc1'] = bc1

            mesh_, mt = dxio.read_meshio_mesh(mesh, bdata)
            _, icorr = dxio.find_vtu_dx_mapping(mesh_)
            bc1 = bc1[icorr]
            circ = run_coord(mesh, bdata, {'face': {1: 0, 2: -1, 3: 1, 4: 1/3, 5: -1/3},
                                            'function': {self.bndry['base']: bc1}})
            ot_circ2 = circ[:len(uvc.lv_mesh.points)]*np.pi

        else:
            uvc.create_ot_circ_bc2(ot_circ1)

            bc_point = uvc.ot_mesh.point_data['bc2']
            lat_marker_neg = uvc.ot_bc_mpi
            lat_marker_pls = uvc.ot_bc_pi

            self.bndry['lat_pi'] = 23
            self.bndry['lat_mpi'] = 24
            uvc.ot_bdata = self.point_to_face_marker(lat_marker_pls, self.bndry['lat_pi'],
                                                     uvc.ot_bdata, new_faces=True, mesh=uvc.ot_mesh)
            uvc.ot_bdata = self.point_to_face_marker(lat_marker_neg, self.bndry['lat_mpi'],
                                                     uvc.ot_bdata, new_faces=True, mesh=uvc.ot_mesh)

            self.init_ot_mesh(uvc)

            bcs_ot = {'function': {self.bndry['lat_pi']: np.ones(len(bc_point))*np.pi,
                      self.bndry['lat_mpi']: np.ones(len(bc_point))*-np.pi,
                      self.bndry['base']: uvc.ot_mesh.point_data['bc1'][self.ot_icorr]}}

            ot_solver = self.get_solver(method, 'ot')
            ot_circ2 = ot_solver.solve(bcs_ot)
            ot_circ2 = ot_circ2.x.petsc_vec.array[self.ot_corr]
        uvc.ot_mesh.point_data['circ'] = ot_circ2

        return ot_circ2

    def run_ot_circumferential(self, uvc, method='laplace'):
        ot_circ1 = self.run_ot_circumferential1(uvc, method)
        ot_circ2 = self.run_ot_circumferential2(ot_circ1, uvc, method)

        return ot_circ2


    def get_local_vectors(self, uvc, which='lv', linear=True):
        glong = self.get_func_gradient(uvc, 'long', which, linear=linear)
        eL = glong/np.linalg.norm(glong, axis=1)[:,None]

        gtrans = self.get_func_gradient(uvc, 'trans', which, linear=linear)
        eT = gtrans/np.linalg.norm(gtrans, axis=1)[:,None]

        eC = np.cross(eL, eT, axisa=1, axisb=1)
        eC = eC/np.linalg.norm(eC, axis=1)[:,None]

        if which == 'lv':
            mesh = uvc.lv_mesh
        elif which == 'rv':
            mesh = uvc.rv_mesh
        elif which == 'bv':
            mesh = uvc.bv_mesh

        if linear:
            mesh.point_data['eL'] = eL
            mesh.point_data['eC'] = eC
            mesh.point_data['eT'] = eT
        else:
            mesh.cell_data['eL'] = [eL]
            mesh.cell_data['eC'] = [eC]
            mesh.cell_data['eT'] = [eT]

