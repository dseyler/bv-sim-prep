#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 08:17:37 2023

@author: Javiera Jilberto Vallejos
"""
import sys
import argparse
import os
import numpy as np
import cheartio as chio
from uvcgen.model_coords import UVCGen
from uvcgen.UVC import UVC
from uvcgen.uvc_outputs import export_info, export_cheart_inputs


def find_apex_from_reference(ref_folder, current_mesh, current_bdata, boundaries_dict):
    """
    Find the LV apex point based on a reference mesh with a pre-defined apex.
    
    Args:
        ref_folder: Path to folder containing reference mesh files
        current_mesh: Current mesh object
        current_bdata: Current boundary data
        boundaries_dict: Current boundaries dictionary
        
    Returns:
        Node ID (0-based) of the closest point on LV_epi to the reference apex
    """
    ref_folder = os.path.abspath(ref_folder)
    if not ref_folder.endswith('/'):
        ref_folder += '/'
    
    # Check for required files
    ref_boundaries_path = os.path.join(ref_folder, 'boundaries.P')
    ref_x_path = os.path.join(ref_folder, 'bv_model_gen_FE.X')
    
    if not os.path.exists(ref_boundaries_path):
        raise FileNotFoundError(f"Reference boundaries.P not found: {ref_boundaries_path}")
    if not os.path.exists(ref_x_path):
        raise FileNotFoundError(f"Reference bv_model_gen_FE.X not found: {ref_x_path}")
    
    # Read reference boundaries to get apex node ID
    ref_boundaries = chio.pfile_to_dict(ref_boundaries_path, fmt=int)
    if 'apex' not in ref_boundaries:
        raise ValueError(f"Reference boundaries.P does not contain 'apex' entry")
    ref_apex_id = ref_boundaries['apex']
    
    # Read reference coordinates from X file
    # X file format: first line is "num_nodes num_dims", then coordinates
    ref_coords = chio.read_dfile(ref_x_path)
    
    # Check if apex ID is valid (X file uses 0-based indexing for array access)
    # boundaries.P may use 0-based or 1-based - check both
    if ref_apex_id < 0 or ref_apex_id >= len(ref_coords):
        # Try 1-based indexing
        ref_apex_id_1based = ref_apex_id - 1
        if ref_apex_id_1based >= 0 and ref_apex_id_1based < len(ref_coords):
            ref_apex_id = ref_apex_id_1based
        else:
            raise ValueError(f"Reference apex ID {ref_boundaries['apex']} is out of range [0, {len(ref_coords)-1}]")
    
    # Get reference apex coordinates
    ref_apex_coords = ref_coords[ref_apex_id]
    
    # Get epi_lv patch ID from current mesh
    if 'lv_epi' in boundaries_dict:
        lv_epi_patch = boundaries_dict['lv_epi']
    elif 'epi' in boundaries_dict:
        lv_epi_patch = boundaries_dict['epi']
    else:
        raise ValueError("Could not find lv_epi or epi in boundaries")
    
    # Find faces on lv_epi surface in current mesh
    lv_epi_faces = current_bdata[current_bdata[:, -1] == lv_epi_patch]
    if len(lv_epi_faces) == 0:
        raise ValueError(f"No faces found for lv_epi patch {lv_epi_patch}")
    
    # Get unique point indices on lv_epi surface
    lv_epi_points = np.unique(lv_epi_faces[:, 1:-1].flatten())
    lv_epi_points = lv_epi_points - 1  # Convert to 0-based indexing
    
    # Get coordinates of epi_lv points in current mesh
    current_epi_coords = current_mesh.points[lv_epi_points]
    
    # Find closest point on LV_epi to reference apex coordinates
    distances = np.linalg.norm(current_epi_coords - ref_apex_coords, axis=1)
    closest_idx_in_epi = np.argmin(distances)
    closest_apex_id = lv_epi_points[closest_idx_in_epi]
    
    print(f"  Reference apex coordinates: {ref_apex_coords}")
    print(f"  Closest LV_epi point ID: {closest_apex_id}, distance: {distances[closest_idx_in_epi]:.6f}")
    
    return closest_apex_id


def find_apex_point(mesh, bdata, boundaries_dict):
    """
    Find the LV apex point (typically the point on epi_lv farthest from the base).
    
    This is a simplified approach - in practice, the apex should be identified
    from anatomical landmarks or user input.
    """
    # Get epi_lv patch ID (should be 3)
    if 'lv_epi' in boundaries_dict:
        lv_epi_patch = boundaries_dict['lv_epi']
    elif 'epi' in boundaries_dict:
        # If no split epi, use epi patch
        lv_epi_patch = boundaries_dict['epi']
    else:
        raise ValueError("Could not find lv_epi or epi in boundaries")
    
    # Find faces on lv_epi surface
    lv_epi_faces = bdata[bdata[:, -1] == lv_epi_patch]
    if len(lv_epi_faces) == 0:
        raise ValueError(f"No faces found for lv_epi patch {lv_epi_patch}")
    
    # Get unique point indices on lv_epi surface
    lv_epi_points = np.unique(lv_epi_faces[:, 1:-1].flatten())
    lv_epi_points = lv_epi_points - 1  # Convert to 0-based indexing
    
    # Get coordinates of epi_lv points
    epi_coords = mesh.points[lv_epi_points]
    
    # Find centroid of top/base (valve surfaces)
    # Use mv (mitral valve) as base reference
    if 'mv' in boundaries_dict:
        mv_patch = boundaries_dict['mv']
        mv_faces = bdata[bdata[:, -1] == mv_patch]
        if len(mv_faces) > 0:
            mv_points = np.unique(mv_faces[:, 1:-1].flatten())
            mv_points = mv_points - 1
            base_center = np.mean(mesh.points[mv_points], axis=0)
        else:
            # Fallback: use mean of all epi points
            base_center = np.mean(epi_coords, axis=0)
    else:
        base_center = np.mean(epi_coords, axis=0)
    
    # Find point farthest from base center
    distances = np.linalg.norm(epi_coords - base_center, axis=1)
    apex_idx_in_epi = np.argmax(distances)
    apex_global_idx = lv_epi_points[apex_idx_in_epi]
    
    return apex_global_idx


def create_boundaries_file(mesh_folder, mesh_path, boundaries_dict, apex_id):
    """Create boundaries.P file from dictionary."""
    boundaries_path = os.path.join(mesh_folder, 'boundaries.P')
    chio.dict_to_pfile(boundaries_path, boundaries_dict)
    print(f"Created boundaries.P with apex={apex_id}")
    return boundaries_path


def get_lv_epi_patch_id(boundaries_dict):
    """
    Helper to retrieve the LV epicardial patch ID from a boundaries dictionary.
    Prefers 'lv_epi' and falls back to 'epi'.
    """
    if 'lv_epi' in boundaries_dict:
        return boundaries_dict['lv_epi']
    if 'epi' in boundaries_dict:
        return boundaries_dict['epi']
    raise ValueError("Could not find lv_epi or epi in boundaries")


def find_lv_epi_apex_face_from_boundaries_apex(mesh, bdata, boundaries_dict):
    """
    Using the apex node stored in boundaries.P (boundaries_dict['apex']),
    select one LV epicardial face that:
      - lies on the LV epicardial surface (lv_epi or epi), and
      - contains that apex node,
    and among those faces, pick the one whose centroid is farthest from the MV
    centroid (computed in the same way as in find_apex_point()).

    Returns
    -------
    apex_face_nodes_0based : np.ndarray or None
        Array of length 3 with 0-based node indices for the chosen face, or
        None if a suitable face cannot be found (caller should then fall back
        to existing behaviour).
    """
    if 'apex' not in boundaries_dict:
        print("  Warning: 'apex' not found in boundaries.P, keeping default bv_sep_apex_nodes")
        return None

    apex_id = boundaries_dict['apex']
    num_points = len(mesh.points)
    if apex_id < 0 or apex_id >= num_points:
        print(f"  Warning: boundaries.P apex ID {apex_id} is out of range [0, {num_points-1}], "
              "keeping default bv_sep_apex_nodes")
        return None

    try:
        lv_epi_patch = get_lv_epi_patch_id(boundaries_dict)
    except ValueError as e:
        print(f"  Warning: {e}, keeping default bv_sep_apex_nodes")
        return None

    # bdata already uses 0-based node indices (cheartio.read_bfile)
    apex_node_bfile = apex_id + 1

    # Faces on LV epi surface
    lv_epi_faces = bdata[bdata[:, -1] == lv_epi_patch]
    if len(lv_epi_faces) == 0:
        print(f"  Warning: No faces found for lv_epi/epi patch {lv_epi_patch}, "
              "keeping default bv_sep_apex_nodes")
        return None

    # Restrict to faces that contain the apex node
    face_nodes = lv_epi_faces[:, 1:-1].astype(int)
    mask_contains_apex = np.any(face_nodes == apex_node_bfile, axis=1)
    candidate_faces = lv_epi_faces[mask_contains_apex]
    if len(candidate_faces) == 0:
        print("  Warning: No LV epicardial faces contain the apex node from boundaries.P, "
              "keeping default bv_sep_apex_nodes")
        return None

    # Compute MV centroid in the same spirit as find_apex_point()
    if 'mv' not in boundaries_dict:
        print("  Warning: 'mv' patch not in boundaries, keeping default bv_sep_apex_nodes")
        return None

    mv_patch = boundaries_dict['mv']
    mv_faces = bdata[bdata[:, -1] == mv_patch]
    if len(mv_faces) == 0:
        print("  Warning: No faces found for mv patch when defining LV apex face, "
              "keeping default bv_sep_apex_nodes")
        return None

    mv_points = np.unique(mv_faces[:, 1:-1].flatten()).astype(int)
    mv_centroid = np.mean(mesh.points[mv_points], axis=0)

    # Among candidate faces, pick the one whose centroid is farthest from MV centroid
    cand_nodes = candidate_faces[:, 1:-1].astype(int)  # shape (n_faces, 3)
    cand_points = mesh.points[cand_nodes]
    cand_centroids = np.mean(cand_points, axis=1)  # (n_faces, 3)
    dists = np.linalg.norm(cand_centroids - mv_centroid[None, :], axis=1)
    farthest_idx = np.argmax(dists)
    apex_face_nodes_0based = cand_nodes[farthest_idx]

    # Basic sanity check: ensure all nodes are within range
    if np.any(apex_face_nodes_0based < 0) or np.any(apex_face_nodes_0based >= num_points):
        print("  Warning: Computed LV apex face has invalid node indices, "
              "keeping default bv_sep_apex_nodes")
        return None

    return apex_face_nodes_0based


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Universal Ventricular Coordinates (UVC) from CH format mesh"
    )
    parser.add_argument(
        "--mesh-folder",
        type=str,
        required=True,
        help="Path to folder containing CH format files (bv_model_gen_FE.T, bv_model_gen_FE.X, bv_model_gen_FE.B)",
    )
    parser.add_argument(
        "--mesh-name",
        type=str,
        default="bv_model_gen",
        help="Base name of CH mesh files (default: bv_model_gen)",
    )
    parser.add_argument(
        "--region-file",
        type=str,
        default=None,
        help="Optional path to region.FE file for LV/RV split (default: None)",
    )
    parser.add_argument(
        "--apex-id",
        type=int,
        default=None,
        help="Optional apex point ID (0-based). If not provided, will be computed automatically.",
    )
    parser.add_argument(
        "--apex-ref",
        type=str,
        default=None,
        help="Path to folder containing reference mesh with pre-defined apex (must contain boundaries.P and bv_model_gen_FE.X)",
    )
    parser.add_argument(
        "--threshold-septum",
        type=float,
        default=0.6,
        help="Threshold for septum identification (default: 0.6)",
    )
    parser.add_argument(
        "--threshold-long",
        type=float,
        default=1.0,
        help="Threshold for longitudinal coordinate (default: 1.0)",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    
    # Setup paths
    mesh_folder = os.path.abspath(args.mesh_folder)
    if not mesh_folder.endswith('/'):
        mesh_folder += '/'
    mesh_path = os.path.join(mesh_folder, args.mesh_name)
    
    # Check if boundaries.P exists, if not create it
    boundaries_path = os.path.join(mesh_folder, 'boundaries.P')
    if os.path.exists(boundaries_path):
        print(f"Loading existing boundaries.P from {boundaries_path}")
        boundaries = chio.pfile_to_dict(boundaries_path, fmt=int)
    else:
        print("Creating boundaries.P file...")
        # Define standard patch mapping (matching bv-sim-prep convention)
        boundaries = {
            'lv_endo': 1,
            'rv_endo': 2,
            'lv_epi': 3,
            'rv_epi': 4,
            'rv_septum': 5,
            'av': 6,
            'pv': 7,
            'tv': 8,
            'mv': 9,
            'rv_lv_junction': 10,
        }
        
        # Load mesh to find apex
        print("  Loading mesh to identify apex point...")
        bv_mesh = chio.read_mesh(mesh_path, meshio=True)
        bdata = chio.read_bfile(mesh_path)
        
        # Priority order: --apex-ref > --apex-id > automatic computation
        if args.apex_ref is not None:
            print(f"  Using reference mesh apex from: {args.apex_ref}")
            apex_id = find_apex_from_reference(args.apex_ref, bv_mesh, bdata, boundaries)
            print(f"  Found apex ID from reference: {apex_id}")
        elif args.apex_id is not None:
            apex_id = args.apex_id
            print(f"  Using provided apex ID: {apex_id}")
        else:
            print("  Computing apex point automatically...")
            apex_id = find_apex_point(bv_mesh, bdata, boundaries)
            print(f"  Found apex ID: {apex_id}")
        
        boundaries['apex'] = apex_id
        create_boundaries_file(mesh_folder, mesh_path, boundaries, apex_id)
    
    # Load boundaries and apex
    boundaries = chio.pfile_to_dict(boundaries_path, fmt=int)
    apex_id = boundaries['apex']  # lv apex point in the epicardium

    thresholds = {
        'septum': args.threshold_septum,  # Splits RV from LV.
        'long': args.threshold_long
    }
    method = 'laplace'
    
    # Read CH mesh
    print('Initializing')
    bv_mesh = chio.read_mesh(mesh_path, meshio=True)
    bdata = chio.read_bfile(mesh_path)
    
    # Generate GlobalNodeID and GlobalElementID if not present
    # CH format files are sorted by these IDs, so we can generate sequential 1-indexed IDs
    if 'GlobalNodeID' not in bv_mesh.point_data:
        # X file is sorted by GlobalNodeID, so generate sequential IDs starting from 1
        bv_mesh.point_data['GlobalNodeID'] = np.arange(1, len(bv_mesh.points) + 1, dtype=np.int32)
        print(f"  Generated GlobalNodeID: {bv_mesh.point_data['GlobalNodeID'][0]} to {bv_mesh.point_data['GlobalNodeID'][-1]}")
    
    if 'GlobalElementID' not in bv_mesh.cell_data:
        # T file is sorted by GlobalElementID, so generate sequential IDs starting from 1
        bv_mesh.cell_data['GlobalElementID'] = [np.arange(1, len(bv_mesh.cells[0].data) + 1, dtype=np.int32)]
        print(f"  Generated GlobalElementID: {bv_mesh.cell_data['GlobalElementID'][0][0]} to {bv_mesh.cell_data['GlobalElementID'][0][-1]}")
    
    # Check if AV (patch ID 6) exists in boundary data
    av_patch_id = boundaries.get('av', 6)
    has_av = av_patch_id in bdata[:, -1]
    if not has_av:
        print("AV (aortic valve) not found in boundary data - will use MV (mitral valve) instead")
        if 'av' in boundaries:
            del boundaries['av']
    else:
        print(f"AV (aortic valve) found in boundary data (patch ID {av_patch_id})")
    
    # Load region split file if provided or if region.FE exists
    rvlv = None
    region_split_file = None
    
    # First, check if user provided explicit region file
    if args.region_file:
        region_split_file = args.region_file
    else:
        # Check for region.FE in mesh folder (cheart subdirectory or root)
        cheart_region_path = os.path.join(mesh_folder, 'cheart', 'region.FE')
        root_region_path = os.path.join(mesh_folder, 'region.FE')
        
        if os.path.exists(cheart_region_path):
            region_split_file = cheart_region_path
            print(f"Found region.FE in cheart/ subdirectory")
        elif os.path.exists(root_region_path):
            region_split_file = root_region_path
            print(f"Found region.FE in mesh folder root")
    
    # Load the region file if found
    if region_split_file:
        if os.path.exists(region_split_file):
            try:
                rvlv = 1 - chio.read_dfile(region_split_file)  # This file is 0 in LV, 1 in RV
                print(f"Loaded region split from {region_split_file}")
                print(f"  LV elements: {np.sum(rvlv == 0)}, RV elements: {np.sum(rvlv == 1)}")
            except Exception as e:
                print(f"Warning: Could not load region file: {e}")
                rvlv = None
        else:
            print(f"Warning: Region file not found: {region_split_file}")
    else:
        print("No region.FE file found - will compute LV/RV split from boundary data or Laplace solver")
    
    # Initialize classes
    uvc = UVC(bv_mesh, bdata, boundaries, thresholds, mesh_folder, rvlv=rvlv)
    mcg = UVCGen(uvc, mmg=True)
    
    # Determine how to get septum
    # Since epi is already split and septum is already identified in boundary data,
    # we extract it directly from the boundary data without solving Laplace
    if not uvc.split_epi:
        # No split epi - solve Laplace for septum
        print("  Computing septum via Laplace solver (no split epi)...")
        septum = mcg.run_septum(uvc)
    elif rvlv is not None:
        # Split epi + rvlv file - read septum from rvlv
        print("  Reading septum from rvlv file...")
        septum = uvc.compute_septum()
    else:
        # Split epi but no rvlv - extract septum directly from boundary data
        # Since epi is already split and septum is already identified, skip computation
        print("  Extracting septum from existing boundary data (epi already split)...")
        if 'rv_septum' in boundaries:
            septum_patch = boundaries['rv_septum']
            septum_faces = bdata[bdata[:, -1] == septum_patch]
            if len(septum_faces) > 0:
                # Get all points on septum surface
                septum_points = np.unique(septum_faces[:, 1:-1].flatten())
                septum_points = septum_points - 1  # Convert to 0-based
                # Create septum field: 0 = LV, 1 = RV, 0.5 = septum
                septum = np.ones(len(bv_mesh.points)) * 0.5  # Default to septum
                # Mark LV points (endo_lv, epi_lv, mv, av or mv if av missing)
                lv_patches = [boundaries.get('lv_endo', 1), boundaries.get('lv_epi', 3), 
                             boundaries.get('mv', 9)]
                # Add AV if it exists, otherwise MV will be used (already added above)
                if 'av' in boundaries:
                    lv_patches.append(boundaries['av'])
                for patch in lv_patches:
                    if patch in bdata[:, -1]:
                        lv_faces = bdata[bdata[:, -1] == patch]
                        lv_points = np.unique(lv_faces[:, 1:-1].flatten()) - 1
                        septum[lv_points] = 0.0
                # Mark RV points (endo_rv, epi_rv, tv, pv)
                rv_patches = [boundaries.get('rv_endo', 2), boundaries.get('rv_epi', 4),
                             boundaries.get('tv', 8), boundaries.get('pv', 7)]
                for patch in rv_patches:
                    if patch in bdata[:, -1]:
                        rv_faces = bdata[bdata[:, -1] == patch]
                        rv_points = np.unique(rv_faces[:, 1:-1].flatten()) - 1
                        septum[rv_points] = 1.0
                # Septum points stay at 0.5
                print(f"    Extracted {np.sum(septum == 0.5)} septum points from boundary data")
                # Store in mesh point data for later use
                uvc.bv_mesh.point_data['septum'] = septum
                
                # Create rvlv cell data from septum point data
                # rvlv: 0 = LV, 1 = RV (opposite of septum where 0.5 = septum)
                # For each cell, use the mean of septum values at its nodes
                ien = bv_mesh.cells[0].data
                rvlv = np.zeros(len(ien), dtype=float)
                for i, cell_nodes in enumerate(ien):
                    cell_septum_vals = septum[cell_nodes]
                    # Use mean to determine cell classification
                    mean_septum = np.mean(cell_septum_vals)
                    if mean_septum < thresholds['septum']:
                        rvlv[i] = 0  # LV
                    else:
                        rvlv[i] = 1  # RV
                # Store rvlv in mesh cell data
                uvc.bv_mesh.cell_data['rvlv'] = [rvlv]
                print(f"    Created rvlv cell data: {np.sum(rvlv == 0)} LV cells, {np.sum(rvlv == 1)} RV cells")
            else:
                # Fallback: solve Laplace for septum
                print("  Warning: No septum faces found in boundary data, solving Laplace for septum")
                septum = mcg.run_septum(uvc)
        else:
            # No septum patch defined, solve Laplace
            print("  Warning: No rv_septum in boundaries, solving Laplace for septum")
            septum = mcg.run_septum(uvc)
    
    uvc.compute_long_plane_coord(septum)
    uvc.split_rv_lv(septum)
    
    # Compute coordinates
    uvc.define_apex_nodes()

    # ------------------------------------------------------------------
    # Custom LV apex triple for longitudinal BCs, based on boundaries.P
    # ------------------------------------------------------------------
    custom_apex_nodes = find_lv_epi_apex_face_from_boundaries_apex(bv_mesh, bdata, boundaries)
    if custom_apex_nodes is not None:
        # Overwrite the default septal apex nodes used by run_longitudinal.
        # These are 0-based indices into uvc.bv_mesh.points.
        uvc.bv_sep_apex_nodes = np.array(custom_apex_nodes, dtype=int)
        print(f"Using custom LV epi apex face nodes for longitudinal BCs: {uvc.bv_sep_apex_nodes}")
    else:
        # Fall back to whatever was defined inside UVC.define_apex_nodes()
        if hasattr(uvc, 'bv_sep_apex_nodes'):
            print("Keeping default bv_sep_apex_nodes defined by UVC.define_apex_nodes()")
        else:
            print("Warning: UVC object has no bv_sep_apex_nodes; longitudinal BCs will not use point constraints at apex.")
    print('Computing Transmural')
    lv_trans, rv_trans, epi_trans = mcg.run_transmural(uvc, method='laplace')
    print('Computing Longitudinal')
    long = mcg.run_longitudinal(uvc)
    # Store pre-correction longitudinal coordinate
    if 'long' in uvc.bv_mesh.point_data:
        uvc.bv_mesh.point_data['long-pre'] = np.copy(uvc.bv_mesh.point_data['long'])
    if hasattr(uvc, 'lv_mesh') and 'long' in uvc.lv_mesh.point_data:
        uvc.lv_mesh.point_data['long-pre'] = np.copy(uvc.lv_mesh.point_data['long'])
    if hasattr(uvc, 'rv_mesh') and 'long' in uvc.rv_mesh.point_data:
        uvc.rv_mesh.point_data['long-pre'] = np.copy(uvc.rv_mesh.point_data['long'])
    if mcg.mmg:
        print('Correcting Longitudinal')
        long = mcg.correct_longitudinal(uvc)
    uvc.merge_lv_rv_point_data(['long'])
    uvc.define_septum_nodes()
    print('Computing Circumferential')
    lv_circ, rv_circ, pv_circ = mcg.run_circumferential(uvc)
    
    print('Computing RV Lateral (rv_lat) using angular method...')
    rv_lat = mcg.run_rv_lateral_angular(uvc)
    
    uvc.merge_rv_lat_to_bv()
    
    print('Postprocessing ')
    # uvc.rv_mesh.point_data['long'] = uvc.bv_mesh.point_data['long'][uvc.map_rv_bv]
    mcg.get_local_vectors(uvc, which='lv')
    mcg.get_local_vectors(uvc, which='rv')
    
    print('Computing AHA segments')
    uvc.compute_aha_segments(aha_type='points')
    uvc.compute_aha_segments(aha_type='elems')
    
    # Collecting results into BV mesh
    uvc.merge_lv_rv_point_data([
        'circ', 'trans', 'circ_aux', 'long', 'long-pre', 'long_grad_curve',
        'pv_circ', 'pv_circ_base', 'pv_circ_sign',
        'aha', 'eC', 'eL', 'eT'
    ])
    uvc.merge_lv_rv_cell_data(['aha'])
    
    import meshio as io
    io.write(mesh_folder + 'lv_mesh.vtu', uvc.lv_mesh)
    io.write(mesh_folder + 'rv_mesh.vtu', uvc.rv_mesh)
    io.write(mesh_folder + 'bv_mesh.vtu', uvc.bv_mesh)
    
    export_info(uvc)
    export_cheart_inputs(uvc)
    
    print(f"\nUVC generation complete!")
    print(f"Output files saved in: {mesh_folder}")
    print(f"  - lv_mesh.vtu")
    print(f"  - rv_mesh.vtu")
    print(f"  - bv_mesh.vtu")
    print(f"  - boundaries.P")
    print(f"  - Various .FE files with coordinates and normals")
