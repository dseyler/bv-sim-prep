#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test script to generate rv_lat coordinate only.
Similar to generate_uvc.py but focused only on rv_lat computation.
"""
import sys
import argparse
import os
import numpy as np
import cheartio as chio
from uvcgen.model_coords import UVCGen
from uvcgen.UVC import UVC
import meshio as io


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate rv_lat coordinate from CH format mesh"
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
        "--threshold-septum",
        type=float,
        default=0.6,
        help="Threshold for septum identification (default: 0.6)",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    
    # Setup paths
    mesh_folder = os.path.abspath(args.mesh_folder)
    if not mesh_folder.endswith('/'):
        mesh_folder += '/'
    mesh_path = os.path.join(mesh_folder, args.mesh_name)
    
    # Load boundaries
    boundaries_path = os.path.join(mesh_folder, 'boundaries.P')
    if not os.path.exists(boundaries_path):
        raise FileNotFoundError(f"boundaries.P not found: {boundaries_path}")
    
    print(f"Loading boundaries from {boundaries_path}")
    boundaries = chio.pfile_to_dict(boundaries_path, fmt=int)
    
    thresholds = {
        'septum': args.threshold_septum,
        'long': 1.0
    }
    
    # Read CH mesh
    print('Loading mesh...')
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
    
    # Check if AV exists
    av_patch_id = boundaries.get('av', 6)
    has_av = av_patch_id in bdata[:, -1]
    if not has_av:
        print("AV (aortic valve) not found in boundary data - will use MV (mitral valve) instead")
        if 'av' in boundaries:
            del boundaries['av']
    else:
        print(f"AV (aortic valve) found in boundary data (patch ID {av_patch_id})")
    
    # Load region split file if provided
    rvlv = None
    if args.region_file:
        if os.path.exists(args.region_file):
            try:
                rvlv = 1 - chio.read_dfile(args.region_file)  # This file is 0 in LV, 1 in RV
                print(f"Loaded region split from {args.region_file}")
                print(f"  LV elements: {np.sum(rvlv == 0)}, RV elements: {np.sum(rvlv == 1)}")
            except Exception as e:
                print(f"Warning: Could not load region file: {e}")
                rvlv = None
        else:
            print(f"Warning: Region file not found: {args.region_file}")
    else:
        # Check for region.FE in mesh folder
        cheart_region_path = os.path.join(mesh_folder, 'cheart', 'region.FE')
        root_region_path = os.path.join(mesh_folder, 'region.FE')
        
        if os.path.exists(cheart_region_path):
            try:
                rvlv = 1 - chio.read_dfile(cheart_region_path)
                print(f"Found and loaded region.FE from cheart/ subdirectory")
                print(f"  LV elements: {np.sum(rvlv == 0)}, RV elements: {np.sum(rvlv == 1)}")
            except Exception as e:
                print(f"Warning: Could not load region file: {e}")
                rvlv = None
        elif os.path.exists(root_region_path):
            try:
                rvlv = 1 - chio.read_dfile(root_region_path)
                print(f"Found and loaded region.FE from mesh folder root")
                print(f"  LV elements: {np.sum(rvlv == 0)}, RV elements: {np.sum(rvlv == 1)}")
            except Exception as e:
                print(f"Warning: Could not load region file: {e}")
                rvlv = None
    
    # Initialize classes
    print('Initializing UVC...')
    uvc = UVC(bv_mesh, bdata, boundaries, thresholds, mesh_folder, rvlv=rvlv)
    mcg = UVCGen(uvc, mmg=True)
    
    # Determine how to get septum (minimal - just enough to split RV/LV)
    print('Computing septum...')
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
                # Mark LV points
                lv_patches = [boundaries.get('lv_endo', 1), boundaries.get('lv_epi', 3), 
                             boundaries.get('mv', 9)]
                if 'av' in boundaries:
                    lv_patches.append(boundaries['av'])
                for patch in lv_patches:
                    if patch in bdata[:, -1]:
                        lv_faces = bdata[bdata[:, -1] == patch]
                        lv_points = np.unique(lv_faces[:, 1:-1].flatten()) - 1
                        septum[lv_points] = 0.0
                # Mark RV points
                rv_patches = [boundaries.get('rv_endo', 2), boundaries.get('rv_epi', 4),
                             boundaries.get('tv', 8), boundaries.get('pv', 7)]
                for patch in rv_patches:
                    if patch in bdata[:, -1]:
                        rv_faces = bdata[bdata[:, -1] == patch]
                        rv_points = np.unique(rv_faces[:, 1:-1].flatten()) - 1
                        septum[rv_points] = 1.0
                print(f"    Extracted {np.sum(septum == 0.5)} septum points from boundary data")
                uvc.bv_mesh.point_data['septum'] = septum
            else:
                print("  Warning: No septum faces found in boundary data, solving Laplace for septum")
                septum = mcg.run_septum(uvc)
        else:
            print("  Warning: No rv_septum in boundaries, solving Laplace for septum")
            septum = mcg.run_septum(uvc)
    
    # Compute long_plane_coord and split RV/LV (needed for rv_lat)
    print('Computing long_plane_coord and splitting RV/LV...')
    uvc.compute_long_plane_coord(septum)
    uvc.split_rv_lv(septum)
    
    # Define apex nodes (needed for rv_lat angular method)
    print('Defining apex nodes...')
    uvc.define_apex_nodes()
    
    # Compute transmural coordinate (needed before longitudinal)
    print('Computing Transmural...')
    lv_trans, rv_trans, epi_trans = mcg.run_transmural(uvc, method='laplace')
    
    # Compute longitudinal coordinate (needed for rv_lat)
    print('Computing Longitudinal...')
    long = mcg.run_longitudinal(uvc)
    uvc.merge_lv_rv_point_data(['long'])
    
    # Define septum nodes (needed for rv_circ1 computation)
    print('Defining septum nodes...')
    uvc.define_septum_nodes()
    
    # Compute rv_circ1 (needed for blending in rv_lat)
    print('Computing rv_circ1 (required for rv_lat blending)...')
    rv_circ1 = mcg.run_rv_circumferential1(uvc, method='laplace')
    
    # Compute rv_lat using angular method
    print('Computing RV Lateral (rv_lat) using angular method...')
    rv_lat = mcg.run_rv_lateral_angular(uvc)
    uvc.merge_rv_lat_to_bv()
    
    # Save output
    print('Saving output...')
    output_folder = mesh_folder
    io.write(output_folder + 'rv_mesh_lat.vtu', uvc.rv_mesh)
    io.write(output_folder + 'bv_mesh_lat.vtu', uvc.bv_mesh)
    
    print(f"\nrv_lat generation complete!")
    print(f"Output files saved in: {output_folder}")
    print(f"  - rv_mesh_lat.vtu (contains rv_lat in point_data)")
    print(f"  - bv_mesh_lat.vtu (contains rv_lat in point_data, LV region = 0)")

