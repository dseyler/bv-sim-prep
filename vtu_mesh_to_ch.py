#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert mesh folders to CH format.

Takes a mesh folder containing mesh-complete.mesh.vtu and mesh-complete.exterior.vtp,
combines with rv_lv_junction.vtp from mesh-surfaces/, and generates CH format files.
"""

import argparse
import numpy as np
from pathlib import Path
import meshio as io
import cheartio as chio

try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
except ImportError:
    try:
        # Try vtkmodules (newer VTK installations)
        from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
        from vtkmodules.util.numpy_support import vtk_to_numpy
        # Create a mock vtk module for compatibility
        class VTKModule:
            VTK_TRIANGLE = 5
            def vtkXMLPolyDataReader(self):
                return vtkXMLPolyDataReader()
        vtk = VTKModule()
        vtk.vtkXMLPolyDataReader = vtkXMLPolyDataReader
    except ImportError:
        raise ImportError(
            "VTK is required to read .vtp files. Please install it with: "
            "conda install -c conda-forge vtk or pip install vtk"
        )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert mesh folder to CH format files.",
    )
    parser.add_argument(
        "--mesh-folder",
        type=str,
        required=True,
        help="Path to mesh folder containing mesh-complete.mesh.vtu and mesh-complete.exterior.vtp",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="bv_model_gen",
        help="Base name for output CH files (default: bv_model_gen)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: cheart/ subfolder in mesh folder)",
    )
    return parser.parse_args()


def vtp_to_meshio(vtp_path: Path) -> io.Mesh:
    """Convert VTK PolyData (.vtp) to meshio Mesh.
    
    Args:
        vtp_path: Path to .vtp file
        
    Returns:
        meshio Mesh object with points, cells, and all data arrays
    """
    # Load with VTK
    try:
        # Try standard vtk import
        reader = vtk.vtkXMLPolyDataReader()
    except AttributeError:
        # Fall back to vtkmodules
        from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
        reader = vtkXMLPolyDataReader()
    
    reader.SetFileName(str(vtp_path))
    reader.Update()
    polydata = reader.GetOutput()
    
    if polydata.GetNumberOfPoints() == 0:
        raise ValueError(f"No points found in {vtp_path}")
    if polydata.GetNumberOfCells() == 0:
        raise ValueError(f"No cells found in {vtp_path}")
    
    # Extract points
    points_array = polydata.GetPoints().GetData()
    points = vtk_to_numpy(points_array)
    
    # Extract point data (including GlobalNodeID)
    point_data = {}
    point_data_array = polydata.GetPointData()
    for i in range(point_data_array.GetNumberOfArrays()):
        array = point_data_array.GetArray(i)
        array_name = array.GetName()
        array_data = vtk_to_numpy(array)
        point_data[array_name] = array_data
    
    # Extract cells (triangles)
    # VTK_TRIANGLE constant
    try:
        VTK_TRIANGLE = vtk.VTK_TRIANGLE
    except AttributeError:
        VTK_TRIANGLE = 5  # Standard VTK triangle type
    
    cells_data = []
    for i in range(polydata.GetNumberOfCells()):
        cell = polydata.GetCell(i)
        if cell.GetCellType() == VTK_TRIANGLE:
            if cell.GetNumberOfPoints() != 3:
                raise ValueError(f"Triangle cell {i} has {cell.GetNumberOfPoints()} points, expected 3")
            point_ids = [cell.GetPointId(j) for j in range(3)]
            cells_data.append(point_ids)
        else:
            raise ValueError(f"Cell {i} is not a triangle (type: {cell.GetCellType()})")
    
    if len(cells_data) == 0:
        raise ValueError(f"No triangle cells found in {vtp_path}")
    
    cells_array = np.array(cells_data)
    
    # Extract cell data
    cell_data = {}
    cell_data_array = polydata.GetCellData()
    for i in range(cell_data_array.GetNumberOfArrays()):
        array = cell_data_array.GetArray(i)
        array_name = array.GetName()
        array_data = vtk_to_numpy(array)
        cell_data[array_name] = [array_data]
    
    # Create meshio mesh
    mesh = io.Mesh(
        points=points,
        cells=[io.CellBlock(cell_type="triangle", data=cells_array)],
        point_data=point_data,
        cell_data=cell_data
    )
    
    return mesh


def map_surface_to_volume_points(
    surface_mesh: io.Mesh,
    volume_mesh: io.Mesh,
) -> np.ndarray:
    """Map surface point indices to volume mesh point indices using GlobalNodeID.
    
    Args:
        surface_mesh: Surface mesh with GlobalNodeID point data
        volume_mesh: Volume mesh with GlobalNodeID point data
        
    Returns:
        Array mapping surface point index to volume point index
    """
    # Check for GlobalNodeID in both meshes
    if "GlobalNodeID" not in surface_mesh.point_data:
        raise ValueError("Surface mesh must have GlobalNodeID point data")
    if "GlobalNodeID" not in volume_mesh.point_data:
        raise ValueError("Volume mesh must have GlobalNodeID point data")
    
    # Get GlobalNodeIDs
    surface_global_ids = surface_mesh.point_data["GlobalNodeID"].astype(int)
    volume_global_ids = volume_mesh.point_data["GlobalNodeID"].astype(int)
    
    # Build mapping from GlobalNodeID to volume point index
    global_id_to_volume_idx = {int(gid): idx for idx, gid in enumerate(volume_global_ids)}
    
    # Map surface points to volume points
    volume_indices = np.zeros(len(surface_global_ids), dtype=int)
    unmatched = []
    
    for i, gid in enumerate(surface_global_ids):
        gid_int = int(gid)
        if gid_int in global_id_to_volume_idx:
            volume_indices[i] = global_id_to_volume_idx[gid_int]
        else:
            unmatched.append((i, gid_int))
            # Use -1 as placeholder for unmatched points
            volume_indices[i] = -1
    
    if unmatched:
        raise ValueError(
            f"Failed to map {len(unmatched)} surface points to volume mesh using GlobalNodeID. "
            f"Example unmatched GlobalNodeIDs: {[gid for _, gid in unmatched[:5]]}"
        )
    
    return volume_indices


def load_and_combine_surfaces(
    exterior_path: Path,
    junction_path: Path,
    volume_mesh: io.Mesh,
) -> io.Mesh:
    """Load and combine exterior surface with rv_lv_junction surface.
    
    Args:
        exterior_path: Path to mesh-complete.exterior.vtp
        junction_path: Path to mesh-surfaces/rv_lv_junction.vtp
        volume_mesh: Volume mesh (for mapping surface points to volume indices using GlobalNodeID)
        
    Returns:
        Combined surface mesh with 'patches' cell data array, using volume mesh points
    """
    # Load exterior surface
    exterior_mesh = vtp_to_meshio(exterior_path)
    
    # Check for ModelFaceID in exterior mesh
    if "ModelFaceID" not in exterior_mesh.cell_data:
        raise ValueError(f"Exterior mesh {exterior_path} must have ModelFaceID cell data")
    
    exterior_model_face_ids = exterior_mesh.cell_data["ModelFaceID"][0]
    
    # Get exterior cells
    exterior_cells = exterior_mesh.cells[0].data  # Assuming first cell block is triangles
    
    # Map exterior surface points to volume mesh point indices using GlobalNodeID
    exterior_point_map = map_surface_to_volume_points(exterior_mesh, volume_mesh)
    
    # Remap exterior cell indices to volume point indices
    exterior_cells_mapped = np.array([
        [exterior_point_map[idx] for idx in cell] for cell in exterior_cells
    ])
    
    # Load junction surface
    junction_mesh = vtp_to_meshio(junction_path)
    
    # Check for ModelFaceID in junction mesh
    if "ModelFaceID" not in junction_mesh.cell_data:
        raise ValueError(f"Junction mesh {junction_path} must have ModelFaceID cell data")
    
    junction_model_face_ids = junction_mesh.cell_data["ModelFaceID"][0]
    
    # Verify junction has ModelFaceID=10
    if not np.all(junction_model_face_ids == 10):
        print(f"Warning: Junction mesh has ModelFaceIDs {np.unique(junction_model_face_ids)}, expected all 10")
    
    # Get junction cells
    junction_cells = junction_mesh.cells[0].data  # Assuming first cell block is triangles
    
    # Map junction surface points to volume mesh point indices using GlobalNodeID
    junction_point_map = map_surface_to_volume_points(junction_mesh, volume_mesh)
    
    # Remap junction cell indices to volume point indices
    junction_cells_mapped = np.array([
        [junction_point_map[idx] for idx in cell] for cell in junction_cells
    ])
    
    # Combine cells (both now reference volume mesh points)
    combined_cells = np.vstack([exterior_cells_mapped, junction_cells_mapped])
    
    # Combine ModelFaceID arrays
    combined_patches = np.concatenate([exterior_model_face_ids, junction_model_face_ids])
    
    # Create combined mesh using volume mesh points
    # This is critical: surface cell indices must reference volume mesh points
    # Copy point data (including GlobalNodeID) from volume mesh
    point_data = {}
    if hasattr(volume_mesh, 'point_data') and volume_mesh.point_data:
        for key, value in volume_mesh.point_data.items():
            point_data[key] = value.copy() if hasattr(value, 'copy') else value
    
    combined_mesh = io.Mesh(
        points=volume_mesh.points,  # Use volume mesh points
        cells=[io.CellBlock(cell_type="triangle", data=combined_cells)],
        point_data=point_data,
        cell_data={"patches": [combined_patches]}
    )
    
    return combined_mesh


def generate_region_FE(
    mesh_folder: Path,
    vol_mesh: io.Mesh,
    vol_cells: np.ndarray,
    output_path: Path,
) -> None:
    """Generate region.FE file with LV/RV labels.
    
    Labels volume elements as 0 (LV) or 1 (RV) based on their membership
    in lv.vtu or rv.vtu, ordered by the element sequence in FE.T.
    
    Args:
        mesh_folder: Folder containing lv.vtu and rv.vtu
        vol_mesh: Volume mesh (meshio.Mesh) with GlobalElementID cell data
        vol_cells: Volume cell connectivity array (same order as FE.T)
        output_path: Path to write region.FE file
        
    Raises:
        FileNotFoundError: If lv.vtu or rv.vtu don't exist
        ValueError: If GlobalElementID missing or elements not found
    """
    # Load LV and RV meshes
    lv_path = mesh_folder / "lv.vtu"
    rv_path = mesh_folder / "rv.vtu"
    
    if not lv_path.exists():
        raise FileNotFoundError(f"LV mesh not found: {lv_path}")
    if not rv_path.exists():
        raise FileNotFoundError(f"RV mesh not found: {rv_path}")
    
    print(f"  Loading LV mesh: {lv_path}")
    lv_mesh = io.read(str(lv_path))
    
    print(f"  Loading RV mesh: {rv_path}")
    rv_mesh = io.read(str(rv_path))
    
    # Extract GlobalElementID from LV and RV meshes
    if "GlobalElementID" not in lv_mesh.cell_data:
        raise ValueError(f"LV mesh must have GlobalElementID cell data: {lv_path}")
    if "GlobalElementID" not in rv_mesh.cell_data:
        raise ValueError(f"RV mesh must have GlobalElementID cell data: {rv_path}")
    
    lv_element_ids = lv_mesh.cell_data["GlobalElementID"][0].astype(int)
    rv_element_ids = rv_mesh.cell_data["GlobalElementID"][0].astype(int)
    
    print(f"  LV mesh: {len(lv_element_ids)} elements")
    print(f"  RV mesh: {len(rv_element_ids)} elements")
    
    # Build mapping: GlobalElementID -> region_label (0 for LV, 1 for RV)
    element_id_to_region = {}
    for eid in lv_element_ids:
        element_id_to_region[int(eid)] = 0  # LV
    for eid in rv_element_ids:
        element_id_to_region[int(eid)] = 1  # RV
    
    # Check for GlobalElementID in volume mesh
    if "GlobalElementID" not in vol_mesh.cell_data:
        raise ValueError("Volume mesh must have GlobalElementID cell data")
    
    vol_element_ids = vol_mesh.cell_data["GlobalElementID"][0].astype(int)
    
    if len(vol_element_ids) != len(vol_cells):
        raise ValueError(
            f"Mismatch: {len(vol_element_ids)} GlobalElementIDs but {len(vol_cells)} cells"
        )
    
    # Build region labels array in vol_cells order (same as FE.T)
    region_labels = []
    missing_elements = []
    
    for idx, eid in enumerate(vol_element_ids):
        eid_int = int(eid)
        if eid_int not in element_id_to_region:
            missing_elements.append((idx, eid_int))
        else:
            region_labels.append(element_id_to_region[eid_int])
    
    if missing_elements:
        missing_str = ", ".join([f"cell {idx} (GlobalElementID {eid})" 
                                for idx, eid in missing_elements[:10]])
        if len(missing_elements) > 10:
            missing_str += f", ... ({len(missing_elements)} total)"
        raise ValueError(
            f"Volume elements not found in LV or RV meshes: {missing_str}"
        )
    
    region_labels = np.array(region_labels, dtype=float)
    
    # Write region.FE file
    print(f"  Writing region.FE file: {output_path}")
    with open(output_path, 'w') as f:
        # Header: <num_elements>\t1
        f.write(f"{len(region_labels)}\t1\n")
        
        # One value per line, formatted as in example (12 spaces + 20.15f format)
        for label in region_labels:
            f.write(f"            {label:20.15f}\n")
    
    print(f"  Generated region.FE with {len(region_labels)} elements")
    print(f"    LV elements: {np.sum(region_labels == 0.0)}")
    print(f"    RV elements: {np.sum(region_labels == 1.0)}")


def match_triangles_to_faces_by_globalnodeid(
    surface_triangles: np.ndarray,
    volume_faces: np.ndarray,
    volume_global_node_ids: np.ndarray,
    surface_global_node_ids: np.ndarray,
    nfaces_per_cell: int,
) -> np.ndarray:
    """Match surface triangles to volume faces using GlobalNodeID triplets.
    
    Args:
        surface_triangles: Array of surface triangle point indices [n_triangles, 3]
        volume_faces: Array of volume faces [n_cells, nfaces, 3] (point indices)
        volume_global_node_ids: GlobalNodeID array for volume mesh points
        surface_global_node_ids: GlobalNodeID array for surface mesh points
        nfaces_per_cell: Number of faces per volume cell (e.g., 4 for tetrahedra)
        
    Returns:
        Array of face indices (linear index: cell_idx * nfaces + face_idx) for each triangle,
        or -1 if no match found
        
    Raises:
        ValueError: If a surface triangle matches multiple volume faces (ambiguous match)
    """
    # Build dictionary: (sorted GlobalNodeID triplet) -> list of linear face indices
    # Note: Interior faces appear twice (shared by two cells), which is expected.
    # We'll filter to only exterior faces (appearing exactly once) for matching.
    face_triplet_to_face_indices = {}
    
    for cell_idx in range(volume_faces.shape[0]):
        for face_idx in range(volume_faces.shape[1]):
            # Get point indices for this face
            face_point_indices = volume_faces[cell_idx, face_idx]
            # Convert to GlobalNodeIDs
            face_global_node_ids = tuple(sorted(
                int(volume_global_node_ids[idx]) for idx in face_point_indices
            ))
            
            # Calculate linear face index
            linear_face_idx = cell_idx * nfaces_per_cell + face_idx
            
            # Add to dictionary (allow duplicates - interior faces will appear twice)
            if face_global_node_ids not in face_triplet_to_face_indices:
                face_triplet_to_face_indices[face_global_node_ids] = []
            face_triplet_to_face_indices[face_global_node_ids].append(linear_face_idx)
    
    # Filter to only exterior faces (appearing exactly once)
    # Interior faces appear twice (shared by two cells) and should be ignored
    exterior_face_triplet_to_face_idx = {}
    for triplet, face_indices in face_triplet_to_face_indices.items():
        if len(face_indices) == 1:
            # This is an exterior face (unique)
            exterior_face_triplet_to_face_idx[triplet] = face_indices[0]
        # If len > 1, it's an interior face - skip it
    
    # Match each surface triangle to exterior faces only
    matches = np.full(len(surface_triangles), -1, dtype=int)
    
    for tri_idx, tri in enumerate(surface_triangles):
        # Get GlobalNodeIDs for this triangle
        tri_global_node_ids = tuple(sorted(
            int(surface_global_node_ids[idx]) for idx in tri
        ))
        
        if tri_global_node_ids in exterior_face_triplet_to_face_idx:
            # Found unique match (exterior face)
            matches[tri_idx] = exterior_face_triplet_to_face_idx[tri_global_node_ids]
        # If not found, matches[tri_idx] remains -1
    
    return matches


def main() -> None:
    """Main function."""
    args = parse_arguments()
    
    # Resolve paths
    mesh_folder = Path(args.mesh_folder)
    if not mesh_folder.is_absolute():
        mesh_folder = Path.cwd() / mesh_folder
    mesh_folder = mesh_folder.resolve()
    
    if not mesh_folder.exists():
        raise FileNotFoundError(f"Mesh folder not found: {mesh_folder}")
    
    # Define input file paths
    volume_path = mesh_folder / "mesh-complete.mesh.vtu"
    exterior_path = mesh_folder / "mesh-complete.exterior.vtp"
    junction_path = mesh_folder / "mesh-surfaces" / "rv_lv_junction.vtp"
    
    # Check that required files exist
    if not volume_path.exists():
        raise FileNotFoundError(f"Volume mesh not found: {volume_path}")
    if not exterior_path.exists():
        raise FileNotFoundError(f"Exterior surface not found: {exterior_path}")
    if not junction_path.exists():
        raise FileNotFoundError(f"Junction surface not found: {junction_path}")
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = mesh_folder / "cheart"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load volume mesh
    print(f"Loading volume mesh: {volume_path}")
    vol_mesh = io.read(str(volume_path))
    
    # Extract volume cells (first cell block, typically tetrahedra)
    if len(vol_mesh.cells) == 0:
        raise ValueError("Volume mesh has no cells")
    
    vol_cells = vol_mesh.cells[0].data
    print(f"  Volume mesh: {len(vol_mesh.points)} points, {len(vol_cells)} cells")
    
    # Check for GlobalNodeID in volume mesh
    if "GlobalNodeID" not in vol_mesh.point_data:
        raise ValueError("Volume mesh must have GlobalNodeID point data")
    
    # Load and combine surface meshes
    print(f"Loading exterior surface: {exterior_path}")
    print(f"Loading junction surface: {junction_path}")
    combined_surf_mesh = load_and_combine_surfaces(exterior_path, junction_path, vol_mesh)
    
    print(f"  Combined surface: {len(combined_surf_mesh.points)} points, {len(combined_surf_mesh.cells[0].data)} cells")
    
    # Verify point matching
    print(f"  Volume points: {vol_mesh.points.shape}, Surface points: {combined_surf_mesh.points.shape}")
    
    # Extract patches cell data
    boundaries = combined_surf_mesh.cell_data["patches"][0]
    labels = np.unique(boundaries)
    print(f"  Found {len(labels)} unique patch labels: {labels}")
    
    # Group cells by patch label (same as vtu2ch.py)
    cell_blocks = []
    for l in labels:
        cell_block = combined_surf_mesh.cells[0].data[boundaries == l]
        cell_blocks.append(io.CellBlock(cell_type="triangle", data=cell_block))
    
    # Preserve point data (including GlobalNodeID) when creating new surface mesh
    point_data = {}
    if hasattr(combined_surf_mesh, 'point_data') and combined_surf_mesh.point_data:
        for key, value in combined_surf_mesh.point_data.items():
            point_data[key] = value.copy() if hasattr(value, 'copy') else value
    
    new_surf_mesh = io.Mesh(
        points=combined_surf_mesh.points,
        cells=cell_blocks,
        point_data=point_data
    )
    
    # Generate CH files
    output_base = str(output_dir / args.output_name)
    print(f"\nGenerating CH files with base name: {output_base}")
    
    # Write volume mesh
    print("  Writing volume mesh...")
    chio.write_mesh(output_base, vol_mesh.points, vol_cells)
    
    # Create boundary file
    print("  Creating boundary file...")
    
    # Check for GlobalNodeID in both meshes
    if "GlobalNodeID" not in vol_mesh.point_data:
        raise ValueError("Volume mesh must have GlobalNodeID point data for triplet matching")
    if "GlobalNodeID" not in new_surf_mesh.point_data:
        raise ValueError("Surface mesh must have GlobalNodeID point data for triplet matching")
    
    volume_global_node_ids = vol_mesh.point_data["GlobalNodeID"]
    surface_global_node_ids = new_surf_mesh.point_data["GlobalNodeID"]
    
    # Extract volume faces
    if vol_cells.shape[1] == 4:  # Tetrahedra
        nfaces = 4
        faces = np.zeros([vol_cells.shape[0], 4, 3], dtype=int)
        faces[:,0,:] = np.array([vol_cells[:,0], vol_cells[:,1], vol_cells[:,2]]).T
        faces[:,1,:] = np.array([vol_cells[:,0], vol_cells[:,2], vol_cells[:,3]]).T
        faces[:,2,:] = np.array([vol_cells[:,0], vol_cells[:,3], vol_cells[:,1]]).T
        faces[:,3,:] = np.array([vol_cells[:,1], vol_cells[:,2], vol_cells[:,3]]).T
    else:
        raise ValueError(f"Unsupported cell type with {vol_cells.shape[1]} nodes per cell")
    
    # Build boundary file data using GlobalNodeID triplet matching
    print("  Matching surface triangles to volume faces using GlobalNodeID triplets...")
    bdata = []
    patches = new_surf_mesh.cells
    
    total_matched = 0
    total_unmatched = 0
    
    for i, p in enumerate(patches):
        surf = p.data  # Surface triangles as point indices
        
        # Match surface triangles to volume faces using GlobalNodeID triplets
        matches = match_triangles_to_faces_by_globalnodeid(
            surf,
            faces,
            volume_global_node_ids,
            surface_global_node_ids,
            nfaces
        )
        
        # Find valid matches (matches != -1)
        valid_mask = matches != -1
        valid_matches = matches[valid_mask]
        valid_surf = surf[valid_mask]
        
        matched_count = len(valid_matches)
        unmatched_count = np.sum(~valid_mask)
        total_matched += matched_count
        total_unmatched += unmatched_count
        
        if len(valid_matches) == 0:
            print(f'  Warning: Boundary {i+1} (patch {i+1}) has no valid matches, skipping')
            continue
        
        if unmatched_count > 0:
            print(f'  Warning: Boundary {i+1} (patch {i+1}) has {unmatched_count} unmatched triangles out of {len(surf)} total')
        
        # Build boundary data: [element_index, triangle_node0, triangle_node1, triangle_node2, patch_id]
        # valid_matches are linear face indices (cell_idx * nfaces + face_idx)
        # Convert to element indices: element_idx = linear_face_idx // nfaces
        element_indices = valid_matches // nfaces
        patch_ids = np.ones(len(valid_matches), dtype=int) * (i + 1)
        
        bdata.append(np.vstack([element_indices, valid_surf.T, patch_ids]).T)
    
    print(f"  Matching complete: {total_matched} triangles matched, {total_unmatched} unmatched")
    
    if len(bdata) == 0:
        raise ValueError("No valid boundary data found - all patches had zero matches")
    
    bdata = np.vstack(bdata)
    
    # Write boundary file
    print("  Writing boundary file...")
    chio.write_bfile(output_base, bdata)
    
    # Generate region.FE file
    print("  Generating region.FE file...")
    region_fe_path = output_dir / "region.FE"
    generate_region_FE(mesh_folder, vol_mesh, vol_cells, region_fe_path)
    
    print(f"\nSuccessfully generated CH files in: {output_dir}")
    print(f"  Base name: {args.output_name}")




if __name__ == "__main__":
    main()

