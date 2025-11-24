#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert mesh directory to bv_model.vtu and bv_model_b.vtu pair.

Takes a mesh folder containing mesh-complete.mesh.vtu, mesh-complete.exterior.vtp,
and mesh-surfaces/rv_lv_junction.vtp, and generates:
- bv_model.vtu: Exact copy of mesh-complete.mesh.vtu
- bv_model_b.vtu: Merged surface mesh with ModelFaceID renamed to 'patches'
"""

import argparse
import numpy as np
from pathlib import Path
import meshio as io

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
        description="Convert mesh directory to bv_model.vtu and bv_model_b.vtu pair.",
    )
    parser.add_argument(
        "--mesh-folder",
        type=str,
        default="/Users/dseyler/Documents/Marsden_Lab/cDTI_processing/Meshes/in_vivo_1d5mm_002-mesh-complete_sept_valves",
        help="Path to mesh folder containing mesh-complete.mesh.vtu and mesh-complete.exterior.vtp",
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
            volume_indices[i] = -1
    
    if unmatched:
        raise ValueError(
            f"Failed to map {len(unmatched)} surface points to volume mesh using GlobalNodeID. "
            f"Example unmatched GlobalNodeIDs: {[gid for _, gid in unmatched[:5]]}"
        )
    
    return volume_indices


def merge_surfaces(
    exterior_mesh: io.Mesh,
    junction_mesh: io.Mesh,
    volume_mesh: io.Mesh,
) -> io.Mesh:
    """Merge exterior and junction surfaces into a single mesh.
    
    Maps surface points to volume mesh points using GlobalNodeID, then combines
    cells and renames ModelFaceID to 'patches'. The output uses volume mesh points
    so that cell indices reference the volume mesh point array.
    
    Args:
        exterior_mesh: Exterior surface mesh with ModelFaceID cell data
        junction_mesh: Junction surface mesh with ModelFaceID cell data
        volume_mesh: Volume mesh (for point mapping)
        
    Returns:
        Combined surface mesh with 'patches' cell data, using volume mesh points
    """
    # Check for ModelFaceID in both meshes
    if "ModelFaceID" not in exterior_mesh.cell_data:
        raise ValueError("Exterior mesh must have ModelFaceID cell data")
    if "ModelFaceID" not in junction_mesh.cell_data:
        raise ValueError("Junction mesh must have ModelFaceID cell data")
    
    # Get ModelFaceID arrays
    exterior_model_face_ids = exterior_mesh.cell_data["ModelFaceID"][0]
    junction_model_face_ids = junction_mesh.cell_data["ModelFaceID"][0]
    
    # Get cells
    exterior_cells = exterior_mesh.cells[0].data
    junction_cells = junction_mesh.cells[0].data
    
    # Map surface points to volume points using GlobalNodeID
    exterior_point_map = map_surface_to_volume_points(exterior_mesh, volume_mesh)
    junction_point_map = map_surface_to_volume_points(junction_mesh, volume_mesh)
    
    # Remap cell indices to volume point indices
    exterior_cells_mapped = np.array([
        [exterior_point_map[idx] for idx in cell] for cell in exterior_cells
    ])
    junction_cells_mapped = np.array([
        [junction_point_map[idx] for idx in cell] for cell in junction_cells
    ])
    
    # Combine cells (both now reference volume mesh points)
    combined_cells = np.vstack([exterior_cells_mapped, junction_cells_mapped])
    
    # Combine ModelFaceID arrays and rename to 'patches'
    combined_patches = np.concatenate([exterior_model_face_ids, junction_model_face_ids])
    
    # Create combined mesh using volume mesh points
    # This is critical: surface cell indices must reference volume mesh points
    combined_mesh = io.Mesh(
        points=volume_mesh.points,  # Use volume mesh points
        cells=[io.CellBlock(cell_type="triangle", data=combined_cells)],
        cell_data={"patches": [combined_patches]}
    )
    
    return combined_mesh


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
    
    # Define output file paths (in same directory as input)
    output_volume_path = mesh_folder / "bv_model.vtu"
    output_surface_path = mesh_folder / "bv_model_b.vtu"
    
    print(f"Loading volume mesh: {volume_path}")
    volume_mesh = io.read(str(volume_path))
    n_volume_cells = sum(len(cell_block.data) for cell_block in volume_mesh.cells)
    print(f"  Volume mesh: {volume_mesh.points.shape[0]} points, {n_volume_cells} cells")
    
    print(f"Copying volume mesh to: {output_volume_path}")
    io.write(str(output_volume_path), volume_mesh)
    print(f"  Saved: {output_volume_path}")
    
    print(f"Loading exterior surface: {exterior_path}")
    exterior_mesh = vtp_to_meshio(exterior_path)
    print(f"  Exterior surface: {exterior_mesh.points.shape[0]} points, {len(exterior_mesh.cells[0].data)} cells")
    
    print(f"Loading junction surface: {junction_path}")
    junction_mesh = vtp_to_meshio(junction_path)
    print(f"  Junction surface: {junction_mesh.points.shape[0]} points, {len(junction_mesh.cells[0].data)} cells")
    
    print("Merging surfaces...")
    combined_surface_mesh = merge_surfaces(exterior_mesh, junction_mesh, volume_mesh)
    print(f"  Combined surface: {combined_surface_mesh.points.shape[0]} points, {len(combined_surface_mesh.cells[0].data)} cells")
    print(f"  Note: Surface mesh uses volume mesh points for compatibility with vtu2ch.py")
    
    # Verify patches data
    if "patches" in combined_surface_mesh.cell_data:
        patches = combined_surface_mesh.cell_data["patches"][0]
        unique_patches = np.unique(patches)
        print(f"  Patches (ModelFaceID): {unique_patches}")
    
    print(f"Saving merged surface mesh to: {output_surface_path}")
    io.write(str(output_surface_path), combined_surface_mesh)
    print(f"  Saved: {output_surface_path}")
    
    print("\nDone!")
    print(f"  Output files:")
    print(f"    {output_volume_path}")
    print(f"    {output_surface_path}")


if __name__ == "__main__":
    main()
