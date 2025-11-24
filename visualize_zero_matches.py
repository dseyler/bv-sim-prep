#!/usr/bin/env python3
"""
Visualize triangles with zero matches from create_bfile.

This script loads the surface and volume meshes, identifies triangles
with zero matches (from query_ball_tree), and visualizes them along
with a transparent volume mesh.
"""

import argparse
import numpy as np
import meshio as io
from pathlib import Path
from scipy.spatial import KDTree

# Try to import PyVista, with helpful error message if it fails
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except (ImportError, AttributeError) as e:
    PYVISTA_AVAILABLE = False
    PYVISTA_ERROR = str(e)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize triangles with zero matches from create_bfile.",
    )
    parser.add_argument(
        "--volume-mesh",
        type=str,
        default="bv_model.vtu",
        help="Path to volume mesh (.vtu file)",
    )
    parser.add_argument(
        "--surface-mesh",
        type=str,
        default="bv_model_b.vtu",
        help="Path to surface mesh (.vtu file)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Tolerance for query_ball_tree (default: 1e-5, only used with --use-kdtree)",
    )
    parser.add_argument(
        "--use-kdtree",
        action="store_true",
        help="Use KDTree spatial matching instead of direct node matching",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image file (optional, if not provided shows interactive window)",
    )
    parser.add_argument(
        "--save-vtp",
        type=str,
        default=None,
        help="Save zero-match triangles to .vtp file (for ParaView visualization)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization, only compute and print statistics",
    )
    return parser.parse_args()


def match_triangles_to_faces_direct(surface_triangles, volume_faces, nfaces_per_cell):
    """Match surface triangles to volume faces by exact node set comparison.
    
    Args:
        surface_triangles: Array of surface triangle node indices [n_triangles, 3]
        volume_faces: Array of volume faces [n_cells, nfaces, 3]
        nfaces_per_cell: Number of faces per volume cell (e.g., 4 for tetrahedra)
        
    Returns:
        Array of face indices (linear index: cell_idx * nfaces + face_idx) for each triangle,
        or -1 if no match found
    """
    # Build a dictionary: (sorted node tuple) -> (cell_idx, face_idx)
    face_node_to_cell_face = {}
    for cell_idx in range(volume_faces.shape[0]):
        for face_idx in range(volume_faces.shape[1]):
            face_nodes = tuple(sorted(volume_faces[cell_idx, face_idx]))
            # Convert to linear face index: cell_idx * nfaces + face_idx
            linear_face_idx = cell_idx * nfaces_per_cell + face_idx
            face_node_to_cell_face[face_nodes] = linear_face_idx
    
    # Match each surface triangle
    matches = np.full(len(surface_triangles), -1, dtype=int)
    for tri_idx, tri in enumerate(surface_triangles):
        tri_nodes = tuple(sorted(tri))
        if tri_nodes in face_node_to_cell_face:
            matches[tri_idx] = face_node_to_cell_face[tri_nodes]
    
    return matches


def find_zero_match_triangles(surface_mesh, volume_mesh, tolerance=1e-5, use_direct_matching=True):
    """Find surface triangles with zero matches in volume mesh.
    
    Args:
        surface_mesh: Surface meshio mesh
        volume_mesh: Volume meshio mesh
        tolerance: Tolerance for query_ball_tree (only used if use_direct_matching=False)
        use_direct_matching: If True, use direct node set matching; if False, use KDTree
        
    Returns:
        List of triangle indices with zero matches, and the full correspondence list
    """
    # Extract surface triangles
    if len(surface_mesh.cells) == 0:
        raise ValueError("Surface mesh has no cells")
    
    # Get all triangles from all cell blocks
    all_triangles = []
    triangle_block_indices = []  # Track which block each triangle belongs to
    
    for block_idx, cell_block in enumerate(surface_mesh.cells):
        if cell_block.type == "triangle":
            triangles = cell_block.data
            all_triangles.append(triangles)
            triangle_block_indices.extend([block_idx] * len(triangles))
    
    if len(all_triangles) == 0:
        raise ValueError("No triangle cells found in surface mesh")
    
    all_triangles = np.vstack(all_triangles)
    points = surface_mesh.points
    
    # Extract volume mesh faces
    if len(volume_mesh.cells) == 0:
        raise ValueError("Volume mesh has no cells")
    
    vol_cells = volume_mesh.cells[0].data  # Assuming first block is volume cells
    
    # Determine number of faces per cell
    if vol_cells.shape[1] == 4:  # Tetrahedra
        nfaces = 4
        faces = np.zeros([vol_cells.shape[0], 4, 3], dtype=int)
        faces[:,0,:] = np.array([vol_cells[:,0], vol_cells[:,1], vol_cells[:,2]]).T
        faces[:,1,:] = np.array([vol_cells[:,0], vol_cells[:,2], vol_cells[:,3]]).T
        faces[:,2,:] = np.array([vol_cells[:,0], vol_cells[:,3], vol_cells[:,1]]).T
        faces[:,3,:] = np.array([vol_cells[:,1], vol_cells[:,2], vol_cells[:,3]]).T
    else:
        raise ValueError(f"Unsupported cell type with {vol_cells.shape[1]} nodes per cell")
    
    vol_points = volume_mesh.points
    
    if use_direct_matching:
        print("Using direct node set matching...")
        # Direct node set matching
        matches = match_triangles_to_faces_direct(all_triangles, faces, nfaces)
        
        # Find triangles with zero matches (matches == -1)
        zero_match_indices = np.where(matches == -1)[0].tolist()
        
        # Convert matches to list format for compatibility
        # Each match is a single face index (or -1)
        corr = [[m] if m != -1 else [] for m in matches]
        
        print(f"\nDirect matching results:")
        print(f"  Total triangles: {len(all_triangles)}")
        print(f"  Matched triangles: {np.sum(matches != -1)}")
        print(f"  Zero-match triangles: {len(zero_match_indices)}")
        
        # Verify that surface and volume use same points
        if not np.array_equal(points, vol_points):
            print(f"  WARNING: Surface and volume meshes use different point arrays!")
            print(f"    Surface points shape: {points.shape}")
            print(f"    Volume points shape: {vol_points.shape}")
            print(f"    This may cause matching failures.")
        else:
            print(f"  Surface and volume meshes use same point array (good for direct matching)")
        
    else:
        print("Using KDTree spatial matching...")
        # Original KDTree approach
        # Compute centroids of surface triangles
        ctri = np.mean(points[all_triangles], axis=1)
        
        # Compute centroids of volume faces
        vertex = vol_points[faces]
        ctet = np.mean(vertex, axis=2).reshape([-1, 3])
        
        # Build KD-trees
        tree1 = KDTree(ctet)  # Volume face centroids
        tree2 = KDTree(ctri)  # Surface triangle centroids
        
        # Find correspondences
        corr = tree2.query_ball_tree(tree1, tolerance)
        
        # Find triangles with zero matches
        zero_match_indices = [i for i, matches in enumerate(corr) if len(matches) == 0]
        
        print(f"\nKDTree matching results:")
        print(f"  Total triangles: {len(corr)}")
        match_lengths = [len(matches) for matches in corr]
        unique_lengths, counts = np.unique(match_lengths, return_counts=True)
        print(f"  Match length distribution:")
        for length, count in zip(unique_lengths, counts):
            print(f"    {length} matches: {count} triangles")
    
    print(f"\nFound {len(zero_match_indices)} triangles with zero matches out of {len(all_triangles)} total triangles")
    
    return zero_match_indices, all_triangles, corr


def save_zero_matches_to_vtp(zero_match_triangles, points, output_path):
    """Save zero-match triangles to a .vtp file for ParaView visualization."""
    try:
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk
    except ImportError:
        try:
            from vtkmodules.vtkCommonCore import vtkPoints, vtkCellArray
            from vtkmodules.vtkCommonDataModel import vtkPolyData
            from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter
            from vtkmodules.util.numpy_support import numpy_to_vtk
            vtk = type('vtk', (), {
                'vtkPoints': vtkPoints,
                'vtkCellArray': vtkCellArray,
                'vtkPolyData': vtkPolyData,
                'vtkXMLPolyDataWriter': vtkXMLPolyDataWriter,
            })()
        except ImportError:
            raise ImportError("VTK is required to save .vtp files")
    
    # Create VTK points
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(points, deep=True))
    
    # Create VTK cells
    vtk_cells = vtk.vtkCellArray()
    for triangle in zero_match_triangles:
        vtk_cells.InsertNextCell(3)
        for vertex_idx in triangle:
            vtk_cells.InsertCellPoint(int(vertex_idx))
    
    # Create PolyData
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetPolys(vtk_cells)
    
    # Write to file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputData(polydata)
    writer.Write()
    
    print(f"Saved zero-match triangles to: {output_path}")


def visualize_zero_matches(volume_mesh_path, surface_mesh_path, tolerance=1e-5, output=None, save_vtp=None, no_viz=False, use_direct_matching=True):
    """Visualize triangles with zero matches."""
    if not no_viz and not PYVISTA_AVAILABLE:
        print(f"WARNING: PyVista is not available due to: {PYVISTA_ERROR}")
        print("\nThis is likely a Python version compatibility issue (PyVista may not work with Python 3.14).")
        print("Skipping visualization. Use --save-vtp to save triangles for ParaView visualization.")
        no_viz = True
    
    # Load meshes
    print(f"Loading volume mesh: {volume_mesh_path}")
    volume_mesh = io.read(str(volume_mesh_path))
    
    print(f"Loading surface mesh: {surface_mesh_path}")
    surface_mesh = io.read(str(surface_mesh_path))
    
    if not no_viz and PYVISTA_AVAILABLE:
        vol_pv = pv.read(str(volume_mesh_path))
        surf_pv = pv.read(str(surface_mesh_path))
    
    # Find zero match triangles
    zero_match_indices, all_triangles, corr = find_zero_match_triangles(
        surface_mesh, volume_mesh, tolerance, use_direct_matching=use_direct_matching
    )
    
    if len(zero_match_indices) == 0:
        print("No triangles with zero matches found!")
        return
    
    # Create a mesh with only zero-match triangles
    zero_match_triangles = all_triangles[zero_match_indices]
    
    # Save to .vtp file if requested
    if save_vtp:
        save_zero_matches_to_vtp(zero_match_triangles, surface_mesh.points, save_vtp)
    
    # Visualize if PyVista is available and not disabled
    if not no_viz and PYVISTA_AVAILABLE:
        # Create PyVista PolyData for zero-match triangles
        # PyVista expects faces in format: [n, v0, v1, v2, n, v0, v1, v2, ...]
        faces_array = np.column_stack([
            np.full(len(zero_match_triangles), 3),
            zero_match_triangles
        ]).flatten().astype(np.int32)
        
        zero_match_surf = pv.PolyData(
            surface_mesh.points,
            faces=faces_array
        )
        
        # Create plotter
        plotter = pv.Plotter(off_screen=output is not None)
        
        # Add volume mesh (transparent)
        plotter.add_mesh(
            vol_pv,
            opacity=0.2,
            color="lightblue",
            show_edges=False,
            label="Volume mesh"
        )
        
        # Add all surface mesh (semi-transparent, gray)
        plotter.add_mesh(
            surf_pv,
            opacity=0.3,
            color="gray",
            show_edges=True,
            edge_color="black",
            label="All surface triangles"
        )
        
        # Add zero-match triangles (opaque, red)
        plotter.add_mesh(
            zero_match_surf,
            opacity=1.0,
            color="red",
            show_edges=True,
            edge_color="darkred",
            line_width=2,
            label=f"Zero-match triangles ({len(zero_match_indices)})"
        )
        
        # Add legend
        plotter.add_legend()
        
        # Set title
        plotter.add_text(
            f"Triangles with Zero Matches (tolerance={tolerance})",
            font_size=12
        )
        
        # Show or save
        if output:
            print(f"Saving visualization to: {output}")
            plotter.screenshot(output)
            plotter.close()
        else:
            print("Showing interactive visualization...")
            plotter.show()
    elif not no_viz:
        print("\nVisualization skipped due to PyVista unavailability.")
        print("Use --save-vtp to save triangles for visualization in ParaView.")


def main():
    """Main function."""
    args = parse_arguments()
    
    # Resolve paths
    volume_path = Path(args.volume_mesh)
    surface_path = Path(args.surface_mesh)
    
    if not volume_path.is_absolute():
        volume_path = Path.cwd() / volume_path
    if not surface_path.is_absolute():
        surface_path = Path.cwd() / surface_path
    
    volume_path = volume_path.resolve()
    surface_path = surface_path.resolve()
    
    if not volume_path.exists():
        raise FileNotFoundError(f"Volume mesh not found: {volume_path}")
    if not surface_path.exists():
        raise FileNotFoundError(f"Surface mesh not found: {surface_path}")
    
    visualize_zero_matches(
        volume_path,
        surface_path,
        tolerance=args.tolerance,
        output=args.output,
        save_vtp=args.save_vtp,
        no_viz=args.no_viz,
        use_direct_matching=not args.use_kdtree
    )


if __name__ == "__main__":
    main()

