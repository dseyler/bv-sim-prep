#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 14:49:36 2023

@author: Javiera Jilberto Vallejos
"""

import numpy as np
import meshio as io

# TODO Add verbose option to disable warnings
verbose = True


vol_to_surf_elem = {'hexahedron': 'quad',
                    'hexahedron27': 'quad9',
                    'triangle': 'line',
                    'tetra': 'triangle',
                    'tetra10': 'triangle6'}


# load CHeart files
def read_mesh(path, meshio=False, element=None, tfile=None, xfile=None):
    # Load mesh
    if xfile is None:
        xyz = np.loadtxt(path + '_FE.X', skiprows = 1)
    else:
        xyz = np.loadtxt(path + xfile, skiprows = 1)
    if tfile is None:
        ien = np.loadtxt(path + '_FE.T', skiprows = 1, dtype=int) - 1
    else:
        ien = np.loadtxt(path + tfile, skiprows = 1, dtype=int) - 1
    try: bfile = np.loadtxt(path + '_FE.B', skiprows = 1)
    except: bfile = np.array([])

    ien, element = get_element_type(ien, element=element, bfile=bfile)

    if meshio:
        return io.Mesh(xyz, {element: ien})
    else:
        return xyz, ien, element
    

def read_bfile(path, element=None):
    array = np.loadtxt(path + '_FE.B', skiprows = 1, dtype=int)
    array[:,0:-1] = array[:,0:-1] - 1
    if element is None:
        _, _, element = read_mesh(path)
    array[:,1:-1] = connectivity_CH2vtu(vol_to_surf_elem[element], array[:,1:-1])    # Correct order to vtu order
    return array


def read_fibers(path, append2d=False):
    fibers = np.loadtxt(path, skiprows = 1)

    if fibers.shape[1] == 9:
        f = fibers[:,0:3]
        s = fibers[:,3:6]
        n = fibers[:,6:9]
        return f, s, n
    elif fibers.shape[1] == 4:
        f = fibers[:,0:2]
        s = fibers[:,2:4]
        if append2d:
            f = np.vstack([f.T, np.zeros(f.shape[0])]).T
            s = np.vstack([s.T, np.zeros(s.shape[0])]).T
        return f, s


def read_dfile(path, **kwargs):
    array = np.loadtxt(path, skiprows = 1, ndmin=1, **kwargs)

    return array


def read_scalar_dfiles(path, times, return_incomplete=False):
    st, et, inc = times
    et += 1
    t = np.arange(st, et, inc)
    nts = len(t)

    array = np.zeros(nts)
    incomplete = False
    for i, cont in enumerate(range(st, et, inc)):
        try:
            array[i] = read_dfile(path + '-%i' % cont + '.D')
        except:
            lts = cont
            incomplete = True
            break

    array = np.array(array)
    if return_incomplete:
        return array
    else:
        if not incomplete:
            return array
        else:
            raise Exception('Missing file: ' +  path + '-%i' % lts + '.D')


def get_element_type(ien, element=None, ch2vtu=True, bfile=np.array([])):
    if element == None:
        element = get_element_type_by_nnodes(ien, bfile)

    if ch2vtu:
        ien = connectivity_CH2vtu(element, ien)
    else:
        ien = connectivity_vtu2CH(element, ien)

    return ien, element


def connectivity_CH2vtu(element, ien):
    # CH to vtu node numeration
    if element == 'line3':
        ien = ien[:, np.array([0, 2, 1])]
    elif element == 'triangle6':
        ien = ien[:, np.array([0, 1, 2, 3, 5, 4])]
    elif element == 'quad':
        ien = ien[:, np.array([0, 1, 3, 2])]
    elif element == 'quad9':
        ien = ien[:, np.array([0, 1, 3, 2,
                               4, 7, 8, 5,
                               6])]
        # ien = ien[:, np.array([0, 2, 8, 6,
        #                        1, 5, 7, 3,
        #                        4])]
    elif element == 'hexahedron':
        ien = ien[:, np.array([0, 1, 3, 2,
                               4, 5, 7, 6])]

    elif element == 'tetra10':
        ien = ien[:, np.array([0, 1, 2, 3, 4, 6, 5, 7, 8, 9])]
    elif element == 'hexahedron27':
        ien = ien[:, np.array([0, 1, 3, 2,
                               4, 5, 7, 6,
                               8, 11, 12, 9,
                               22, 25, 26, 23,
                               13, 15, 21, 19,
                               16, 18, 14, 20,
                               10, 24, 17])]

    return ien

def face_array(element):
    # This returns the array that return the faces of an element
    if element == 'triangle':
        array = np.array([[0,1],[1,2],[2,0]])
    elif element == 'tetra':
        array = np.array([[0,1,2],[1,2,3],[0,2,3],[0,1,3]])
    else:
        raise 'Not Implemented'
    return array




def get_element_type_by_nnodes(ien, bfile=np.array([])):
    if len(ien.shape) == 1:
        return 'point'
    if ien.shape[1] == 2:
        element = 'line'
    elif ien.shape[1] == 3:
        if bfile.size == 0:
            element = 'triangle'
            if verbose:
                print('WARNING: No .Bfile, ambiguous number of nodes, choosing triangle')
        else:
            if bfile.shape[1] == 3:
                element = 'line3'
            elif bfile.shape[1] == 4:
                element = 'triangle'
            else:
                if verbose:
                    print('WARNING: Something wrong with .Bfile, ambiguous number of nodes, choosing triangle')
                element = 'triangle'
    elif ien.shape[1] == 6:
        element = 'triangle6'
        ien = ien[:, np.array([0, 1, 2, 3, 5, 4])]
    elif ien.shape[1] == 4:     # TODO how do I handle this?
        if bfile.size == 0:
            element = 'tetra'
            if verbose:
                print('WARNING: No .Bfile, ambiguous number of nodes, choosing tetra')
        else:
            if bfile.shape[1] == 4:
                element = 'quad'
            elif bfile.shape[1] == 5:
                element = 'tetra'
            else:
                element = 'tetra'
                if verbose:
                    print('WARNING: Something wrong with .Bfile, ambiguous number of nodes, choosing tetra')
    elif ien.shape[1] == 9:
        element = 'quad9'
    elif ien.shape[1] == 8:
        element = 'hexahedron'
    elif ien.shape[1] == 10:
        element = 'tetra10'
    elif ien.shape[1] == 27:
        element = 'hexahedron27'
    else:
        print('WARNING: element not found')
    return element

# save CHeart files


def connectivity_vtu2CH(element, ien):  # TODO
    # CH to vtu node numeration
    if element == 'line3':
        ien = ien[:, np.array([0, 2, 1])]
    elif element == 'triangle6':
        ien = ien[:, np.array([0, 1, 2, 3, 5, 4])]
    elif element == 'quad':
        ien = ien[:, np.array([0, 1, 3, 2])]
    elif element == 'quad9':
        ien = ien[:, np.array([0, 1, 3, 2,
                               4, 7, 8, 5,
                               6])]
    elif element == 'hexahedron':
        ien = ien[:, np.array([0, 1, 3, 2,
                               4, 5, 7, 6])]

    elif element == 'tetra10':
        ien = ien[:, np.array([0, 1, 2, 3, 4, 6, 5, 7, 8, 9])]
    elif element == 'hexahedron27':
        ien = ien[:, np.array([0, 1, 3, 2,
                               4, 5, 7, 6,
                               8, 11, 24, 9,
                               10, 16, 22, 17,
                               20, 26, 21, 19,
                               23, 18, 12, 15,
                               25, 13, 14])]

    return ien


def write_xfile(fname, pts):    # TODO check if the extension is correct, if not add it
    np.savetxt(fname, pts, fmt='%30.15f',
               delimiter='\t',  header = str(pts.shape[0]) + '\t' + str(pts.shape[1]),
               comments = '')

def write_tfile(fname, elems, pts):
    np.savetxt(fname, elems+1, fmt='%i',
           delimiter='\t', header = str(elems.shape[0]) + '\t' + str(pts.shape[0]),
           comments = '')

def write_mesh(fname, pts, elems, element=None):
    write_xfile(fname + '_FE.X', pts)
    elems, element = get_element_type(elems, ch2vtu=False, element=element)

    write_tfile(fname + '_FE.T', elems, pts)

def write_bfile(fname, bound):
    boundary = bound.copy()
    # Fix the numeration of elements and points (or unfix it depending who you ask)
    boundary[:,0:-1] += 1

    bfaces = boundary[:,1:-1]
    bfaces, _ = get_element_type(bfaces, ch2vtu=False)
    boundary[:,1:-1] = bfaces
    np.savetxt(fname+'_FE.B', boundary, fmt='%i',
           delimiter='\t', header = str(boundary.shape[0]), comments = '')


def write_dfile(fname, array, fmt=None):     # TODO check if name finish in D or not
    shape = array.shape
    size = array.size
    if size == 1:
        np.savetxt(fname, array, header = str(1) + '\t' + str(1),
               comments = '')
    else:
        if len(shape) == 1:
            s1 = shape[0]
            s2 = 1
        else:
            s1 = shape[0]
            s2 = shape[1]
        if fmt == None:
            fmt = '%30.15f'
        np.savetxt(fname, array, fmt=fmt,header = str(s1) + '\t' + str(s2),
               comments = '')

def write_specific(fname, nodes, values, fmt=None):
    if len(values.shape) == 1:
        specific = np.vstack([nodes+1, values]).T
        fmt = ['%i', '%f']
    else:
        specific = np.hstack([nodes[:,None]+1, values])
        fmt = ['%i'] + ['%f']*values.shape[1]
    np.savetxt(fname, specific, header = str(len(specific)), comments = '', fmt = fmt)

def read_specific(fname):
    data = np.loadtxt(fname, skiprows=1)
    nodes = data[:,0].astype(int) - 1
    values = data[:,1:]
    return nodes, values

def vtu_to_mesh(iname, oname, dim=3):
    mesh = io.read(iname)
    pts = mesh.points
    if len(mesh.cells) > 1:
        elem_nodes = []
        for c in mesh.cells:
            elem_nodes = c.data.shape[1]
        ind = np.argmax(elem_nodes)
        print('WARNING: Multiple types of cells defined. Choosing ' + c.type)
        elem = mesh.cells[ind].type
        elems = mesh.cells[ind].data

    else:
        elem = mesh.cells[0].type
        elems = mesh.cells[0].data

    if dim == 2:
        pts = pts[:,0:2]

    # This is just to fix the node order in case is needed
    ien, elem = get_element_type(elems, element=elem)

    write_xfile(oname + '_FE.X', pts)
    write_tfile(oname + '_FE.T', ien, pts)


# to .vtu
def mesh_to_vtu(mesh_path, out_name, elem=None, xfile=None):

    X, T, elem = read_mesh(mesh_path, element=elem)
    if xfile is not None:
        X = read_dfile(xfile)

    io.write_points_cells(out_name, X, {elem: T})

def bfile_to_vtu(mesh_path, out_name, element = None):
    X, T, _ = read_mesh(mesh_path)
    B = read_bfile(mesh_path)
    ien = B[:,1:-1]
    marker = B[:,-1]
    element = get_element_type_by_nnodes(ien, B)
    io.write_points_cells(out_name, X, {element: ien},
                          cell_data = {'patches': [marker]})

def bfile_to_blockmesh(mesh_path):
    X, T, _ = read_mesh(mesh_path)
    B = read_bfile(mesh_path)
    ien = B[:,1:-1]
    marker = B[:,-1]
    ien, element = get_element_type(ien)

    nmarkers = np.unique(marker)
    cells = []
    for i in nmarkers:
        cells.append(io.CellBlock(element, ien[marker==i]))

    return io.Mesh(X, cells)


def dfile_to_vtu(D_path, out_name, mesh_path=None, mesh=None, var_name = 'f',
                 array_type = 'points', element=None, inverse=False, xfile=None):

    if not mesh_path == None:
        mesh = read_mesh(mesh_path, element=element, meshio=True)
    if xfile is not None:
        mesh.points = read_dfile(xfile)
    X = mesh.points

    # Check if 2D data
    if X.shape[1] == 2:
        X = np.hstack([X, np.zeros([X.shape[0],1])])

    if type(D_path) == list:
        for d, v in zip(D_path, var_name):
            array = np.loadtxt(d, skiprows = 1)
            if len(array.shape) != 1:
                if array.shape[1] == 2:
                    array = np.hstack([array, np.zeros([array.shape[0],1])])

            if array_type == 'points':
                # Check that array has correct size
                assert len(array) == len(mesh.points), 'number of data points is not the same as the number of mesh points'
                mesh.point_data[v] = array
            elif array_type == 'cells':
                # Check that array has correct size
                assert len(array) == len(mesh.cells[0].data), 'number of data points is not the same as the number of elements'
                mesh.cell_data[v] = [array]


    else:
        array = np.loadtxt(D_path, skiprows = 1)
        if inverse: array = -array
        if len(array.shape) != 1:
            if array.shape[1] == 2:
                array = np.hstack([array, np.zeros([array.shape[0],1])])

        if array_type == 'points':
            # Check that array has correct size
            assert len(array) == len(mesh.points), 'number of data points is not the same as the number of mesh points'
            mesh.point_data[var_name] = array
        elif array_type == 'cells':
            # Check that array has correct size
            assert len(array) == len(mesh.cells[0].data), 'number of data points is not the same as the number of elements'
            mesh.cell_data[var_name] = [array]

    io.write(out_name, mesh)


def dict_to_pfile(out_name, dictionary):    # TODO make this way more general
    f = open(out_name, "w")
    for key in dictionary.keys():
        f.write('#' + key + '=' + str(dictionary[key]) + '\n')
    f.close()

def pfile_to_dict(pfile, fmt=float):    # TODO make this way more general
    dictionary = {}
    f = open(pfile, "r")
    for line in f:
        # Check for comments
        c_line = line.split('%')
        if len(c_line) > 1:
            line = c_line[0]
        try:
            var, value = line.split('=')
        except:
            continue
        var = var.replace('#', '')
        dictionary[var] = fmt(value)
    f.close()
    return dictionary

def fibers_to_vtu(fib_path, out_name, mesh_path, array_type = 'points', element=None):

    mesh = read_mesh(mesh_path, meshio=True, element=element)
    fibs = read_fibers(fib_path, append2d=True)

    if len(fibs) == 3:
        if array_type == 'points':
            mesh.point_data['f'] = fibs[0]
            mesh.point_data['s'] = fibs[1]
            mesh.point_data['n'] = fibs[2]
        elif array_type == 'cells':
            mesh.cell_data['f'] = [fibs[0]]
            mesh.cell_data['s'] = [fibs[1]]
            mesh.cell_data['n'] = [fibs[2]]
    else:
        if array_type == 'points':
            mesh.point_data['f'] = fibs[0]
            mesh.point_data['s'] = fibs[1]
        elif array_type == 'cells':
            mesh.cell_data['f'] = [fibs[0]]
            mesh.cell_data['s'] = [fibs[1]]

    io.write(out_name, mesh)


def cell_blocks_to_vtu(ifile, ofile):
    try:
        mesh = io.read(ifile)
    except:
        mesh = ifile
    patches = mesh.cells

    for i, p in enumerate(patches):
        patch_mesh = io.Mesh(mesh.points, {p.type: p.data})
        for cd in mesh.cell_data.keys():
            patch_mesh.cell_data[cd] = mesh.cell_data[cd][i]
        io.write(ofile + '_patch%i' % (i+1) + '.vtk', patch_mesh)


def patches_to_vtu(mesh, ofile, elem = 'triangle', ext='.vtk'):
    patches = mesh.cells

    for i, p in enumerate(patches):
        io.write_points_cells(ofile + '_patch%i' % (i+1) + ext, mesh.points, {elem: p.data})


# Utils
def compute_difference(dfile1, dfile2):
    d1 = read_dfile(dfile1)
    d2 = read_dfile(dfile2)
    return np.linalg.norm(d1-d2)

def map_between_meshes(mesh1, mesh2):
    from scipy.spatial import KDTree
    if isinstance(mesh1, str):
        mesh1 = read_mesh(mesh1, meshio=True)
    if isinstance(mesh2, str):
        mesh2 = read_mesh(mesh2, meshio=True)

    # mesh2 < mesh1
    tree1 = KDTree(mesh1.points)
    tree2 = KDTree(mesh2.points)
    corr = tree2.query_ball_tree(tree1, 1e-6)
    corr = np.asarray(corr).flatten()
    return corr

def map_between_meshes_disc(mesh1, mesh2):
    # mesh1 < mesh2
    from scipy.spatial import KDTree
    if isinstance(mesh1, str):
        mesh1 = read_mesh(mesh1, meshio=True)
    if isinstance(mesh2, str):
        mesh2 = read_mesh(mesh2, meshio=True)

    xyz1 = mesh1.points
    xyz2 = mesh2.points
    ien1 = mesh1.cells[0].data
    ien2 = mesh2.cells[0].data

    # Get midpoints
    midpoints1 = np.mean(xyz1[ien1], axis=1)
    midpoints2 = np.mean(xyz2[ien2], axis=1)

    # Find cell correspondence
    tree1 = KDTree(midpoints1)
    tree2 = KDTree(midpoints2)

    corr = tree2.query_ball_tree(tree1, 1e-6)
    corr = np.asarray(corr).flatten()

    mapp = np.zeros(len(xyz1), dtype=int)
    for e in range(len(ien1)):
        nodes = ien1[e]
        tree = KDTree(xyz1[nodes])
        dist, node_corr = tree.query(xyz2[ien2[corr[e]]])
        mapp[nodes] = ien2[corr[e], node_corr[dist==0]]


    return mapp



def get_mesh_boundary(mesh):
    ien = mesh.cells[0].data
    element = mesh.cells[0].type

    # Find boundary of mesh
    array = face_array(element)      # TODO do this dependent of the element
    nfaces = len(array)
    sface = array.shape[1]

    faces_elem = ien[:,array]
    faces = np.sort(faces_elem.reshape([-1,sface]), axis=1)
    face_elem = np.repeat(np.arange(ien.shape[0]), nfaces).flatten()

    unique, ind, count = np.unique(faces, return_counts=True, axis=0, return_index=True)

    bind = ind[count==1]
    belem = face_elem[bind]

    marker = np.zeros(len(faces), dtype=int)
    marker[bind] = bind

    bdata = np.zeros([len(bind), sface+2], dtype=int)
    bdata[:,1:-1] = faces_elem.reshape([-1,sface])[bind]
    bdata[:,0] = belem

    return bdata

def get_face_normal(xyz, ien):
    vertex_elem = xyz[ien]

    v1 = vertex_elem[:,1,:] - vertex_elem[:,0,:]
    v1 = v1/np.linalg.norm(v1, axis=1)[:,None]
    v2 = vertex_elem[:,2,:] - vertex_elem[:,0,:]
    v2 = v2/np.linalg.norm(v2, axis=1)[:,None]

    normal = np.cross(v1,v2,axisa=1,axisb=1)
    normal = normal/np.linalg.norm(normal, axis=1)[:,None]

    return normal



def get_surface_normals(points, ien):
    points_elems = points[ien]

    v1 = points_elems[:,1] - points_elems[:,0]
    v2 = points_elems[:,2] - points_elems[:,0]

    normal = np.cross(v1, v2, axisa=1, axisb=1)
    normal = normal/np.linalg.norm(normal,axis=1)[:,None]

    return normal

def create_bfile(mesh, cells, corrs = None, inner_faces = False):
    """
    Parameters
    ----------
    mesh : meshio mesh with the patches defined
    cells : volumetric mesh cells
    corrs : array, optional
        Correspondance matrix if precomputed. The default is None.

    """
    points = mesh.points
    patches = mesh.cells

    if cells.shape[1] == 4:
        nfaces = 4
        faces = np.zeros([cells.shape[0], 4, 3], dtype=int)     # num_cells, num_faces per cell, num_nodes per face
        faces[:,0,:] = np.array([cells[:,0], cells[:,1], cells[:,2]]).T
        faces[:,1,:] = np.array([cells[:,0], cells[:,2], cells[:,3]]).T
        faces[:,2,:] = np.array([cells[:,0], cells[:,3], cells[:,1]]).T
        faces[:,3,:] = np.array([cells[:,1], cells[:,2], cells[:,3]]).T
    elif cells.shape[1] == 10:
        nfaces = 4
        faces = np.zeros([cells.shape[0], 4, 6], dtype=int)
        faces[:,0,:] = np.array([cells[:,0], cells[:,1], cells[:,2], cells[:,4], cells[:,5], cells[:,6]]).T
        faces[:,1,:] = np.array([cells[:,0], cells[:,2], cells[:,3], cells[:,6], cells[:,9], cells[:,7]]).T
        faces[:,2,:] = np.array([cells[:,0], cells[:,3], cells[:,1], cells[:,7], cells[:,8], cells[:,4]]).T
        faces[:,3,:] = np.array([cells[:,1], cells[:,2], cells[:,3], cells[:,5], cells[:,9], cells[:,8]]).T
    elif cells.shape[1] == 3:
        nfaces = 3
        faces = np.zeros([cells.shape[0], 3, 2], dtype=int)
        faces[:,0,:] = np.array([cells[:,0], cells[:,1]]).T
        faces[:,1,:] = np.array([cells[:,1], cells[:,2]]).T
        faces[:,2,:] = np.array([cells[:,2], cells[:,0]]).T
    else:
        raise 'Not implemented'


    # Compute centroids of all elements in the mesh
    vertex = points[faces]
    ctet = np.mean(vertex, axis = 2).reshape([-1,3])
    if inner_faces:
        midpoints = np.mean(points[cells],axis=1)

    # Loop the patches to build the B file
    from scipy.spatial import KDTree
    tree1 = KDTree(ctet)
    ind = np.repeat(np.arange(cells.shape[0]),nfaces)

    bdata = []
    compute = False
    if corrs == None:
        corrs = []
        compute = True
    for i, p in enumerate(patches):
        surf = p.data
        ctri = np.mean(points[surf],axis=1)

        if compute:
            # Find correspondance # TODO there is probably a better way to do this.
            tree2 = KDTree(ctri)

            corr = tree2.query_ball_tree(tree1, 1e-5)
            
            # Diagnostic prints to understand the structure
            print(f"\n=== Diagnostic for patch {i+1} ===")
            print(f"  Number of surface triangles: {len(corr)}")
            print(f"  Type of corr: {type(corr)}")
            print(f"  First few elements:")
            for j in range(min(10, len(corr))):
                print(f"    corr[{j}]: type={type(corr[j])}, length={len(corr[j]) if hasattr(corr[j], '__len__') else 'N/A'}, value={corr[j]}")
            
            # Count distribution of match lengths
            match_lengths = [len(matches) for matches in corr]
            unique_lengths, counts = np.unique(match_lengths, return_counts=True)
            print(f"  Match length distribution:")
            for length, count in zip(unique_lengths, counts):
                print(f"    {length} matches: {count} triangles")
            
            # Find examples of triangles with multiple matches
            multi_match_indices = [j for j, matches in enumerate(corr) if len(matches) > 1]
            if len(multi_match_indices) > 0:
                print(f"  Found {len(multi_match_indices)} triangles with multiple matches")
                print(f"  First 5 examples:")
                for idx in multi_match_indices[:5]:
                    print(f"    Triangle {idx}: {len(corr[idx])} matches = {corr[idx]}")
            
            # Find triangles with no matches
            no_match_indices = [j for j, matches in enumerate(corr) if len(matches) == 0]
            if len(no_match_indices) > 0:
                print(f"  Found {len(no_match_indices)} triangles with no matches")
            
            print(f"  Attempting to convert to numpy array...")
            try:
                corr = np.asarray(corr)
                print(f"  Success! Shape: {corr.shape}")
            except ValueError as e:
                print(f"  ERROR: {e}")
                print(f"  This is why we need the patched version!")
                raise
            
            corrs.append(corr)

        else:
            corr = corrs[i]
        tet_face = ind[corr]

        if cells.shape[1] == 10:
            bdata.append(np.vstack([tet_face, surf[:,np.array([0, 1, 2, 3, 5, 4])].T,
                                    np.ones(surf.shape[0], dtype=int)*(i+1)]).T)
        else:
            if tet_face.shape[1] == 1:
                tet_face = tet_face.flatten()
                bdata.append(np.vstack([tet_face, surf.T, np.ones(surf.shape[0], dtype=int)*(i+1)]).T)
            else:
                if inner_faces == False:
                    print('Boundary {:d} could not be added'.format(i+1))
                    continue
                assert tet_face.shape[1] == 2

                # Computing the vector from the face midpoint to the tet midpoint
                tri_midpoint = np.mean(points[surf],axis=1)
                tet_midpoint = midpoints[tet_face]
                vector = tet_midpoint - tri_midpoint[:,None,:]

                # Computing face normal and dot product with vector
                normals = get_surface_normals(points, surf)
                dot = np.sum(vector*normals[:,None,:],axis=2)

                # Correspondent tet would be the one that produces positive normal
                tet_face = tet_face[dot>0]
                tet_face = tet_face.flatten()
                bdata.append(np.vstack([tet_face, surf.T, np.ones(surf.shape[0], dtype=int)*(i+1)]).T)
                corr=corr[dot>0][:,None]

    bdata = np.vstack(bdata)

    return bdata, corrs

