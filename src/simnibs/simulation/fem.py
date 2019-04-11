'''
    FEM calculations using getdp
    This program is part of the SimNIBS package.
    Please check on www.simnibs.org how to cite our work in publications.

    Copyright (C) 2018 Guilherme Saturnino

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.



'''


from __future__ import print_function
import os
import copy
import tempfile
import numpy as np
import warnings
from ..msh import gmsh_numpy as gmsh
from ..utils.simnibs_logger import logger
from . import getdp
from . import cond as cond_lib

import gc

def calc_fields(potentials, fields, cond=None, dadt=None, units='mm'):
    ''' Given a mesh and the electric potentials at the nodes, calculates the fields
    Parameters:
    -----
    potentials: simnibs.msh.gmsh_numpy.NodeData
        NodeData field with potentials. The mesh property should be set
    fields: Any combination of 'vEeJjsDg'
        Fields to output
        v: electric potential at the nodes
        E: Electric field at the elements
        e: Electric field norm at the elements
        J: Current density at the elements
        j: Current density norm at the elements
        s: Conductivity at the elements
        D: dA/dt at the nodes
        g: gradiet of the potential at the elements
    cond: simnibs.mesh.gmsh_numpy.ElementData (optional)
        Conductivity at the elements, used to calculate J, j and s. Might be a scalar (1
        element) or a tensor (9 elements)
    dadt: simnibs.msh.gmsh_numpy.NodeData (optional)
        dA/dt at the nodes for TMS simulations
    units: {'mm' or 'm'} (optional)
        Mesh units, either milimiters (mm) or meters (m). Default: mm
    '''
    if units == 'mm':
        scaling_factor = 1e3
    elif units == 'm':
        scaling_factor = 1
    else:
        raise ValueError('Invalid units: {0}'.format(units))

    mesh = potentials.mesh
    if mesh is None:
        raise ValueError('potential does not have the mesh property set')

    assert mesh.nodes.nr == potentials.nr, \
        ('The number of nodes in the mesh and of data points in the'
         ' potential does not match')

    if cond is not None:
        assert mesh.elm.nr == cond.nr, \
            ('The number of elements in the mesh and of data points in the'
             ' conductivity field does not match')

    out_mesh = copy.deepcopy(mesh)
    out_mesh.elmdata = []
    out_mesh.nodedata = []
    if 'v' in fields:
        out_mesh.nodedata.append(
            gmsh.NodeData(
                potentials.value,
                name='v',
                mesh=out_mesh))
    if 'D' in fields:
        if dadt is None:
            warnings.warn('Cannot calculate D field: needs dadt input')
        if isinstance(dadt, gmsh.NodeData):
            out_mesh.nodedata.append(
                gmsh.NodeData(
                    dadt.value,
                    name='D',
                    mesh=out_mesh))
        else:
            out_mesh.elmdata.append(
                gmsh.ElementData(
                    dadt.value,
                    name='D',
                    mesh=out_mesh))

    if any(f in ['E', 'e', 'J', 'j', 'g', 's'] for f in fields):
        grad = potentials.gradient() * scaling_factor
        grad.assign_triangle_values()
        grad.field_name = 'g'
        grad.mesh = out_mesh

        if 'g' in fields:
            out_mesh.elmdata.append(grad)

        if dadt is not None:
            if isinstance(dadt, gmsh.NodeData):
                dadt_elmdata = dadt.node_data2elm_data()
            else:
                dadt_elmdata = dadt
            dadt_elmdata.assign_triangle_values()
            E = gmsh.ElementData(
                -grad.value - dadt_elmdata.value,
                name='E',
                mesh=out_mesh)
        else:
            E = gmsh.ElementData(
                -grad.value,
                name='E',
                mesh=out_mesh)

        if 'E' in fields:
            out_mesh.elmdata.append(E)
        if 'e' in fields:
            e = np.linalg.norm(E.value, axis=1)
            out_mesh.elmdata.append(
                gmsh.ElementData(
                    e, name='normE', mesh=out_mesh))

        if any(f in ['J', 'j', 's'] for f in fields):
            if cond is None:
                raise ValueError('Cannot calculate J, j os s field: No conductivity input')
            cond.assign_triangle_values()
            if 's' in fields:
                cond.field_name = 'conductivity'
                cond.mesh = out_mesh
                if cond.nr_comp == 9:
                    out_mesh.elmdata += cond_lib.TensorVisualization(cond)
                else:
                    out_mesh.elmdata.append(cond)

            if cond.nr_comp == 1:
                J = gmsh.ElementData(
                    cond.value[:, None] * E.value,
                    name='J',
                    mesh=out_mesh)
            elif cond.nr_comp == 9:
                J = gmsh.ElementData(
                    np.einsum('ikj, ik -> ij',
                              cond.value.reshape(-1, 3, 3),
                              E.value),
                    name='J',
                    mesh=out_mesh)
            else:
                raise ValueError('Invalid number of components in cond:'
                                 '{0}'.format(cond.nr_comp))
            if 'J' in fields:
                out_mesh.elmdata.append(J)
            if 'j' in fields:
                j = np.linalg.norm(J.value, axis=1)
                out_mesh.elmdata.append(
                    gmsh.ElementData(
                        j, name='normJ', mesh=mesh))

    return out_mesh


def _mesh_prep(mesh):
    ''' If a file name, loads the mesh. If a mesh structure, writes it into a temporary
    file '''
    if isinstance(mesh, gmsh.Msh):
        return mesh
    else:
        fn_mesh = os.path.abspath(os.path.expanduser(mesh))
        mesh = gmsh.read_msh(fn_mesh)
        if not os.path.isfile(fn_mesh):
            raise IOError('mesh argument is neither a Msh class instance'
                          ' or a valid file name')
    return mesh


def _cond_prep(mesh, cond, ret_cond=False, tmpfolder=None):
    if isinstance(cond, gmsh.ElementData):
        fn_cond = tempfile.NamedTemporaryFile(suffix='.msh', delete=False, dir=tmpfolder).name
        m = copy.deepcopy(mesh)
        m.nodedata = []
        m.elmdata = [cond]
        gmsh.write_msh(m, fn_cond)
        remove_cond = True
    else:
        fn_cond = os.path.abspath(os.path.expanduser(cond))
        if ret_cond:
            cond = gmsh.read_msh(fn_cond).elmdata[0]
        remove_cond = False
        if not os.path.isfile(fn_cond):
            raise IOError('cond argument is neither an ElementData instance'
                          ' or a valid file name')

    if ret_cond:
        return fn_cond, cond, remove_cond
    else:
        return fn_cond, remove_cond

def tms_getdp(mesh, cond, dAdt, fn_pro=None, keepfiles=False, tmpfolder=None):
    ''' Simulates a TMS field using getDP
    If strings are used, it will use already existing files. Otherwise, temporary files
    are created

    Parameters:
    ------
    mesh: simnibs.msh.gmsh_numpy.Msh or str
        Mesh file with geometry information
    cond: simnibs.msh.gmsh_numpy.ElementData or str
        An ElementData field with conductivity information
    dAdt: simnibs.msh.gmsh_numpy.NodeData or str
        An NodeData field with dAdt or file name to a mesh with dA/dt informaion
    fn_pro: str (optional)
        The file name of where the pro file is to be written. Default: use a temporary file
    Returns:
    -------
    v:  simnibs.msh.gmsh_numpy.NodeData
        NodeData instance with potential at the nodes
    '''
    mesh = _mesh_prep(mesh)
    fn_cond, remove_cond = _cond_prep(mesh, cond, ret_cond=False, tmpfolder=tmpfolder)

    if isinstance(dAdt, gmsh.NodeData):
        fn_dadt = tempfile.NamedTemporaryFile(suffix='.msh', delete=False, dir=tmpfolder).name
        m = copy.deepcopy(mesh)
        m.elmdata = []
        m.nodedata = [dAdt]
        gmsh.write_msh(m, fn_dadt)
        remove_dadt = True
    else:
        fn_dadt = os.path.abspath(os.path.expanduser(dAdt))
        remove_dadt = False
        if not os.path.isfile(fn_dadt):
            raise IOError('dadt argument is neither a NodeData class instance'
                          ' or a valid file name')

    if fn_pro is None:
        fn_pro = tempfile.NamedTemporaryFile(suffix='.pro', delete=False, dir=tmpfolder).name
        remove_pro = True
    else:
        remove_pro = False

    getdp.create_tms_pro(mesh, fn_cond, fn_dadt, fn_pro)
    v = getdp.solve(mesh, fn_pro, keepfiles=keepfiles, tmpfolder=tmpfolder)

    if remove_dadt:
        os.remove(fn_dadt)

    if remove_cond:
        os.remove(fn_cond)

    if remove_pro:
        os.remove(fn_pro)
    return v


def tdcs_getdp(mesh, cond, currents, electrode_surface_tags,
               fn_pro=None, units='mm', keepfiles=False, tmpfolder=None):
    ''' Simulates a tDCS field using getDP
    If strings are used, it will use already existing files. Otherwise, temporary files
    are created

    Parameters:
    ------
    mesh: simnibs.msh.gmsh_numpy.Msh or str
        Mesh file with geometry information
    cond: simnibs.msh.gmsh_numpy.ElementData or str
        An ElementData field with conductivity information
    currents: list or ndarray
        A list of currents going though each electrode
    electrode_surface_tags: list
        A list of the indices of the surfaces where the dirichlet BC is to be applied

    Returns:
    ---------------
    potential: simnibs.msh.gmsh_numpy.NodeData
        Total electric potential
    '''
    assert len(currents) == len(electrode_surface_tags),\
        'there should be one channel for each current'
    mesh = _mesh_prep(mesh)
    fn_cond, cond, remove_cond = _cond_prep(mesh, cond, ret_cond=True,
                                            tmpfolder=tmpfolder)

    if fn_pro is None:
        fn_pro = tempfile.NamedTemporaryFile(suffix='.pro', delete=False, dir=tmpfolder).name
        remove_pro = True
    else:
        remove_pro = False

    surf_tags = np.unique(mesh.elm.tag1[mesh.elm.elm_type == 2])
    assert np.all(np.in1d(electrode_surface_tags, surf_tags)),\
        'Could not find all the electrode surface tags in the mesh'

    assert np.isclose(np.sum(currents), 0),\
        'Currents should sum to 0'

    ref_electrode = electrode_surface_tags[0]
    total_p = np.zeros(mesh.nodes.nr, dtype=np.float)
    # Creates a list of pro_file names
    if isinstance(fn_pro, list) or isinstance(fn_pro, tuple):
        fn_pro_list = fn_pro
        assert len(fn_pro_list) == len(electrode_surface_tags) - 1
    else:
        fn_pro_list = (len(electrode_surface_tags) - 1) * [fn_pro]

    for el_surf, el_c, fn_p in zip(electrode_surface_tags[1:], currents[1:],
                                   fn_pro_list):
        getdp.create_tdcs_pro(mesh, fn_cond, ref_electrode, el_surf, fn_p)
        v = getdp.solve(mesh, fn_p, keepfiles=keepfiles, tmpfolder=tmpfolder)
        flux = np.array([
            _calc_flux_electrodes(v, cond,
                                  [el_surf - 1000, el_surf - 600,
                                   el_surf - 2000, el_surf - 1600],
                                  units=units),
            _calc_flux_electrodes(v, cond,
                                  [ref_electrode - 1000, ref_electrode - 600,
                                   ref_electrode - 2000, ref_electrode - 1600],
                                  units=units)])
        current = np.average(np.abs(flux))
        total_p += el_c / current * v.value
        error = np.abs(np.abs(flux[0]) - np.abs(flux[1])) / current
        logger.info('Estimated current calibration error: {0:.1%}'.format(error))

    if remove_cond:
        os.remove(fn_cond)

    if remove_pro:
        os.remove(fn_pro)
    gc.collect()
    return gmsh.NodeData(total_p, 'v', mesh=mesh)


def _calc_flux_electrodes(v, cond, el_volume, scalp_tag=[5, 1005], units='mm'):
    # Set-up a mesh with a mesh
    m = copy.deepcopy(v.mesh)
    m.nodedata = [v]
    m.elmdata = [cond]
    # Select mesh nodes wich are is in one electrode as well as the scalp
    # Triangles in scalp
    tr_scalp = np.in1d(m.elm.tag1, scalp_tag) * (m.elm.elm_type == 2)
    if not np.any(tr_scalp):
        raise ValueError('Could not find skin surface')
    tr_scalp_nodes = m.elm.node_number_list[tr_scalp, :3]
    tr_index = m.elm.elm_number[tr_scalp]

    # Tetrahehedra in electrode
    th_el = np.in1d(m.elm.tag1, el_volume) * (m.elm.elm_type == 4)
    if not np.any(th_el):
        raise ValueError('Could not find electrode volume')
    th_el_nodes = m.elm.node_number_list[th_el]
    nodes_el = np.unique(th_el_nodes)
    th_index = m.elm.elm_number[th_el]

    # Triangles in interface
    tr_interface = tr_index[
        np.all(np.isin(tr_scalp_nodes, nodes_el), axis=1)]
    if len(tr_interface) == 0:
        raise ValueError('Could not find skin-electrode interface')
    keep = np.hstack((th_index, tr_interface))

    # Now make a mesh with only tetrahedra and triangles in interface
    crop = m.crop_mesh(elements=keep)
    crop.elm.tag1 = np.ones_like(crop.elm.tag1)
    crop.elm.tag2 = np.ones_like(crop.elm.tag2)

    # Calculate J in the interface
    crop = calc_fields(crop.nodedata[0], 'J', crop.elmdata[0], units=units)

    # Calculate flux
    flux = crop.elmdata[0].calc_flux()
    if units == 'mm':
        flux *= 1e-6

    del m
    del crop
    return flux


def tdcs_neumann_getdp(mesh, cond, currents, electrode_surface_tags,
                       fn_pro=None, units='mm', tmpfolder=None):
    ''' Simulates a tDCS field using getDP
    If strings are used, it will use already existing files. Otherwise, temporary files
    are created

    Parameters:
    ------
    mesh: simnibs.msh.gmsh_numpy.Msh or str
        Mesh file with geometry information
    cond: simnibs.msh.gmsh_numpy.ElementData or str
        An ElementData field with conductivity information
    currents: list or ndarray
        A list of currents going though each electrode
    electrode_surface_tags: list
        A list of the indices of the surfaces where the dirichlet BC is to be applied
    '''
    raise NotImplementedError()
    #TODO: Fix this function
    assert len(currents) == len(electrode_surface_tags),\
        'there should be one electrode surface for each current'

    mesh = _mesh_prep(mesh)
    fn_cond, remove_cond = _cond_prep(mesh, cond, ret_cond=False, tmpfolder=tmpfolder)

    if fn_pro is None:
        fn_pro = tempfile.NamedTemporaryFile(suffix='.pro', delete=False, dir=tmpfolder).name
        remove_pro = True
    else:
        remove_pro = False

    surf_tags = np.unique(mesh.elm.tag1[mesh.elm.elm_type == 2])
    assert np.all(np.in1d(electrode_surface_tags, surf_tags)),\
        'Could not find all the electrode surface tags in the mesh'

    assert np.isclose(np.sum(currents), 0),\
        'Currents should sum to 0'

    areas = mesh.elements_volumes_and_areas().value
    surf_area = np.zeros(len(electrode_surface_tags), dtype=float)
    for i, s in enumerate(electrode_surface_tags):
        surf_area[i] = np.sum(areas[(mesh.elm.tag1 == s) * (mesh.elm.elm_type == 2)])
    if units == 'mm':
        surf_area *= 1e-6
    current_density = currents / surf_area
    getdp.create_tdcs_neumann_pro(mesh, fn_cond, current_density, electrode_surface_tags, fn_pro)
    v = getdp.solve(mesh, fn_pro, tmpfolder=tmpfolder)

    if remove_cond:
        os.remove(fn_cond)

    if remove_pro:
        os.remove(fn_pro)
    return v
