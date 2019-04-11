# -*- coding: utf-8 -*-\
'''
    This program is part of the SimNIBS package.
    Please check on www.simnibs.org how to cite our work in publications.

    Copyright (C) 2013-2018 Andre Antunes, Guilherme Saturnino

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

import tempfile
import os
import re
import warnings
import time
import sys
import numpy as np
from datetime import datetime
import gc

from .. import SIMNIBSDIR
from ..utils.simnibs_logger import logger
from ..utils.run_shell_command import run_command
from ..msh import gmsh_numpy as gmsh


# create the pro file for getdp - TMS like problems
def create_tms_pro(mesh, fn_cond, fn_dadt, fn_out):
    """Creates a .pro file for running getDP

    Parameters
    -----------------
    mesh: simnibs.msh.gmsh_numpy.Msh()
        Mesh where the simulation will be performed
    fn_cond: str
        String with name to the file with conductivity information
    fn_dadt: str
        String with the name of the file with dadt information
    fn_out: str
       Name of file to be written
    """

    with open(fn_out, 'w') as f:
        f.write('// simnibs automatically generated file for getdp simulation\n')
        f.write('// file created with getdp.py on ' +
                datetime.now().strftime('%Y%m%d_%H%M%S') + '\n\n')

        f.write(pro_file_conductivity(mesh, fn_cond))

        f.write('Function {\n')
        f.write('\tdadt[] = VectorField[XYZ[]]{2};\n')
        f.write('}\n\n\n')

        # create Jacobian
        f.write('Jacobian {\n')
        f.write('\t{ Name Volume;  Case {{ Region Omega; Jacobian Vol;  }} }\n')
        f.write('}\n\n\n')

        # create Integration
        f.write('Integration {{\n')
        f.write('\tName GradGrad;\n')
        f.write('\tCase {{\n')
        f.write('\t\tType Gauss;\n')
        f.write('\t\tCase {\n')
        f.write('\t\t\t{GeoElement Tetrahedron; NumberOfPoints 1;}\n')
        f.write('\t\t}\n')
        f.write('\t}}\n')
        f.write('}}\n\n\n')

        # create FunctionSpace
        f.write('FunctionSpace {{\n')
        f.write('\tName Hgrad_vf_Ele;\n')
        f.write('\tType Form0;\n')
        f.write('\tBasisFunction {{\n')
        f.write('\t\tName sn;\n\t\tNameOfCoef vn;\n\t\tFunction BF_Node;\n')
        f.write('\t\tSupport Region[{Omega}];\n\t\tEntity NodesOf[All];\n')
        f.write('\t}}\n')
        f.write('}}\n\n\n')

        # create Formulation

        f.write('Formulation {{\n')
        f.write('\tName QS_Formulation;\n')
        f.write('\tType FemEquation;\n')
        f.write('\tQuantity {{\n')
        f.write('\t\tName v;\n')
        f.write('\t\tType Local;\n')
        f.write('\t\tNameOfSpace Hgrad_vf_Ele;\n')
        f.write('\t}}\n')
        f.write('\tEquation {\n')
        f.write('\t\tGalerkin {\n')
        f.write('\t\t\t[1e3 * sigma[] * Dof{Grad v}, {Grad v}];\n')
        f.write('\t\t\tIn Omega;\n')
        f.write('\t\t\tJacobian Volume;\n')
        f.write('\t\t\tIntegration GradGrad;\n')
        f.write('\t\t}\n')
        f.write('\t\tGalerkin {\n')
        f.write('\t\t\t[sigma[] * dadt[], {Grad v}];\n')
        f.write('\t\t\tIn Omega;\n')
        f.write('\t\t\tJacobian Volume;\n')
        f.write('\t\t\tIntegration GradGrad;\n')
        f.write('\t\t}\n')
        f.write('\t}\n')
        f.write('}}\n\n\n')

        # create Resolution

        f.write('Resolution {{\n')
        f.write('\tName QS;\n')
        f.write('\tSystem {{\n')
        f.write('\t\tName QS_dAdt;\n')
        f.write('\t\tNameOfFormulation QS_Formulation;\n')
        f.write('\t}}\n')
        f.write('\tOperation {\n')
        f.write('\t\tGmshRead[\"{0}\", 1];\n'.format(
            os.path.abspath(os.path.expanduser(fn_cond))))
        f.write('\t\tGmshRead[\"{0}\", 2];\n'.format(
            os.path.abspath(os.path.expanduser(fn_dadt))))
        f.write('\t\tGenerate QS_dAdt;\n')
        f.write('\t\tSolve QS_dAdt;\n')
        f.write('\t\tSaveSolution QS_dAdt;\n')

        f.write('\t}\n')
        f.write('}}\n\n\n')

# some code is repeated, could get common function to generate these parts
def create_tdcs_pro(mesh, fn_cond, reference, active, fn_out):
    """ Creates a .pro file for running getDP

    Parameters
    -----------------
    mesh: simnibs.msh.gmsh_numpy.Msh()
        Mesh where the simulation will be performed
    fn_cond: str
        String with name to the file with conductivity information
    reference: int
        Surface tag of the reference electrode. Will be assigned a potential of 0
    active: int
       Surface tag of the active electrode. Will be assigned a potential of 1
    fn_out:
        Name of file to be written
    """
    with open(fn_out, 'w') as f:
        f.write('// simnibs automatically generated file for getdp simulation\n')
        f.write('// file created with getdp.py on ' +
                datetime.now().strftime('%Y%m%d_%H%M%S') + '\n\n')
        f.write(pro_file_conductivity(mesh, fn_cond))
        f.write('Group{\n')
        f.write('\tReference=Region[{0}];Active=Region[{1}];\n'.format(reference, active))
        f.write('}\n\n\n')


        # create constraint for the electrodes
        f.write('Constraint {{\n')
        f.write('\tName ElectricScalarPotential;\n')
        f.write('\tType Assign;\n')
        f.write('\tCase {\n')
        #f.write('\t\t{Region Region[{Anode_surface}]; Value 1.;}\n')
        f.write('\t\t{Region Active; Value 1.;}\n')
        #f.write('\t\t{Region Region[{Cathode_surface}]; Value -1.;}\n')
        f.write('\t\t{Region Reference; Value 0.;}\n')
        f.write('\t}\n')
        f.write('}}\n\n\n')

        # create Jacobian
        f.write('Jacobian {\n')
        f.write('\t{ Name Volume;  Case {{ Region Omega; Jacobian Vol;  }} }\n')
        f.write('}\n\n\n')

        # create Integration
        f.write('Integration {{\n')
        f.write('\tName GradGrad;\n')
        f.write('\tCase {{\n')
        f.write('\t\tType Gauss;\n')
        f.write('\t\tCase {\n')
        f.write('\t\t\t{GeoElement Tetrahedron; NumberOfPoints 1;}\n')
        f.write('\t\t}\n')
        f.write('\t}}\n')
        f.write('}}\n\n\n')

        # create FunctionSpace
        f.write('FunctionSpace {{\n')
        f.write('\tName Hgrad_vf_Ele;\n')
        f.write('\tType Form0;\n')
        f.write('\tBasisFunction {{\n')
        f.write('\t\tName sn;\n\t\tNameOfCoef vn;\n\t\tFunction BF_Node;\n')
        f.write('\t\tSupport Region[{Omega}];\n\t\tEntity NodesOf[All];\n')
        f.write('\t}}\n')
        f.write('\tConstraint {{\n')
        f.write('\t\tNameOfCoef vn;\n')
        f.write('\t\tEntityType NodesOf;\n')
        f.write('\t\tNameOfConstraint ElectricScalarPotential;\n')
        f.write('\t}}\n')
        f.write('}}\n\n\n')

        # create Formulation
        f.write('Formulation {{\n')
        f.write('\tName Electrostatics_Formulation;\n')
        f.write('\tType FemEquation;\n')
        f.write('\tQuantity {{\n')
        f.write('\t\tName v;\n')
        f.write('\t\tType Local;\n')
        f.write('\t\tNameOfSpace Hgrad_vf_Ele;\n')
        f.write('\t}}\n')
        f.write('\tEquation {\n')
        f.write('\t\tGalerkin {\n')
        f.write('\t\t\t[sigma[] * Dof{Grad v}, {Grad v}];\n')
        f.write('\t\t\tIn Omega;\n')
        f.write('\t\t\tJacobian Volume;\n')
        f.write('\t\t\tIntegration GradGrad;\n')
        f.write('\t\t}\n')
        f.write('\t}\n')
        f.write('}}\n\n\n')

        # create Resolution
        f.write('Resolution {{\n')
        f.write('\tName QS;\n')
        f.write('\tSystem {{\n')
        f.write('\t\tName Electrostatic_System;\n')
        f.write('\t\tNameOfFormulation Electrostatics_Formulation;\n')
        f.write('\t}}\n')
        f.write('\tOperation {\n')
        f.write('\t\tGmshRead[\"{0}\", 1];\n'.format(
            os.path.abspath(os.path.expanduser(fn_cond))))
        f.write('\t\tGenerate Electrostatic_System;\n')
        f.write('\t\tSolve Electrostatic_System;\n')
        f.write('\t\tSaveSolution Electrostatic_System;\n')
        f.write('\t}\n')
        f.write('}}\n\n\n')


def create_tdcs_neumann_pro(mesh, fn_cond, currents, el_surface_tags, fn_out):
    """ Creates a .pro file for running getDP

    Parameters
    -----------------
    mesh: simnibs.msh.gmsh_numpy.Msh()
        Mesh where the simulation will be performed
    fn_cond: str
        String with name to the file with conductivity information
    currents: list
        Current density going through each electrode
    el_surface_tags: list
       Surface tag of each electrode surface
    fn_out:
        Name of file to be written
    """
    with open(fn_out, 'w') as f:
        f.write('// simnibs automatically generated file for getdp simulation\n')
        f.write('// file created with getdp.py on ' +
                datetime.now().strftime('%Y%m%d_%H%M%S') + '\n\n')
        f.write(pro_file_conductivity(mesh, fn_cond))
        f.write('Group{\n')
        f.write('\tReference=Region[{0}];\n'.format(el_surface_tags[0]))
        s_active = []
        for i, t in enumerate(el_surface_tags[1:]):
            f.write('\tActive{0}=Region[{1}];\n'.format(i+1, t))
            s_active.append('Active{0}'.format(i+1))
        f.write('\tGama=Region[{{{0}}}];\n'.format(','.join(s_active)))
        f.write('}\n\n\n')

        f.write('Function {\n')
        for i, c, t in zip(range(len(currents) - 1), currents[1:], el_surface_tags):
            f.write(
                '\tEin[Active{0}]={1};\n'.format(
                    i+1, c, t))
        f.write('}\n')

        # create constraint for the electrodes
        f.write('Constraint {{\n')
        f.write('\tName ElectricScalarPotential;\n')
        f.write('\tType Assign;\n')
        f.write('\tCase {\n')
        f.write('\t\t{Region Reference; Value 0.;}\n')
        f.write('\t}\n')
        f.write('}}\n\n\n')

        # create Jacobian
        f.write('Jacobian {\n')
        f.write('\t{ Name Volume;  Case {{ Region Omega; Jacobian Vol;  }} }\n')
        f.write('\t{ Name Surface; Case {{  Region Gama;  Jacobian Sur; }} }\n')
        f.write('}\n\n\n')

        # create Integration
        f.write('Integration {{\n')
        f.write('\tName GradGrad;\n')
        f.write('\tCase {{\n')
        f.write('\t\tType Gauss;\n')
        f.write('\t\tCase {\n')
        f.write('\t\t\t{GeoElement Tetrahedron; NumberOfPoints 1;}\n')
        f.write('\t\t\t{GeoElement Triangle; NumberOfPoints 4;}\n')
        f.write('\t\t}\n')
        f.write('\t}}\n')
        f.write('}}\n\n\n')

        # create FunctionSpace
        f.write('FunctionSpace {{\n')
        f.write('\tName Hgrad_vf_Ele;\n')
        f.write('\tType Form0;\n')
        f.write('\tBasisFunction {{\n')
        f.write('\t\tName sn;\n\t\tNameOfCoef vn;\n\t\tFunction BF_Node;\n')
        f.write('\t\tSupport Region[{Omega,Gama}];\n\t\tEntity NodesOf[All];\n')
        f.write('\t}}\n')
        f.write('\tConstraint {{\n')
        f.write('\t\tNameOfCoef vn;\n')
        f.write('\t\tEntityType NodesOf;\n')
        f.write('\t\tNameOfConstraint ElectricScalarPotential;\n')
        f.write('\t}}\n')
        f.write('}}\n\n\n')

        # create Formulation
        f.write('Formulation {{\n')
        f.write('\tName Electrostatics_Formulation;\n')
        f.write('\tType FemEquation;\n')
        f.write('\tQuantity {{\n')
        f.write('\t\tName v;\n')
        f.write('\t\tType Local;\n')
        f.write('\t\tNameOfSpace Hgrad_vf_Ele;\n')
        f.write('\t}}\n')
        f.write('\tEquation {\n')
        f.write('\t\tGalerkin {\n')
        f.write('\t\t\t[sigma[] * Dof{Grad v}, {Grad v}];\n')
        f.write('\t\t\tIn Omega;\n')
        f.write('\t\t\tJacobian Volume;\n')
        f.write('\t\t\tIntegration GradGrad;\n')
        f.write('\t\t}\n')
        f.write('\t\tGalerkin {\n')
        f.write('\t\t\t[-Ein[], {v}];\n')
        f.write('\t\t\tIn Gama;\n')
        f.write('\t\t\tJacobian Surface;\n')
        f.write('\t\t\tIntegration GradGrad;\n')
        f.write('\t\t}\n')
        f.write('\t}\n')
        f.write('}}\n\n\n')

        # create Resolution
        f.write('Resolution {{\n')
        f.write('\tName QS;\n')
        f.write('\tSystem {{\n')
        f.write('\t\tName Electrostatic_System;\n')
        f.write('\t\tNameOfFormulation Electrostatics_Formulation;\n')
        f.write('\t}}\n')
        f.write('\tOperation {\n')
        f.write('\t\tGmshRead[\"{0}\", 1];\n'.format(
            os.path.abspath(os.path.expanduser(fn_cond))))
        f.write('\t\tGenerate Electrostatic_System;\n')
        f.write('\t\tSolve Electrostatic_System;\n')
        f.write('\t\tSaveSolution Electrostatic_System;\n')
        f.write('\t}\n')
        f.write('}}\n\n\n')


def pro_file_conductivity(mesh, fn_cond):
    """Returns the string with the conductivities part of the pro file

    Parameters
    ---------------------
    fn_cond: simnibs.msh.gmsh_numpy.Msh
        Mesh structure with conductivity information as the first ElementData field
    Returns:
    ---------------------
    str:
        string to be printed to the pro file

    """
    vol_regions = np.unique(mesh.elm.tag1[mesh.elm.elm_type == 4])
    vol_regions_string = ';'.join('V{0}=Region[{0}]'.format(r) for r in vol_regions)
    omega_group_string = ','.join('V{0}'.format(r) for r in vol_regions)
    s = 'Group{\n'
    s += '\t{0};\n'.format(vol_regions_string)
    s += '\tOmega = Region[{{{0}}}];\n'.format(omega_group_string)

    surf_regions = np.unique(mesh.elm.tag1[mesh.elm.elm_type == 2])
    surf_regions_string = ';'.join('S{0}=Region[{0}]'.format(r) for r in surf_regions)
    surf_group_string = ','.join('S{0}'.format(r) for r in surf_regions)
    if len(surf_regions) > 0:
        s += '\t{0};\n'.format(surf_regions_string)
        s += '\tGama = Region[{{{0}}}];\n'.format(surf_group_string)
    s += '}\n\n\n'


    # We need to do this very ugly thing to make getdp see the conductivity as piecewise
    # constant
    m = gmsh.read_msh(fn_cond)
    cond = m.elmdata[0]
    assert cond.nr == mesh.elm.nr, \
            'The number of conductivities in the file must be' \
            +' equal to the number of elements in the mesh'

    s += 'Function {\n'
    for v in vol_regions:
        c = cond.value[mesh.elm.tag1 == v][0]
        if np.allclose(cond.value[mesh.elm.tag1 == v], c):
            if cond.nr_comp == 1:
                s += '\tsigma[V{0}] = {1};\n'.format(v, c)
            if cond.nr_comp == 9:
                s += '\tsigma[V{0}] = Tensor[{1}];\n'.format(v, ','.join(str(i) for i in c))
        else:
            s += '\tsigma[V{0}] = Field[XYZ[]]{{1}};\n'.format(v)

    '''
    for su in surf_regions:
        c = cond.value[mesh.elm.tag1 == su][0]
        if np.allclose(cond.value[mesh.elm.tag1 == su], c):
            if cond.nr_comp == 1:
                s += '\tsigma[S{0}] = {1};\n'.format(su, c)
            if cond.nr_comp == 9:
                s += '\tsigma[S{0}] = Tensor[{1}];\n'.format(su, ','.join(str(i) for i in c))
        else:
            s += '\tsigma[S{0}] = Field[XYZ[]]{{1}};\n'.format(su)
    '''

    s += '}\n\n\n'

    del m
    del cond
    gc.collect()
    return s


def solve(mesh, fn_pro,
          getdp_bin=None,
          petsc_options='-ksp_type cg -ksp_rtol 1e-10 -pc_type icc -pc_factor_levels 2',
          getdp_options='-v2 -bin',
          keepfiles=False, tmpfolder=None):

    ''' Solves a getdp-formulated problem

    Parameters:
    ---------
    mesh: simnibs.msh.gmsh_numpy.Msh or string
        If a mesh structure, will write it into a temporary file to do the simulations.
        If a string, will assume it is the name of a mesh fike
    fn_pro: str
        Name of ".pro" file
    petsc_options: string (optional)
        Petsc options to be passed to getdp
    getdp_options: string (optional)
        getdp options to be passed to getdp
    keepall: bool (optional)
        Wether to keep the intermediary files

    Returns:
    ---------
    potentials: simnibs.msh.gmsh_numpy.NodeData
        Output of the getdp calculation
    '''
    if getdp_bin is None:
        getdp_bin =os.path.join(SIMNIBSDIR, 'bin', 'getdp')

    if isinstance(mesh, gmsh.Msh):
        fn_mesh = tempfile.NamedTemporaryFile(suffix='.msh', delete=False, dir=tmpfolder).name
        gmsh.write_msh(mesh, fn_mesh)
        remove_mesh = True
    elif os.path.isfile(mesh):
        fn_mesh = os.path.abspath(os.path.expanduser(mesh))
        mesh = gmsh.read_msh(fn_mesh)
        remove_mesh = False
    else:
        raise ValueError('Invalid first argument! Must be either a file name or a mesh'
                         ' instance')
    sim_name = os.path.splitext(fn_pro)[0]
    cmd = [getdp_bin,
           fn_pro,
           '-solve', 'QS',
           '-msh', fn_mesh]

    [cmd.append(op) for op in petsc_options.split(' ')]
    [cmd.append(op) for op in getdp_options.split(' ')]
    out = run_command(cmd, realtime_output=True)
    # Looks at output and find final residual norm
    string_residuals = re.findall(
        r'(?:[KSP Residual norm  ])\d\.\d+e[\+\-]\d+', out)
    if len(string_residuals) > 1 and float(string_residuals[-1]) - float(string_residuals[-2]) > 0:
        raise RuntimeError('Solver diverged!')
        logger.warn("Solver Diverged! Final residual norm: %1.5g" %
                    float(string_residuals[-1]))

    potentials = gmsh.NodeData(
            gmsh.read_res_file(sim_name + '.res', sim_name + '.pre'),
            name='v',
            mesh=mesh)

    if not keepfiles:
        os.remove(sim_name + '.pre')
        os.remove(sim_name + '.res')
    if remove_mesh:
        os.remove(fn_mesh)
    return potentials
