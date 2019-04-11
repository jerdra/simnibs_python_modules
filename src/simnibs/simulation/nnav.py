# -*- coding: utf-8 -*-\
'''
    Structures for Simnibs.py
    This program is part of the SimNIBS package.
    Please check on www.simnibs.org how to cite our work in publications.
    Copyright (C) 2018 Andre Antunes, Guilherme B Saturnino, Axel Thielscher

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''

import os
import sys
import warnings
import pickle
from collections import namedtuple, OrderedDict
import time
import copy
from functools import partial
import copy_reg
import types
import re
import glob
import gc

import numpy as np
import scipy.io
import nibabel

from . import cond
from ..msh import transformations
from ..msh import gmsh_numpy as gmsh
from ..utils.simnibs_logger import logger
from ..utils.run_shell_command import run_command
from ..utils.run_multiprocess import run_multiprocess
from . import fem
from . import coil_numpy as coil
from . import electrode_placement
from .. import SIMNIBSDIR

# Theach pickle how to pickle instance methods
def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    if func_name.startswith('__') and not func_name.endswith('__'):  # deal with mangled names
        cls_name = cls.__name__.lstrip('_')
        func_name = '_' + cls_name + func_name
    return _unpickle_method, (func_name, obj, cls)


def _unpickle_method(func_name, obj, cls):
    for cls in cls.__mro__:
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
        return func.__get__(obj, cls)


copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)


class SESSION(object):
    """Class that defines a set of simnibs simulations

    Attributes:
    ------------------------------
    date: str
        Date and time when the session struct was initiated
    volfn: str
        file name of volume file
    vol: simnibs.simulation.nnav.VOLUME()
        Vol structure for subject
    poslist: list of nnav.SimuList()
        simulation set-up
    subpath: str
        path to m2m folder
    fnamehead: str
        file name of mesh
    pathfem: str
        path where the simulation should be saved
    fname_tensor: str
        name of DTI tensor file
    map_to_surf: bool
        Whether to map fields to middle GM surface
    map_to_fsavg: bool
        Whether to map fields to FSAverage template
    map_to_vol: bool
        Whether to map fields to a NifTI volume
    map_to_MNI: bool
        Whether to map fields to the MNI template
    fiducials: FIDUCIALS
        Structure with fiducil points
    fields: str
        Fields to be calculated for the simulations
    eeg_cap: str
        Name of eeg cap (in subject space)
    Parameters
    ------------------------
    fn_mesh: (optional) str
        mesh file name
    matlab_struct: (optional) scipy.io.loadmat()
        matlab structure

    """

    def __init__(self, fn_mesh=None, matlab_struct=None):
        # : Date when the session aws initiated
        self.date = time.strftime("%Y-%m-%d %H:%M:%S")
        self.volfn = None
        self.vol = VOLUME()
        self.poslists = []
        self.fnamehead = None
        self.subpath = None
        self.pathfem = None
        self.fname_tensor = None
        self.map_to_surf = False
        self.map_to_fsavg = False
        self.map_to_vol = False
        self.map_to_MNI = False
        self.fiducials = FIDUCIALS()
        self.fields = 'eE'
        self.eeg_cap = None

        if matlab_struct:
            self.read_mat_struct(matlab_struct)

    @property
    def type(self):
        return self.__class__.__name__

    def add_poslist(self, pl):
        """ Adds a SimList object to the poslist variable

        Parameters:
        ----------------
        pl: nnav.SimuList
            SimuList object
        """
        if not isinstance(pl, SimuList):
            raise TypeError('Elements in poslist must be subclasses of SimuList')
        self.poslists.append(pl)

    def remove_poslist(self, number):
        """Removes the specified poslist

        Parameters:
        ------------------------------
        number: int
        indice of postist to be removed
        """
        del self.poslists[number]

    def clear_poslist(self):
        """ Removes all poslists
        """
        self.poslists = []

    def prepare(self):
        """Prepares session for simulations
        relative paths are made absolute,
        empty fields are set to default values,
        check if required fields exist
        """

        self.fnamehead = os.path.abspath(os.path.expanduser(self.fnamehead))
        if not os.path.isfile(self.fnamehead):
            logger.critical('Cannot locate head mesh file: %s' % self.fnamehead)

        if not self.subpath:
            p = get_m2m_folder_from_mesh_name(self.fnamehead)
            if not p:
                logger.warning('Cannot locate subjects m2m folder')
            else:
                self.subpath = p

        else:
            self.subpath = os.path.abspath(os.path.expanduser(self.subpath))
            if not os.path.isdir(self.subpath):
                logger.warning('Cannot locate head segmentation folder: %s' % self.subpath)

        if self.subpath and not self.fname_tensor:
            self.fname_tensor = get_dir_tensor_from_m2m_folder(self.subpath)

        if self.fname_tensor and not os.path.isfile(self.fname_tensor):
            logger.warning('Cannot locate dti tensor file: %s' % self.fname_tensor)

        if self.subpath and not self.eeg_cap:
            self.eeg_cap = get_eeg_cap_from_m2m_folder(self.subpath)

        logger.info('Head Mesh: {0}'.format(self.fnamehead))
        logger.info('Subject Path: {0}'.format(self.subpath))
        self.pathfem = os.path.abspath(os.path.expanduser(self.pathfem))
        logger.info('Simulation Folder: {0}'.format(self.pathfem))

        if os.path.isfile(self.fnamehead):
            mesh = gmsh.read_msh(self.fnamehead)
            mesh.fix_surface_labels()
            for PL in self.poslists:
                PL.postprocess = self.fields
                PL.fn_tensor_nifti = self.fname_tensor
                PL.eeg_cap = self.eeg_cap
                PL.prepare()
                if not PL.mesh:
                    PL.mesh = mesh

    def run_simulatons(self, cpus=1, allow_multiple_runs=False, save_fn=None):
        self.prepare()
        dir_name = os.path.abspath(os.path.expanduser(self.pathfem))
        final_names = []

        if os.path.isdir(dir_name):
            g = glob.glob(os.path.join(dir_name, 'simnibs_simulation*.mat'))
            if len(g) > 0 and not allow_multiple_runs:
                raise IOError('Found already existing simulation results in directory.'
                              ' Please run the simulation in a new directory or delete'
                              ' the simnibs_simulation*.mat files from the folder : {0}'.format(dir_name))
            logger.info('Running simulations in the directory: {0}'.format(dir_name))
        else:
            logger.info('Running simulations on new directory: {0}'.dir_name)
            os.makedirs(dir_name)

        if save_fn:
            save_matlab_nnav(self, save_fn)

        name = os.path.split(self.fnamehead)[1]
        name = os.path.splitext(name)[0]
        for i, PL in enumerate(self.poslists):
            logger.info('Running Poslist Number: {0}'.format(i+1))
            if PL.name:
                simu_name = os.path.join(dir_name, PL.name)
            else:
                if PL.type == 'TMSLIST':
                    simu_name = os.path.join(dir_name, '{0}_TMS_{1}'.format(name, i+1))
                elif PL.type == 'TDCSLIST':
                    simu_name = os.path.join(dir_name, '{0}_TDCS_{1}'.format(name, i+1))
                else:
                    simu_name = os.path.join(dir_name, '{0}'.format(i+1))
            fn = PL.run_simulation(simu_name, cpus=cpus)
            del PL.mesh
            gc.collect()
            final_names += fn
            logger.info('Finished Running Poslist Number: {0}'.format(i+1))
            logger.info('Result Files:\n{0}'.format('\n'.join(fn)))

        folders = []
        if self.map_to_surf or self.map_to_fsavg or self.map_to_vol or self.map_to_MNI:
            if not self.subpath:
                raise IOError('Cannot run postprocessing: subpath not set')
            elif not os.path.isdir(self.subpath):
                raise IOError('Cannot run postprocessing: {0} is not a '
                              'directory'.format(self.subpath))

        if self.map_to_surf or self.map_to_fsavg:
            logger.info('Interpolating to the middle of Gray Matter')
            out_folder = os.path.join(dir_name, 'subject_overlays')
            folders += [out_folder]
            if self.map_to_fsavg:
                out_fsavg = os.path.join(dir_name, 'fsavg_overlays')
                folders += [out_fsavg]
            else:
                out_fsavg = None
            for f in final_names:
                transformations.middle_gm_interpolation(
                    f, self.subpath, out_folder,
                    out_fsaverage=out_fsavg, depth=0.5)

        if self.map_to_vol:
            logger.info('Mapping to volume')
            out_folder = os.path.join(dir_name, 'subject_volumes')
            folders += [out_folder]
            if not os.path.isdir(out_folder):
                os.mkdir(out_folder)

            for f in final_names:
                name = os.path.split(f)[1]
                name = os.path.splitext(name)[0] + '.nii.gz'
                name = os.path.join(out_folder, name)
                transformations.interpolate_to_volume(
                    f, self.subpath, name)

        if self.map_to_MNI:
            logger.info('Mapping to MNI space')
            out_folder = os.path.join(dir_name, 'mni_volumes')
            folders += [out_folder]
            if not os.path.isdir(out_folder):
                os.mkdir(out_folder)

            for f in final_names:
                name = os.path.split(f)[1]
                name = os.path.splitext(name)[0] + '.nii.gz'
                name = os.path.join(out_folder, name)
                transformations.warp_volume(
                    f, self.subpath, name)

        logger.info('=====================================')
        logger.info('SimNIBS finished running simulations')
        logger.info('Simulation Result Meshed:')
        [logger.info(f) for f in final_names]
        if len(folders) > 0:
            logger.info('Postprocessing Folders:')
            [logger.info(f) for f in folders]
        logger.info('=====================================')

        return final_names

    def read_mat_struct(self, mat):
        """ Reads form matlab structure
        Parameters
        ------------------
        mat: scipy.io.loadmat
            Loaded matlab structure
        """
        self.date = try_to_read_matlab_field(mat, 'date', str, self.date)
        self.volfn = try_to_read_matlab_field(mat, 'volfn', str, self.volfn)
        self.vol = try_to_read_matlab_field(mat, 'volfn', VOLUME, VOLUME())
        self.subpath = try_to_read_matlab_field(mat, 'subpath', str,
                                                     self.subpath)
        self.fnamehead = try_to_read_matlab_field(mat, 'fnamehead', str, self.fnamehead)
        self.pathfem = try_to_read_matlab_field(mat, 'pathfem', str, self.pathfem)
        self.fname_tensor = try_to_read_matlab_field(mat, 'fname_tensor', str,
                                                     self.fname_tensor)
        self.eeg_cap = try_to_read_matlab_field(mat, 'eeg_cap', str, self.eeg_cap)

        self.map_to_vol = try_to_read_matlab_field(
            mat, 'map_to_vol', bool, self.map_to_vol)
        self.map_to_MNI = try_to_read_matlab_field(
            mat, 'map_to_MNI', bool, self.map_to_MNI)
        self.map_to_surf = try_to_read_matlab_field(
            mat, 'map_to_surf', bool, self.map_to_surf)
        self.map_to_fsavg = try_to_read_matlab_field(
            mat, 'map_to_fsavg', bool, self.map_to_fsavg)

        self.fields = try_to_read_matlab_field(
            mat, 'fields', str, self.fields)

        self.fiducials.read_mat_struct(mat)
        if len(mat['poslist'] > 0):
            for PL in mat['poslist'][0]:
                if PL['type'][0] == 'TMSLIST':
                    self.add_poslist(TMSLIST(PL[0][0]))
                elif PL['type'][0] == 'TDCSLIST':
                    self.add_poslist(TDCSLIST(PL[0][0]))
                else:
                    raise IOError(
                        "poslist type is not of type TMSLIST or TDCSLIST")

    def nnav2mat(self):
        """ Makes a dictionary for saving a matlab structure with scipy.io.savemat()

        Returns
        --------------------
        dict
            Dictionaty for usage with scipy.io.savemat
        """
        mat = {}
        mat['type'] = 'SESSION'
        mat['date'] = remove_None(self.date)
        mat['volfn'] = remove_None(self.volfn)
        mat['subpath'] = remove_None(self.subpath)
        mat['eeg_cap'] = remove_None(self.eeg_cap)
        mat['fnamehead'] = remove_None(self.fnamehead)
        mat['pathfem'] = remove_None(self.pathfem)
        mat['fname_tensor'] = remove_None(self.fname_tensor)
        mat['vol'] = self.vol.nnav2mat()
        mat['map_to_vol'] = remove_None(self.map_to_vol)
        mat['map_to_MNI'] = remove_None(self.map_to_MNI)
        mat['map_to_fsavg'] = remove_None(self.map_to_fsavg)
        mat['map_to_surf'] = remove_None(self.map_to_surf)
        mat['fields'] = remove_None(self.fields)
        mat['fiducials'] = self.fiducials.nnav2mat()
        mat['poslist'] = []
        for PL in self.poslists:
            mat['poslist'].append(PL.nnav2mat())
            # pass

        return mat

    def add_tdcslist(self, tdcslist=None):
        ''' Appends a TDCSLIST to the current SESSION

        Parameters:
        ------------
        tdcslist: TDCSLIST (optional)
            tdcslist to be added. (Default: empty TDCSLIST)

        Returns:
        -------
        tdcslist: TDCSLIST
            the tdcslist added to this SESSION
        '''
        if tdcslist is None:
            tdcslist = TDCSLIST()

        self.poslists.append(tdcslist)
        return tdcslist

    def add_tmslist(self, tmslist=None):
        ''' Appends a TMSLIST to the current SESSION

        Parameters:
        ------------
        tmslist: TMSLIST (optional)
            tmslist to be added. (Default: empty TMSLIST)

        Returns:
        -------
        tmslist: TMSLIST
            the tmslist added to this SESSION
        '''
        if tmslist is None:
            tmslist = TMSLIST()

        self.poslists.append(tmslist)
        return tmslist

    def __str__(self):
        string = 'Subject Folder: %s\n' % self.subpath
        string += 'Mesh file name: %s\n' % self.fnamehead
        string += 'Date: %s\n' % self.date
        string += 'Number of Poslists:%d' % len(self.poslists)
        return string

    def __eq__(self, other):
        try:
            return self.__dict__ == other.__dict__
        except AttributeError:
            return False


class FIDUCIALS(object):
    def __init__(self, Nz=[], Iz=[], LPA=[], RPA=[]):
        self.Nz = Nz
        self.Iz = Iz
        self.LPA = LPA
        self.RPA = RPA

    def read_mat_struct(self, mat):
        """ Reads form matlab structure
        Parameters
        ------------------
        mat: scipy.io.loadmat
            Loaded matlab structure
        """
        self.Nz = try_to_read_matlab_field(mat, 'Nz', list, self.Nz)
        self.Iz = try_to_read_matlab_field(mat, 'Iz', list, self.Iz)
        self.LPA = try_to_read_matlab_field(mat, 'LPA', list, self.LPA)
        self.RPA = try_to_read_matlab_field(mat, 'RPA', list, self.RPA)

    def nnav2mat(self):
        """ Makes a dictionary for saving a matlab structure with scipy.io.savemat()

        Returns
        --------------------
        dict
            Dictionaty for usage with scipy.io.savemat
        """
        mat = {}
        mat['type'] = 'FIDUCIALS'
        mat['Nz'] = remove_None(self.Nz)
        mat['Iz'] = remove_None(self.Iz)
        mat['LPA'] = remove_None(self.LPA)
        mat['RPA'] = remove_None(self.RPA)
        return mat

    def from_csv(self, fn_csv):
        ''' Reads a csv file and loads the fiducials defined in it

        ----------
        fn_csv: str
            CSV file with the fields
                 Type, pos_x, pos_y, pos_z, name, whatever
            Type must be Fiducial, and name Nz, Iz, LPA, RPA
        '''
        type_, coordinates, _, name, _, _ = transformations._read_csv(fn_csv)
        for t, c, n in zip(type_, coordinates, name):
            if t == 'Fiducial':
                if n in ['Nz', 'Iz', 'LPA', 'RPA']:
                    self.__dict__[n] = c
                else:
                    logger.warning(
                        'Unrecognized Fiducial: {0} '
                        'Acceptable fiducuals are: {1}'
                        .format(n, ['Nz', 'Iz', 'LPA', 'RPA']))


class SimuList(object):
    """ Parent class

    Attributes:
    ----------------------------
    cond: list
        list of COND structures with conductivity information
    mesh: simnibs.gmsh.Msh
        Mesh where the simulations will be performed
    anisotropy_type: property, can be 'scalar', 'vn' or 'mc'
        type of anisotropy for simulation
    fn_tensor_nifti: str
        file name of nifti with tensor information
    postprocess: property
        fields to be calculated. valid fields are: 'v' , 'E', 'e', 'J', 'j', 'g', 's', 'D', 'q'
    anisotropy_vol: ndarray
        Volume with anisotropy information (lower priority over fn_tensor_nifti)
    anisotropy_affine: ndarray
        4x4 affine matrix describing the transformation from the regular grid to the mesh
        space (lower priority over fn_tensor_nifti)
    anisotropic_tissues: list
        List with tissues with anisotropic conductivities
    eeg_cap: str
        Name of csv file with EEG positions
    """

    def __init__(self, mesh=None):
        # list of conductivities (using COND class)
        self.cond = cond.standard_cond()
        self.mesh = mesh # The mesh where the simulation will be performed
        self.fn_tensor_nifti = None # File name with anisotropy information
        # The 2 variables bellow are set when the _get_vol() method is called
        # If set, they have priority over fn_tensor_nifti
        self.anisotropy_vol = None # 4-d data with anisotropy information
        self.anisotropy_affine = None # 4x4 affine transformation from the regular grid
        self.fn_nnav = None             # NNAV file that originated this poslist
        self.anisotropic_tissues = [1, 2]  # if an anisotropic conductivity is to be used,
        self.suppl = []
        self.name = None # Name to be given by simulations
        self.eeg_cap = None
        self.aniso_maxratio = 10
        self.aniso_maxcond = 2
        self._anisotropy_type = 'scalar'
        self._postprocess = ['e', 'E', 'j', 'J']

    @property
    def type(self):
        return self.__class__.__name__

    @property
    def anisotropy_type(self):
        return self._anisotropy_type

    @anisotropy_type.setter
    def anisotropy_type(self, value):
        if value not in ['scalar', 'dir', 'vn', 'mc']:
            raise ValueError("Invalid anisotroy type: {0}".format(value))
        else:
            self._anisotropy_type = value

    @property
    def postprocess(self):
        return self._postprocess

    @postprocess.setter
    def postprocess(self, value):
        valid_postptocessing = set(
            ['v', 'E', 'e', 'J', 'j', 'g', 's', 'D'])
        if not set(value).issubset(valid_postptocessing):
            raise ValueError('postprocessing operation: {0} \n'
                             'Not Supported, Supported operations are: '
                             '{1}'.format(list(set(value) - valid_postptocessing),
                                          list(valid_postptocessing)))
        else:
            self._postprocess = value

    @property
    def conductivity(self):
        c = copy.copy(self.cond)
        c.insert(0, None)
        return c

    def check_conductivities(self):
        if not self.anisotropy_type:
            self.anisotropy_type = 'scalar'

        if self.anisotropy_type not in ['vn', 'mc', 'dir', 'scalar']:
            raise ValueError(
                'Unrecognized anisotropy type:{0}'.format(self.anisotropy_type))

        if self.anisotropy_type != 'scalar' and not self.fn_tensor_nifti:
            raise IOError(
                'Cannot perform simulation with anisotropic conductivity. '
                'Invalid tensor file')

    def cond_mat_struct(self):
        """Returns a mat structure for the conductivity list

        Returns
        --------------------------
        dict
            Dictionaty for scipy.io.savemat
        """
        mat_cond = {}

        # cond data type
        cond_dt = np.dtype([('type', 'O'),
                            ('name', 'O'), ('value', 'O'), ('descrip', 'O'),
                            ('distribution_type', 'O'),
                            ('distribution_parameters', 'O')])
        cond_mat = np.empty((0,), dtype=cond_dt)

        for c in self.cond:
            c = copy.deepcopy(c)
            if not c.value:  # for data type reasons
                c.value = []
            if not c.name:  # for data type reasons
                c.name = []
            if not c.distribution_type:
                c._distribution_type = []
            cond_mat = np.append(cond_mat, np.array(
                [('COND', c.name, c.value, c.descrip,
                  c.distribution_type, c.distribution_parameters)],
                dtype=cond_dt))

        mat_cond['cond'] = cond_mat
        mat_cond['anisotropy_type'] = remove_None(self.anisotropy_type)
        mat_cond['fn_tensor_nifti'] = remove_None(self.fn_tensor_nifti)
        mat_cond['aniso_maxratio'] = remove_None(self.aniso_maxratio)
        mat_cond['aniso_maxcond'] = remove_None(self.aniso_maxcond)
        # Not really related to the conductivity
        mat_cond['name'] = remove_None(self.name)

        return mat_cond

    def read_cond_mat_struct(self, mat_struct):
        """ Reads the conductivity part of the matlab structure
        sets self.cond and set.anisotropy_type

        Parameters
        ------------------------
        mat_struct: matlab structure, as loaded by scipy.io.loadmat()

        """
        self.cond = []
        for c in mat_struct['cond'][0]:
            self.cond.append(COND(c))

        self.anisotropy_type = try_to_read_matlab_field(
            mat_struct, 'anisotropy_type', str, self.anisotropy_type)
        self.fn_tensor_nifti = try_to_read_matlab_field(
            mat_struct, 'fn_tensor_nifti', str, self.fn_tensor_nifti)
        self.name = try_to_read_matlab_field(
            mat_struct, 'name', str, self.name)
        self.aniso_maxcond = try_to_read_matlab_field(
            mat_struct, 'aniso_maxcond', float, self.aniso_maxcond)
        self.aniso_maxratio = try_to_read_matlab_field(
            mat_struct, 'aniso_maxratio', float, self.aniso_maxratio)


    def compare_conductivities(self, other):
        if self.anisotropy_type != other.anisotropy_type:
            return False

        if len(self.cond) != len(other.cond):
            return False

        for ii in range(len(self.cond)):
            if not self.cond[ii] == other.cond[ii]:
                return False
        return True

    def __eq__(self, other):
        if not isinstance(other, SimuList):
            return False

        return self.compare_conductivities(other)

    def cond2elmdata(self, mesh=None, excentricity_scale=None, logger_level=20):
        ''' Transforms a conductivity list to an ElementData field

        Parameters:
        -------
        excentricity_scale: float
            Scales the excentricity of conductivity tensors. Used in gPC simulations. Default: do not scale
            excentricities
        Returns:
        --------
        conductivity: gmsh_numpy.msh.ElementData()
            ElementData structure with conductivity information for each tetrahedron
        '''
        if mesh is None:
            mesh = self.mesh

        if mesh is None:
            raise ValueError('The mesh for this simulation is not set')

        cond_list = [c.value for c in self.cond]
        level = logger_level

        if self.anisotropy_type == 'scalar':
            logger.log(level, 'Using isotropic conductivities')
            return cond.cond2elmdata(mesh, cond_list)

        elif self.anisotropy_type == 'dir':
            logger.log(level, 'Using anisotropic direct conductivities based on the file:'
                        ' {0}'.format(self.fn_tensor_nifti))
            image, affine = self._get_vol_info()
            return cond.cond2elmdata(mesh, cond_list,
                                     anisotropy_volume=image,
                                     affine=affine,
                                     aniso_tissues=self.anisotropic_tissues,
                                     max_cond=self.aniso_maxcond,
                                     max_ratio=self.aniso_maxratio,
                                     excentricity_scaling=excentricity_scale)

        elif self.anisotropy_type == 'vn':
            logger.log(level, 'Using anisotropic volume normalized conductivities based on the file:'
                        ' {0}'.format(self.fn_tensor_nifti))
            image, affine = self._get_vol_info()
            return cond.cond2elmdata(mesh, cond_list,
                                     anisotropy_volume=image,
                                     affine=affine,
                                     aniso_tissues=self.anisotropic_tissues,
                                     normalize=True,
                                     max_cond=self.aniso_maxcond,
                                     max_ratio=self.aniso_maxratio,
                                     excentricity_scaling=excentricity_scale)

        elif self.anisotropy_type == 'mc':
            logger.log(level, 'Using isotropic mean conductivities based on the file:'
                        ' {0}'.format(self.fn_tensor_nifti))
            image, affine = self._get_vol_info()
            return cond.cond2elmdata(mesh, cond_list,
                                     anisotropy_volume=image,
                                     affine=affine,
                                     aniso_tissues=self.anisotropic_tissues,
                                     max_cond=self.aniso_maxcond,
                                     max_ratio=self.aniso_maxratio,
                                     excentricity_scaling=0.)

        else:
            raise ValueError('Invalid anisotropy_type: {0}'
                             ' valid types are: "scalar", "mc", "dir", "vn" '
                             ''.format(self.anisotropy_type))

    def _get_vol_info(self):
        if self.anisotropy_vol is not None:
            if self.anisotropy_affine is not None:
                return self.anisotropy_vol, self.anisotropy_affine

        if not self.fn_tensor_nifti:
            raise ValueError('could not get anisotropy information: '
                             'fn_tensor_nifti not set')

        fn_nifti = \
            os.path.abspath(os.path.expanduser(self.fn_tensor_nifti))
        if not os.path.isfile(fn_nifti):
            raise ValueError(
                'Could not find file \'{0}\' to get anisotropy '
                'information'.format(self.fn_tensor_nifti))

        # Load the nifti and interpolate the conductivieis
        image = nibabel.load(fn_nifti)
        affine = image.affine
        return image.dataobj, affine

    def _write_conductivity_to_hdf5(self, fn_hdf5, path='cond/'):
        """
        Parameters
        -----------
        fn_hdf5: str
            file name of hdf5 file
        path: str
            path in the hdf5 file where the conductivity information should be saved
        """
        import h5py
        with h5py.File(fn_hdf5, 'a') as f:
            try:
                g = f.create_group(path)
            except ValueError:
                g = f[path]
            value_array = np.array([c.value for c in self.cond], dtype=float)
            g.create_dataset('values', data=value_array)
            g.create_dataset('names',
                             data=np.array([c.name for c in self.cond],
                                           dtype=np.string_))
            g.create_dataset('distribution_types',
                             data=np.array(
                                 [c.distribution_type for c in self.cond],
                                 dtype=np.string_))

            distribution_parameters = np.nan * np.zeros((len(self.cond), 4), dtype=float)
            for i, c in enumerate(self.cond):
                distribution_parameters[i, :len(c.distribution_parameters)] = c.distribution_parameters
            g.create_dataset('distribution_parameters',
                             data=distribution_parameters)
            g.attrs['anisotropic_tissues'] = self.anisotropic_tissues
            g.attrs['anisotropy_type'] = self.anisotropy_type
            aniso = False
            if self.fn_tensor_nifti or self.anisotropy_vol is not None:
                image, affine = self._get_vol_info()
                aniso = True

            if aniso:
                g.create_dataset('anisotropy_vol', data=image)
                g.create_dataset('anisotropy_affine', data=affine)

    def _get_conductivity_from_hdf5(self, fn_hdf5, path='cond/'):
        '''get conductivity information from HDF5 file '''
        import h5py
        with h5py.File(fn_hdf5, 'r') as f:
            try:
                g = f[path]
            except:
                raise IOError('Could not find the group {0} '
                              'in the HDF5 file'.format(path))
            self.anisotropy_type = g.attrs['anisotropy_type']
            self.anisotropic_tissues = g.attrs['anisotropic_tissues']
            value_array = g['values'][:]
            name_array = g['names'][:]
            dist_array = g['distribution_types'][:]
            dist_p_array = g['distribution_parameters'][:]
            self.cond = [COND() for i in range(max(len(value_array), len(name_array)))]
            for i, c in enumerate(self.cond):
                if not np.isnan(value_array[i]):
                    self.cond[i].value = value_array[i]
                if name_array[i] != 'None':
                    self.cond[i].name = name_array[i]
                if dist_array[i] != 'None':
                    self.cond[i].distribution_type = dist_array[i]
                self.cond[i].distribution_parameters = \
                    dist_p_array[i][~np.isnan(dist_p_array[i])].tolist()
                                         
            try:
                self.anisotropy_affine = g['anisotropy_affine'][:]
            except KeyError:
                self.anisotropy_affine = None
            try:
                self.anisotropy_vol = g['anisotropy_vol'][:]
            except KeyError:
                self.anisotropy_vol = None

class TMSLIST(SimuList):
    """List of TMS coil position

    Note: Children of SimuList class
    Parameters
    -------------------------
    matlab_struct(optional): scipy.io.loadmat structure
        matlab structure defining the posist

    Attributes
    -------------------------
    fnamecoil: str
        Name of coil file
    poscoil: list of simnibs.simulation.nnav.POSCOIL() structures
        Definition of coil positions
    """

    def __init__(self, matlab_struct=None):
        SimuList.__init__(self)
        self.fnamecoil = ''
        self.pos = []
        self.postprocess = ['E', 'e', 'J', 'j']

        if matlab_struct:
            self.read_mat_struct(matlab_struct)

    def prepare(self):
        """Prepares structures for simulations
        Changes anisotropy_type and fnamecoil, prepares poscoil structures
        """
        self.check_conductivities()
        self.resolve_fnamecoil()
        for poscoil in self.pos:
            poscoil.eeg_cap = self.eeg_cap
            poscoil.prepare()

    def read_mat_struct(self, PL):
        """ Reads matlab poslist structure

        Changes all the fields in poslist, as well as poscoil

        Parameters:
        ------------------------------
        PL: scipy.io.loadmat structure
            Putput of loading a mat structure with scipy
        """
        self.read_cond_mat_struct(PL)

        try:
            self.fnamecoil = str(PL['fnamecoil'][0])
        except:
            self.fnamecoil = ''

        if len(PL['pos']) > 0:
            for pos in PL['pos'][0]:
                self.pos.append(POSITION(pos))

        try:
            self.suppl = PL['suppl'][0]
        except:
            pass

    def nnav2mat(self):
        """ Dictionaty for saving as a matlab structure with scipy.io.savemat

        Returns:
        ----------------------------
        dict
            Dictionary with poslist parameters for saving into a matlab structure

        """
        mat_poslist = self.cond_mat_struct()

        mat_poslist['type'] = 'TMSLIST'
        mat_poslist['fnamecoil'] = remove_None(self.fnamecoil)
        mat_poslist['suppl'] = remove_None(self.suppl)

        # pos data type
        pos_dt = np.dtype([('type', 'O'),
                           ('name', 'O'), ('date', 'O'),
                           ('istrig', 'O'), ('matORG', 'O'), ('orient', 'O'),
                           ('matsimnibs', 'O'), ('didt', 'O'),
                           ('fnamefem', 'O'), ('centre', 'O'),
                           ('pos_ydir', 'O'), ('distance', 'O')])

        pos_mat = np.empty((0,), dtype=pos_dt)
        for pos in self.pos:
            pos_array = np.array([('POSITION', remove_None(pos.name),
                                   remove_None(pos.date),
                                   remove_None(pos.istrig),
                                   remove_None(pos.matORG),
                                   remove_None(pos.orient),
                                   remove_None(pos.matsimnibs),
                                   remove_None(pos.didt),
                                   remove_None(pos.fnamefem),
                                   remove_None(pos.centre),
                                   remove_None(pos.pos_ydir),
                                   remove_None(pos.distance))],
                                 dtype=pos_dt)
            pos_mat = np.append(pos_mat, pos_array)

        mat_poslist['pos'] = pos_mat

        return mat_poslist

    def add_positions_from_csv(self, fn_csv):
        ''' Reads a csv file and adds the positions defined to the tmslist

        Parameters
        ----------
        fn_csv: str
            CSV file with the fields
                Type, pos_x, pos_y, pos_z, ez_x, ez_y, ez_z, ey_x, ey_y, ey_z, dist, name, ...
                "Type" needs to be CoilPos. The positions are in subject space. The
                transformations module can transfrom from MNI to subject space
        '''
        type_, coordinates, extra, name, _, _ = transformations._read_csv(fn_csv)
        for t, c, e, n in zip(type_, coordinates, extra, name):
            if t == 'CoilPos':
                p = POSITION()
                vz = e[:3]
                vy = e[3:6]
                vx = np.cross(vy, vz)
                mat = np.eye(4)
                mat[:3, 0] = vx
                mat[:3, 1] = vy
                mat[:3, 2] = vz
                mat[:3, 3] = c
                p.matsimnibs = mat.tolist()
                p.name = n
                self.pos.append(p)

    def resolve_fnamecoil(self):
        fnamecoil = os.path.expanduser(self.fnamecoil)
        if os.path.isfile(fnamecoil):
            self.fnamecoil = fnamecoil
        else:
            fnamecoil = os.path.join(
                SIMNIBSDIR, 'ccd-files', self.fnamecoil)
            if os.path.isfile(fnamecoil):
                self.fnamecoil = fnamecoil
            else:
                raise IOError(
                    'Could not find coil file: {0}'.format(self.fnamecoil))

    def _get_simu_name(self, simu_name, simu_nr):
        coil_name = os.path.splitext(os.path.basename(self.fnamecoil))[0]
        if coil_name.endswith('.nii'):
            coil_name = coil_name[:-4] + '_nii'

        simu_name = simu_name +\
            "-{0:0=4d}_{1}".format(simu_nr+1, coil_name)
        return simu_name

    def _run_position(self, simu_nr, simu_name, fn_cond, keepall, tmpfolder):
        """ Thes function is ment to be used by run_simulation for paralelization
        """
        logger.info('Running TMS Position {0}'.format(simu_nr))
        pos = self.pos[simu_nr]
        logger.info(str(pos))
        pos.calc_matsimnibs(self.mesh)
        simu_name = self._get_simu_name(simu_name, simu_nr)
        pro_file_fn = simu_name + '.pro'
        dadt_fn = simu_name + '_dadt.msh'
        out_fn = simu_name + '_' + self.anisotropy_type + '.msh'
        dadt = coil.set_up_tms(self.mesh,
                               self.fnamecoil,
                               pos.matsimnibs,
                               didt=pos.didt,
                               fn_out=dadt_fn,
                               fn_geo=simu_name + '_coil_pos.geo')

        v = fem.tms_getdp(self.mesh, fn_cond, dadt_fn, pro_file_fn,
                          keepall, tmpfolder=tmpfolder)

        cond = gmsh.read_msh(fn_cond).elmdata[0]
        m = fem.calc_fields(v, self.postprocess, cond=cond, dadt=dadt.nodedata[0])
        gmsh.write_msh(m, out_fn)
        del m
        if not keepall:
            map(os.remove, [pro_file_fn, dadt_fn])


    def run_simulation(self, fn_simu, keepall=False, cpus=1):
        """ Runs the TMS simulation defined by this structure

        Parameters
        ----------
        keepall(optional): bool
            Wether or not to keep all intermediary files
        Returns:
        ---------
        Final output file: fn_simu + '.msh'
        """
        if len(self.pos) == 0:
            raise ValueError('There are no positions defined for this poslist!')
        fn_simu = os.path.abspath(os.path.expanduser(fn_simu))
        assert isinstance(self.mesh, gmsh.Msh), \
            'mesh property not set appropriately!'

        for c in self.cond:
            if c.distribution_type:
                logger.warning('Distribution value for conductivity found, starting gPC')
                self.run_gpc(self, fn_simu)

        logger.info('Began to run TMS simulations')
        logger.info('Using coil: {0}'.format(self.fnamecoil))
        self.prepare()
        path, basename = os.path.split(fn_simu)

        if not os.path.isdir(path) and path != '':
            os.mkdir(path)

        cond = self.cond2elmdata()
        self.mesh.elmdata = [cond]
        fn_cond = fn_simu + '_cond.msh'
        gmsh.write_msh(self.mesh, fn_cond)
        self.mesh.elmdata = []
        # Multiprocessing version
        partialized = partial(self._run_position, simu_name=fn_simu,
                              keepall=keepall, fn_cond=fn_cond, tmpfolder=path)
        if sys.platform == 'win32':
            # disable Multiprocessing on Windows
            map(partialized, range(len(self.pos)))
        else:
            args = [(i, ) for i in range(len(self.pos))]
            results = run_multiprocess(partialized, args, cpus)
            if any([r != 0 for r in results]):
                raise RuntimeError('There was a problem during simulation, see the log'
                                   'file for more detais')
        names = [self._get_simu_name(fn_simu, i) + '_' + self.anisotropy_type + '.msh' for i in
                 range(len(self.pos))]

        for p, n in zip(self.pos, names):
            p.fnamefem = n

        if not keepall:
            os.remove(fn_cond)
        gc.collect()
        return names

    def run_gpc(self, fn_simu, tissues=[2], eps=1e-3):
        try:
            from simnibs.simulation.gpc import run_tms_gpc
            gPC_regression = run_tms_gpc(self, fn_simu, tissues, eps)
            return gPC_regression
        except ImportError:
            raise NotImplementedError('GPC implementation not complete yet!')

    def _run_gpc_simulation(self, random_variables,
                            identifiers, fn_simu,
                            simu_nr,
                            tissues=[2], keepall=False):
        try:
            from simnibs.simulation.gpc import _run_tms_gpc_simulation
            return _run_tms_gpc_simulation(
                self, random_variables, identifiers, fn_simu, simu_nr, tissues, keepall)
        except ImportError:
            raise NotImplementedError('GPC implementation not complete yet!')

    def add_position(self, position=None):
        ''' Adds a position to the current TMSLIST

        Parameters
        -----
        position: POSITION (Optional)
            Position structure defining the coil position (Default: empty POSITION())

        Returns
        ------
        position: POSITION
            POSITION structure defining the coil position
        '''
        if position is None:
            position = POSITION()

        self.pos.append(position)
        return position


    def __str__(self):
        string = "type: {0} \n" \
                 " fnamecoil: {1}, \n" \
                 " nr coil positions: {2} \n"\
                 " anisotropy: {3}"\
                 "".format(self.type,
                           self.fnamecoil,
                           len(self.pos),
                           self.anisotropy_type)
        return string

    def __eq__(self, other):
        if not isinstance(other, TMSLIST):
            return False

        if self.type != other.type:
            return False

        if self.anisotropy_type != other.anisotropy_type:
            return False

        if len(self.cond) != len(other.cond):
            return False

        for ii in range(len(self.cond)):
            if not self.cond[ii] == other.cond[ii]:
                return False

        if len(self.pos) != len(other.pos):
            return False

        for ii in range(len(self.pos)):
            if not self.pos[ii] == other.pos[ii]:
                return False
        return True


class POSITION(object):
    """ TMS coil position, orientation, name and dI/dt

    Parameters:
    -----------------------------------------
    matlab_struct(optional): output from scipy.io.loadmat
        Matlab structure defining the position

    Attributes:
    ---------------------------------------
    name: str
        name of position
    date: str
        date when stimulation was done
    matORG: list
        4x4 matrix defining coil center and orientation
        in the original coordinate system
    matsimnibs: list with 16 numbers
        4x4 matrix defining coil center and orientation
        in simnibs coordinate system.
        HAS PREFERENCE OVER (centre, pos_y, distance)
    dIdt: float
        Change of current in coil, in A/s
    fnamefem: str
        Name of simulation output
    ---
    The 3 variables bellow offer an alternative way to set-up a simulation.
    matsimnibs has preference over them.
    ---
    centre: list or str
        Center of the coil. Will be projected in the head surface.
        if a string, also define an eeg_cap
    pos_ydir: list of str
        Reference position for the prolongation of the coil handle.
        Will be projected in the head surface. if a string, also define an eeg_cap
    distance: float
        Distance to head
    """

    def __init__(self, matlab_struct=None):
        self.name = ''
        self.date = None
        self.istrig = False
        self.matORG = None
        self.orient = ''
        self.matsimnibs = None
        self.didt = 1e6  # corresponding to 1 A/us
        self.fnamefem = ''
        self.centre = None
        self.pos_ydir = None
        self.distance = 4.
        self.eeg_cap = None

        if matlab_struct is not None:
            self.read_mat_struct(matlab_struct)

    def prepare(self):
        """ Prepares structure for simulation
        """
        if len(self.name):
            self.pos_name = '-' + self.name.translate(None, ' ')
            self.pos_name = self.pos_name.replace('_', '-')
        if self.didt == 1:
            logger.warning("dI/dt = 1, Most likely units were set in A/us")

    # read TMS coil position structure
    def read_mat_struct(self, p):
        """ Reads matlab structure

        Parameters
        --------------------------
        p: scipy.io.loadmat
            strucure as loaded by scipy
        """
        self.name = try_to_read_matlab_field(p, 'name', str, self.name)
        self.date = try_to_read_matlab_field(p, 'date', str, self.date)
        self.didt = try_to_read_matlab_field(p, 'didt', float, self.didt)
        self.istrig = try_to_read_matlab_field(p, 'istrig', bool, self.istrig)
        self.orient = try_to_read_matlab_field(p, 'orient', str, self.orient)
        self.fnamefem = try_to_read_matlab_field(p, 'fnamefem', str, self.fnamefem)
        try:
            self.matORG = (p['matORG'][0]).tolist()
        except:
            pass
        try:
            self.matsimnibs = (p['matsimnibs']).tolist()
        except:
            pass
        self.centre = try_to_read_matlab_field(p, 'centre', list, self.centre)
        self.centre = try_to_read_matlab_field(p, 'center', list, self.centre)
        self.pos_ydir = try_to_read_matlab_field(p, 'pos_ydir', list, self.pos_ydir)
        self.distance = try_to_read_matlab_field(p, 'distance', float, self.distance)

        # Parse string values for centre and pos_ydir
        if sys.version_info >= (3, 0):
            clss = str
        else:
            clss = basestring

        if self.centre and isinstance(self.centre[0], clss):
            self.centre = ''.join(self.centre)

        if self.pos_ydir and isinstance(self.pos_ydir[0], clss):
            self.pos_ydir = ''.join(self.pos_ydir)

    def substitute_positions_from_cap(self, cap=None):
        if cap is None:
            cap = self.eeg_cap
        self.centre = _substitute(self.centre, cap)
        self.pos_ydir = _substitute(self.pos_ydir, cap)

    def matsimnibs_is_defined(self):
        if isinstance(self.matsimnibs, np.ndarray):
            if self.matsimnibs.ndim == 2 and\
               self.matsimnibs.shape == (4,4):
                return True
        elif self.matsimnibs and \
           np.array(self.matsimnibs).ndim == 2 and\
           np.array(self.matsimnibs).shape == (4, 4):
            return True
        else:
            return False

    def calc_matsimnibs(self, msh, cap=None):
        if cap is None:
            cap = self.eeg_cap
        if self.matsimnibs_is_defined():
            return self.matsimnibs
        else:
            logger.info('Calculating Coil position fromm (centre, pos_y, distance)')
            if not self.centre:
                raise ValueError('Coil centre not set!')
            if not self.pos_ydir:
                raise ValueError('Coil pos_ydir not set!')
            if not self.distance:
                raise ValueError('Coil distance not set!')
            self.substitute_positions_from_cap(cap=cap)
            self.matsimnibs = msh.calc_matsimnibs(
                self.centre, self.pos_ydir, self.distance)
            logger.info('Matsimnibs: \n{0}'.format(self.matsimnibs))
            return self.matsimnibs

    def __eq__(self, other):
        if self.name != other.name or self.date != other.date or \
           self.mat != other.mat or self.orient != other.orient or \
           self.matsimnibs != other.matsimnibs or self.didt != other.didt:
            return False

        else:
            return True

    def __str__(self):
        s = 'Coil Position Matrix: {0}\n'.format(self.matsimnibs)
        s += 'dIdt: {0}\n'.format(self.didt)
        if not self.matsimnibs_is_defined():
            s += 'Centre: {0}\n'.format(self.centre)
            s += 'pos_ydir: {0}\n'.format(self.pos_ydir)
            s += 'distance: {0}\n'.format(self.distance)
        return s


class COND(object):
    """ conductivity information
    Conductivity information for simulations

    Attributes:
    ---------------------
    name: str
        Name of tissue
    value: float
        value of conductivity
    descrip: str
        description of conductivity
    distribution_type: 'uniform', 'normal', 'beta' or None
        type of distribution for gPC simulation
    distribution_parameters: list of floats
        if distribution_type is 'uniform': [min_value, max_value]
        if distribution_type is 'normal': [mean, standard_deviation]
        if distribution_type is 'beta': [p, q, min_value, max_value]
    """

    def __init__(self, matlab_struct=None):
        self.name = None      # e.g. WM, GM
        self.value = None     # in S/m
        self.descrip = ''
        self._distribution_type = None
        self.distribution_parameters = []

        if matlab_struct:
            self.read_mat_struct(matlab_struct)

    @property
    def distribution_type(self):
        return self._distribution_type

    @distribution_type.setter
    def distribution_type(self, dist):
        if dist == '':
            dist = None
        if dist in ['uniform', 'normal', 'beta', None]:
            self._distribution_type = dist
        else:
            raise ValueError('Invalid distribution type')

    def read_mat_struct(self, c):
        try:
            self.name = str(c['name'][0])
        except:
            pass

        try:
            self.value = c['value'][0][0]
        except:
            self.value = None

        try:
            self.descrip = str(c['descrip'][0])
        except:
            pass

        try:
            self.distribution_type = str(c['distribution_type'][0])
        except:
            pass

        try:
            self.distribution_parameters = c['distribution_parameters'][0]
        except:
            pass

    def __eq__(self, other):
        if self.name != other.name or self.value != other.value:
            return False
        else:
            return True

    def __str__(self):
        s = "name: {0}\nvalue: {1}\ndistribution: {2}\ndistribution parameters: {3}".format(
            self.name, self.value, self.distribution_type, self.distribution_parameters)
        return s


class TDCSLIST(SimuList):
    """Structure that defines a tDCS simulation

    Parameters
    ----------------------------------------
    matlab_struct(optinal): dict
        matlab structure as red with scipy.io, not compressed

    Attributes
    ------------------------------------------
    currets: list of floats
        current in each channel
    electrode: list of nnav.ELECTRODE structures
        electrodes
    cond: list
        list of COND structures with conductivity information
    anisotropy_type: property, can be 'scalar', 'vn' or 'mc'
        type of anisotropy for simulation
    postprocess: property
        fields to be calculated. valid fields are: 'v' , 'E', 'e', 'J', 'j', 'g', 's', 'D', 'q'
    """
    def __init__(self, matlab_struct=None):
        SimuList.__init__(self)
        # currents in A (not mA!; given per stimulator channel)
        self.currents = []
        self.electrode = []
        self.fnamefem = ''
        self.postprocess = 'eEjJ'

        # internal to simnibs
        self.tdcs_msh_name = None
        self.tdcs_msh_electrode_name = None

        if matlab_struct is not None:
            self.read_mat_struct(matlab_struct)

    @property
    def unique_channels(self):
        return list(set([el.channelnr for el in self.electrode]))

    def prepare(self):
        if None in self.unique_channels:
            raise ValueError('Found a None in an Electrode Channel number',
                             'please connect all electrodes to a channel')

        if len(self.unique_channels) != len(self.currents):
            raise ValueError("Number of channels should correspond to" +
                             "the size of the currents array:\n" +
                             "unique channels:" + str(self.unique_channels) + " "
                             "Currents:" + str(self.currents))

        for i in self.unique_channels:
            while len(self.cond) < 500 + i:
                self.cond.append(COND())
            if not self.cond[99 + i].name:
                self.cond[99 + i].name = 'el' + str(i)
                self.cond[99 + i].value = self.cond[99].value
            if not self.cond[499 + i].name:
                self.cond[499 + i].name = 'gel_sponge' + str(i + 1)
                self.cond[499 + i].value = self.cond[499].value

        self.check_conductivities()
        if not np.isclose(np.sum(self.currents), 0):
            raise ValueError('Sum of currents should be zero!')

        if np.allclose(self.currents, 0):
            logger.warning('All current values are set to zero!')

        if len(self.unique_channels) != len(self.currents):
            raise ValueError('Number of unique channels is not equal to the number of'
                             'current values')

        for electrode in self.electrode:
            electrode.eeg_cap = self.eeg_cap

    def read_mat_struct(self, PL):
        self.read_cond_mat_struct(PL)
        self.currents = try_to_read_matlab_field(PL, 'currents', list, self.currents)
        self.fnamefem = try_to_read_matlab_field(PL, 'fnamefem', str, self.fnamefem)

        if len(PL['electrode']) > 0:
            for el in PL['electrode'][0]:
                self.electrode.append(ELECTRODE(el))

    def nnav2mat(self):
        mat_poslist = self.cond_mat_struct()
        mat_poslist['type'] = 'TDCSLIST'
        mat_poslist['currents'] = remove_None(self.currents)
        mat_poslist['fnamefem'] = remove_None(self.fnamefem)

        def save_electrode_mat(electrode_list):
            elec_dt = np.dtype([('type', 'O'),
                                ('name', 'O'), ('definition', 'O'),
                                ('shape', 'O'), ('centre', 'O'),
                                ('dimensions', 'O'),
                                ('pos_ydir', 'O'), ('vertices', 'O'),
                                ('thickness', 'O'),
                                ('channelnr', 'O'), ('holes', 'O'),
                                ('plug', 'O'), ('dimensions_sponge', 'O')])

            elec_mat = np.empty((0,), dtype=elec_dt)

            for elec in electrode_list:
                holes = save_electrode_mat(elec.holes)
                plug = save_electrode_mat(elec.plug)
                elec_array = np.array([('ELECTRODE',
                                        remove_None(elec.name),
                                        remove_None(elec.definition),
                                        remove_None(elec.shape),
                                        remove_None(elec.centre),
                                        remove_None(elec.dimensions),
                                        remove_None(elec.pos_ydir),
                                        remove_None(elec.vertices),
                                        remove_None(elec.thickness),
                                        remove_None(elec.channelnr),
                                        remove_None(holes),
                                        remove_None(plug),
                                        remove_None(elec.dimensions_sponge))],
                                      dtype=elec_dt)
                elec_mat = np.append(elec_mat, elec_array)

            return elec_mat

        mat_poslist['electrode'] = save_electrode_mat(self.electrode)
        return mat_poslist

    def add_electrode(self, electrode=None):
        ''' Adds an electrode to the current TDCSLIST

        Parameters
        -----
        electrode: ELECTRODE (Optional)
            Electrode structure. (Default: empty ELECTRODE)

        Returns
        ------
        electrode: ELECTRODE
            electrode structure added to this TDCSLIST
        '''
        if electrode is None:
            electrode = ELECTRODE()
        self.electrode.append(electrode)
        return electrode

    def add_electrodes_from_csv(self, fn_csv):
        ''' Reads a csv file and adds the electrodes defined to the tdcslist

        How this function behaves depends on how many electrodes are defined
        in the structure.
        0 electrodes: New electrode structures are created
        1 electrode: The electrode structured is copyed over. The electrode centre and
                     y_dir are changed according to the information in the CSV
        N electrodes: N must be the same number of electrodes defined in fn_csv.
                      Keeps the electrode geometry
        ----------
        fn_csv: str
            CSV file with the fields
                 Type, pos_x, pos_y, pos_z, name, whatever
                 Type, pos_x, pos_y, pos_z, pos2_x, pos2_y, pos2_z, name, ...
            Type must be Electrode or ReferenceElectrode. "pos_x, pos_y, pos_z" are the
            positions of the electrode, and "pos2_x, pos2_y, pos2_z" are the reference
            positions
        '''
        type_, coordinates, extra, name, _, _ = transformations._read_csv(fn_csv)
        count_csv = len([t for t in type_ if t in ['Electrode', 'ReferenceElectrode']])
        count_struct = len(self.electrode)
        if count_struct == 0:
            self.electrode = [ELECTRODE() for i in range(count_csv)]
        elif count_struct == 1:
            self.electrode = [copy.deepcopy(self.electrode[0]) for i in range(count_csv)]
        elif count_struct != count_csv:
            raise IOError('The number of electrodes in the structure in not 0, 1 or the'
                          ' same number as in the CSV file')
        i = 0
        ref_idx = None
        for t, c, e, n in zip(type_, coordinates, extra, name):
            if t in ['Electrode', 'ReferenceElectrode']:
                el = self.electrode[i]
                el.centre = c
                el.name = n
                if e is not None:
                    el.pos_ydir = e
                else:
                    el.pos_ydir = []
                if t == 'ReferenceElectrode':
                    ref_idx = i
                i += 1

        if ref_idx is not None:
            self.electrode[0], self.electrode[ref_idx] = \
                self.electrode[ref_idx], self.electrode[0]

        for i, el in enumerate(self.electrode):
                el.channelnr = i + 1

    def place_electrodes(self):
        """ Add the defined electrodes to a mesh

        Parameters:
        ------------
        fn_out: str
            name of output file
        """
        w_elec = copy.deepcopy(self.mesh)
        w_elec.fix_tr_node_ordering()
        for el in self.electrode:
            logger.info('Placing Electrode:\n{0}'.format(str(el)))
            w_elec = el.add_electrode_to_mesh(w_elec)

        w_elec.fix_thin_tetrahedra()
        gc.collect()
        return w_elec

    def run_simulation(self, fn_simu, keepall=False, cpus=1):
        """ Runs the tDCS simulation defined by this structure

        Parameters
        ----------
        fn_msh: str
            filename of mesh
        fn_simu: str
            simulation path and name
        keepall(optional): bool
            Wether or not to keep all intermediary files
        Returns:
        ---------
        Final output file: fn_simu + '.msh'
        """
        fn_simu = os.path.abspath(os.path.expanduser(fn_simu))
        if cpus != 1:
            logger.warning('tDCS simulations not parallelized yet')

        if not self.mesh:
            raise ValueError('The mesh for this simulation is not set')

        for c in self.cond:
            if c.distribution_type:
                logger.warning('Distribution value for conductivity found, starting gPC')
                self.run_gpc(self, fn_simu)

        logger.info('Began to run tDCS simulation')
        logger.info('Channels: {0}'.format(self.unique_channels))
        logger.info('Currents (A): {0}'.format(self.currents))
        self.prepare()
        path, basename = os.path.split(fn_simu)

        if not os.path.isdir(path) and path != '':
            os.mkdir(path)

        fn_no_extension, extension = os.path.splitext(fn_simu)
        mesh_elec = self.place_electrodes()
        fn_elec = fn_simu + '_elec.msh'
        gmsh.write_msh(mesh_elec, fn_elec)
        cond = self.cond2elmdata(mesh_elec)
        electrode_surfaces = [2100 + c for c in self.unique_channels]
        fn_pro = [fn_simu + '_{0}-{1}.pro'.format(self.unique_channels[0], u) for u in
                  self.unique_channels[1:]]
        v = fem.tdcs_getdp(mesh_elec, cond, self.currents,
                           electrode_surfaces, fn_pro,
                           keepfiles=keepall, tmpfolder=path)
        m = fem.calc_fields(v, self.postprocess, cond=cond)
        final_name = fn_simu + '_' + self.anisotropy_type + '.msh'
        gmsh.write_msh(m, final_name)
        self.fnamefem = final_name

        if not keepall:
            os.remove(fn_elec)
            [os.remove(f) for f in fn_pro]
        del m
        gc.collect()
        return [final_name]

    def run_leadfield(self, fn_simu,
                      tissues=[2], keepall=False,
                      PutElectrodeOnHead_bin='PutElectrodeOnHead',
                      fix_thin_tetrahedra_bin='fix_thin_tetrahedra',
                      sum_and_scale_fields_bin='sum_and_scale_fields',
                      cpus=1, slurm=False, use_partial_run=False,
                      **run_getdp_options):
        try:
            from simnibs.optimize.run_leadfield import run_leadfield
            return run_leadfield(
                self, fn_simu, tissues, keepall, PutElectrodeOnHead_bin,
                fix_thin_tetrahedra_bin, sum_and_scale_fields_bin, cpus, slurm,
                use_partial_run, **run_getdp_options)
        except ImportError:
            raise NotImplementedError('Implementation not complete yet!')

    def run_gpc(self, fn_simu, tissues=[2], eps=1e-3):
        try:
            from simnibs.simulation.gpc import run_tcs_gpc
            self.prepare()
            gPC_regression = run_tcs_gpc(self, fn_simu, tissues, eps)
            return gPC_regression
        except ImportError:
            raise NotImplementedError('GPC implementation not complete yet!')

    def __str__(self):
        string = "Numer of electrodes: {0}\n".format(len(self.electrode)) + \
                 "Currents: {0}".format(self.currents)
        return string

    def __eq__(self, other):
        if self.type != other.type:
            return False

        value = self.compare_conductivities(other)

        if len(self.electrode) != len(other.electrode):
            return False

        for elec1, elec2 in zip(self.electrode, other.electrode):
            value *= elec1 == elec2

        value *= self.currents == other.currents
        value *= self.fn_nnav == other.fn_nnav

        return value


class ELECTRODE(object):
    """ Class defining tDCS electrodes

    Attributes:
    ------------
    name: str
        Electrode name
    definition: 'plane' or 'conf'
        Where the electrode is defined: in a plane or in confom (suject) space
    shape: 'rect', 'ellipse' or 'custom'
        shape of electrode (plane)
    centre: list
        centre of electrode (in subject space) (plane)
    pos_ydir: list
        position in the head to define Y direction in the electrode. The Y direction is
        from the center to pos_ydir (plane)
    dimensions: list
        dimensions (x, y) of the electrode
    vertices: nx2 ndarray
        List of vertice positions in a plane, used to define custom electrodes(plane)
    thickness: int or list
        List of thickness. The number of elements specifies the type of electrode
        (conf+plane)
        1 element: simple electrode
        2 elements: electrode + gel
        3 elements: electrode in sponge
    dimensions_sponge: list
        dimensions (x, y) of the electrode

    """

    def __init__(self, matlab_struct=None):
        self.name = ''
        # how the electrode's position is evaluated: plane (2D) or conf
        # (vertices - 3D)
        self.definition = 'plane'

        # for definition = 'plane'
        self._shape = ''  # 'rect' or 'ellipse' or 'custom' for the general polygon
        self.centre = None  # centre of the 2D plane in conf coordinates
        self.dimensions = None  # for rectangle / ellipse
        self.pos_ydir = []  # second position used to define electrode orientation

        # for definition = 'plane' and 'conf'
        self.vertices = []  # array of vertices (nx2) for S.definition=='plane'
        # or (nx3) for S.definition=='conf'
        # common fields:
        # can have up to 3 arguments. 1st argument is the lower part of the
        # electrode
        self.thickness = []
        # 2nd arrgument is the rubber; 3rd is the upper
        # layer
        self.channelnr = None
        self.holes = []
        self.plug = []
        self.dimensions_sponge = None
        self.eeg_cap = None # is overwitten by TDCSLIST.prepare()

        if matlab_struct is not None:
            self.read_mat_struct(matlab_struct)

    @property
    def type(self):
        return self.__class__.__name__

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, s):
        if s in ['rectangle', 'rect', 'ellipse', 'custom', '']:
            self._shape = s
        else:
            raise ValueError('Electrode shape must be'
                             '\'rect\', \'ellipse\', \'custom\' or \'\'')

    def prepare(self):
        self.thickness = np.atleast_1d(self.thickness)
        if len(self.thickness) == 0:
            raise ValueError("Electrode thickness not defined!")

        if self.channelnr is None:
            logger.warning('Please connect the electrode to a channel')


    def read_mat_struct(self, el):
        self.name = try_to_read_matlab_field(el, 'name', str, self.name)
        self.definition = try_to_read_matlab_field(el, 'definition', str, self.definition)
        self.shape = try_to_read_matlab_field(el, 'shape', str, self.shape)
        self.dimensions = try_to_read_matlab_field(el, 'dimensions', list,
                                                   self.dimensions)
        self.centre = try_to_read_matlab_field(el, 'centre', list, self.centre)
        self.centre = try_to_read_matlab_field(el, 'center', list, self.centre)
        self.pos_ydir = try_to_read_matlab_field(el, 'pos_ydir', list, self.pos_ydir)
        self.thickness = try_to_read_matlab_field(el, 'thickness', list, self.thickness)
        self.channelnr = try_to_read_matlab_field(el, 'channelnr', int, self.channelnr)
        self.dimensions_sponge = try_to_read_matlab_field(el, 'dimensions_sponge', list,
                                                          self.dimensions_sponge)
        try:
            self.vertices = el['vertices'].tolist()
        except:
            # array of vertices (nx2) for S.definition=='plane'
            self.vertices = []

        try:
            for h in el['holes']:
                if len(h) > 0:
                    self.holes.append(ELECTRODE(h[0]))
        except ValueError:
            pass

        try:
            for p in el['plug']:
                if len(p) > 0:
                    self.plug.append(ELECTRODE(p[0]))
        except ValueError:
            pass

        if not self.definition or self.definition == '[]': # simplify matlab synthax
            self.definition = 'plane'

        # Parse string values for centre and pos_ydir
        if sys.version_info >= (3, 0):
            clss = str
        else:
            clss = basestring

        if self.centre and isinstance(self.centre[0], clss):
            self.centre = ''.join(self.centre)

        if self.pos_ydir and isinstance(self.pos_ydir[0], clss):
            self.pos_ydir = ''.join(self.pos_ydir)

    def add_electrode_to_mesh(self, mesh):
        """ Uses information in the structure in order to place an electrode

        Parameters:
        ------------
        mesh: simnibs.msh.gmsh_numpy.Msh
            Mesh where the electrode is to be placed
        """
        self.prepare()
        return electrode_placement.put_electrode_on_mesh(self, mesh,
                                                         100 + self.channelnr)

    def add_hole(self, hole=None):
        ''' Adds a hole to the current Electrode

        Parameters
        -----
        hole: ELECTRODE (Optional)
            Electrode structure defining the hole (Default: empty ELECTRODE)

        Returns
        ------
        hole: ELECTRODE
            electrode structure defining the hole
        '''
        if hole is None:
            hole = ELECTRODE()
        self.holes.append(hole)
        return hole

    def add_plug(self, plug=None):
        ''' Adds a plug to the current Electrode

        Parameters
        -----
        plug: ELECTRODE (Optional)
            Electrode structure defining the plug (Default: empty ELECTRODE)

        Returns
        ------
        plug: ELECTRODE
            electrode structure defining the plug
        '''
        if plug is None:
            plug = ELECTRODE()
        self.plug.append(plug)
        return plug

    def __str__(self):
        string = "definition: {0}\n".format(self.definition)
        if self.definition == 'plane':
            string += "shape: {0}\n".format(self.shape)
            string += "centre: {0}\n".format(self.centre)
            string += "pos_ydir: {0}\n".format(self.pos_ydir)
            string += "dimensions: {0}\n".format(self.dimensions)
        string += "thickness:{0}\n".format(self.thickness)
        if self.definition == 'conf' or self.shape == 'custom':
            string += "vertices: {0}\n".format(self.vertices)
        string += "channelnr: {0}\n".format(self.channelnr)
        string += "number of holes: {0}\n".format(len(self.holes))
        return string

    def substitute_positions_from_cap(self, cap=None):
        if cap is None:
            cap = self.eeg_cap
        for h in self.holes:
            h.eeg_cap = self.eeg_cap
            h.substitute_positions_from_cap(cap=cap)
        for p in self.plug:
            p.eeg_cap = self.eeg_cap
            p.substitute_positions_from_cap(cap=cap)

        self.centre = _substitute(self.centre, cap)
        self.pos_ydir = _substitute(self.pos_ydir, cap)

    def __eq__(self, other):
        try:
            return self.__dict__ == other.__dict__

        except AttributeError:
            return False


class VOLUME:
    def __init__(self, matlab_struct=None):
        self.org = []  #used for parsing neuronavigation data; not stored permanently; optional
        self.fname=''  #string; points towards neuronavigation file specifying details of structural MRI; optional
        self.ftype=''  #string; file-type of the T1 used by the nnav-system ('NIFTI' or 'DICOM')
        self.manufacturer='unknown'  #string; currently, only 'LOCALITE' is supported
        self.volfiles=[]  #list of files of the T1 (one file for NIFTI; many for DICOM)
        self.img=[]  #used to temporarily store the T1; can be deleted after coregistration to simnibs T1
        self.voxsize=[]  #voxel size of the T1
        self.dim=[]  #dimensions of the T1
        self.m_qform=[]  #qform of the T1
        self.fname_conf=''  #path and filename of the simnibs T1 of the subject
        self.m_toconform=[]  #4x4 transformation matrix from nnav T1 to simnibs T1 (for mm-to-mm mapping of real world coordinates)

        if matlab_struct:
            self.read_mat_struct(matlab_struct)

    def read_mat_struct(self, v):
        self.org = try_to_read_matlab_field(v, 'org', list, self.org)
        self.fname = try_to_read_matlab_field(v, 'fname', str, self.fname)
        self.ftype = try_to_read_matlab_field(v, 'ftype', str, self.ftype)
        self.manufacturer = try_to_read_matlab_field(v, 'manufacturer', str,
                                                     self.manufacturer)
        self.volfiles = try_to_read_matlab_field(v, 'volfiles', list, self.volfiles)
        self.img = try_to_read_matlab_field(v, 'img', list, self.img)
        self.voxsize = try_to_read_matlab_field(v, 'voxsize', list, self.voxsize)
        self.dim = try_to_read_matlab_field(v, 'dim', list, self.dim)
        self.fname_conf = try_to_read_matlab_field(v, 'fname_conf', str, self.fname_conf)
        try:
            self.m_qform = v['m_qform'].tolist()
        except:
            pass
        try:
            self.m_toconform = v['m_toconform'].tolist()
        except:
            pass

    def nnav2mat(self):
        mat_vol = {}
        mat_vol['org'] = remove_None(self.org)
        mat_vol['fname'] = remove_None(self.fname)
        mat_vol['ftype'] = remove_None(self.ftype)
        mat_vol['manufacturer'] = remove_None(self.manufacturer)
        mat_vol['volfiles'] = remove_None(self.volfiles)
        mat_vol['img'] = remove_None(self.img)
        mat_vol['voxsize'] = remove_None(self.voxsize)
        mat_vol['dim'] = remove_None(self.dim)
        mat_vol['m_qform'] = remove_None(self.m_qform)
        mat_vol['m_toconform'] = remove_None(self.m_toconform)
        mat_vol['fname_conf'] = remove_None(self.fname_conf)
        return mat_vol


"""
    IMPORT FUNCTIONS
"""


def import_struct(fn):

    if os.path.isfile(fn):
        fn = os.path.abspath(fn)
    else:
        raise IOError("Could not find NNAV file: %s" % fn)

    if os.path.splitext(fn)[1] == '.mat':
        struct = read_matlab_nnav(fn)
    else:
        raise IOError('Deprecated file format: SimNIBS only accepts matlab ".mat" files')

    return struct


def read_matlab_nnav(fn):
    # read .mat file
    try:
        mat = scipy.io.loadmat(fn, struct_as_record=True, squeeze_me=False)
    except:
        raise IOError("File not found. If file exists, it was possibly saved with -v7.3")
    # check structure type
    try:
        structure_type = mat['type'][0]
    except:
        try:
            keys = [k for k in mat.keys() if not k.startswith('__')]
            if len(keys) > 1:
                raise IOError(
                    'Could not open .mat file. Nested structure?')
            structure_type = mat[keys[0]][0]['type'][0][0]
            mat = mat[keys[0]][0][0]
        except:
            raise IOError(
                "Could not access structure type in this .mat file")

    if structure_type == 'SESSION':
        path_to_file = os.path.split(os.path.realpath(fn))[0]
        structure = SESSION()
        structure.read_mat_struct(mat)
        for PL in structure.poslists:
            PL.fn_nnav = fn     # keep track of the NNAV filename inside each poslist
    else:
        raise IOError('Not a SESSION structure type!')

    return structure


"""
    EXPORT FUNCTIONS
"""

def save_matlab_nnav(struct, fn):
    if struct.type == 'SESSION':
        mat = struct.nnav2mat()
        scipy.io.savemat(fn, mat)

    else:
        raise IOError('This type of structure cannot be saved in matlab format')


def remove_None(src):
    if src is None:
        src = ''
    return src


def try_to_read_matlab_field(matlab_structure, field_name, field_type, alternative=None):
    """Function for flexibilly reading a field from the mesh file
    Tries to read the field with the specified name
    if sucesseful, returns the read
    if not, returns the alternative

    Parameters
    -------------------------------------
    matlab_struct: dict
        matlab structure as read by scipy.io, without squeeze
    field_name: str
        name of field in mat structure
    field_type: 'int', 'float', 'str',....
        function that transforms the field into the desired type
    alternative (optional): any
        if the field could not be red, return alternative
    """
    try:
        return field_type(matlab_structure[field_name][0])
    except (TypeError, KeyError, IndexError, ValueError):
        pass
    try:
        return field_type(matlab_structure[field_name][0][0])
    except (TypeError, KeyError, IndexError, ValueError):
        pass
    return alternative


def get_m2m_folder_from_mesh_name(mesh_name):
    subdir, subid = os.path.split(mesh_name)
    subid = os.path.splitext(subid)[0]
    subdir = os.path.normpath(os.path.expanduser(subdir))
    m2m_dir = os.path.join(subdir, 'm2m_' + subid)
    if not os.path.isdir(m2m_dir):
        logger.warning('Could not find m2m directory from mesh name')
        return None
    return m2m_dir


def get_subid_from_m2m_folder(m2m_folder):
    d = os.path.normpath(os.path.expanduser(m2m_folder))
    if not os.path.isdir(m2m_folder):
        logger.warning('The given m2m folder name does not correspond to a directory')
        return None

    m2m_folder_name = d.split(os.sep)[-1]
    try:
        subid = re.search('m2m_(.+)', m2m_folder_name).group(1)
    except:
        logger.warning('Could not find subject ID from path')
        return None
    return subid


def get_mesh_name_from_m2m_folder(m2m_folder):
    d = os.path.normpath(os.path.expanduser(m2m_folder))
    if not os.path.isdir(m2m_folder):
        logger.warning('The given m2m folder name does not correspond to a directory')
        return None

    m2m_folder_name = d.split(os.sep)[-1]
    try:
        subid = re.search('m2m_(.+)', m2m_folder_name).group(1)
    except:
        logger.warning('Could not find subject ID from path')
        return None

    mesh = os.path.normpath(os.path.join(m2m_folder, '..', subid + '.msh'))
    if not os.path.isfile(mesh):
        logger.info('Could not find mesh file in the standard '
                    'location: {0}'.format(mesh))
        return None

    return mesh


def get_dir_tensor_from_m2m_folder(m2m_folder):
    subid = get_subid_from_m2m_folder(m2m_folder)
    if subid is None:
        return None
    tensor_file = os.path.normpath(
        os.path.join(m2m_folder, '..', 'd2c_' + subid, 'dti_results_T1space',
                     'DTI_conf_tensor.nii.gz'))
    if not os.path.isfile(tensor_file):
        logger.info('Could not find the dti tensor file in the standard '
                    'location: {0}'.format(tensor_file))
        return None
    return tensor_file


def get_eeg_cap_from_m2m_folder(m2m_folder):
    eeg_cap = os.path.join(m2m_folder, 'eeg_positions', 'EEG10-10_UI_Jurak_2007.csv')
    if not os.path.isfile(eeg_cap):
        logger.warning('Could not find Electrode cap')
        return None
    else:
        return eeg_cap


def get_eeg_positions(fn_csv):
    type_, coordinates, _, name, _, _ = transformations._read_csv(fn_csv)
    eeg_pos = OrderedDict()
    for i, t in enumerate(type_):
        if t in ['Electrode', 'ReferenceElectrode', 'Fiducial']:
            eeg_pos[name[i]] = coordinates[i]
    return eeg_pos


def _substitute(pos, eeg_cap):
    if sys.version_info >= (3, 0):
        clss = str
    else:
        clss = basestring

    if isinstance(pos, clss):
        if eeg_cap:
            eeg_pos = get_eeg_positions(eeg_cap)
            try:
                pos = eeg_pos[pos]
            except:
                raise ValueError(
                    'Could not find position {0} in cap {1}'.format(
                        pos, eeg_cap))
        else:
            raise ValueError(
                'Tried to read position: {0} but eeg_cap is not set'.format(pos))

    return pos
