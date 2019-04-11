import tempfile
import os
import pytest
import copy
import numpy as np
import simnibs.simulation.fem as fem
import simnibs.msh.gmsh_numpy as gmsh
import simnibs.simulation.analytical_solutions.sphere as analytical_solutions


@pytest.fixture
def sphere3_msh():
    fn = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '..', 'testing_files', 'sphere3.msh')
    return gmsh.read_msh(fn)

@pytest.fixture
def sphere_el_msh():
    fn = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '..', 'testing_files', 'sphere_w_electrodes.msh')
    return gmsh.read_msh(fn)


@pytest.fixture
def cube_msh():
    fn = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '..', 'testing_files', 'cube_w_electrodes.msh')
    return gmsh.read_msh(fn)


@pytest.fixture
def cube_lr():
    fn = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '..', 'testing_files', 'cube.msh')
    return gmsh.read_msh(fn)


def rdm(a, b):
    return np.linalg.norm(a / np.linalg.norm(a) -
                          b / np.linalg.norm(b))


def mag(a, b):
    return np.abs(np.log(np.linalg.norm(a) / np.linalg.norm(b)))

class Testcalc_fields():
    def test_calc_vEJgs(self, sphere3_msh):
        phi = sphere3_msh.nodes.node_coord[:, 0] + \
            2 * sphere3_msh.nodes.node_coord[:, 1] + \
            -3 * sphere3_msh.nodes.node_coord[:, 2]
        potential = gmsh.NodeData(phi, mesh=sphere3_msh)

        E = np.zeros((sphere3_msh.elm.nr, 3))
        E[:] = [-1., -2., 3.]
        E = gmsh.ElementData(E, mesh=sphere3_msh)

        cond = sphere3_msh.elm.tag1
        cond = gmsh.ElementData(cond, mesh=sphere3_msh)

        m = fem.calc_fields(potential, 'vJEgsej', cond)
        assert np.allclose(m.field['v'].value, potential.value)
        assert np.allclose(m.field['E'].value, E.value * 1e3)
        assert np.allclose(m.field['J'].value, cond.value[:, None] * E.value * 1e3)
        assert np.allclose(m.field['g'].value, -E.value * 1e3)
        assert np.allclose(m.field['conductivity'].value, cond.value)
        assert np.allclose(m.field['normE'].value, np.linalg.norm(E.value, axis=1) * 1e3)
        assert np.allclose(m.field['normJ'].value,
                           np.linalg.norm(cond.value[:, None] * E.value, axis=1) * 1e3)

    def test_calc_dadt(self, sphere3_msh):
        phi = sphere3_msh.nodes.node_coord[:, 0] + \
            2 * sphere3_msh.nodes.node_coord[:, 1] + \
            -3 * sphere3_msh.nodes.node_coord[:, 2]
        potential = gmsh.NodeData(phi, mesh=sphere3_msh)

        dadt = .2 * sphere3_msh.nodes.node_coord
        dadt = gmsh.NodeData(dadt, mesh=sphere3_msh)

        E = np.zeros((sphere3_msh.elm.nr, 3))
        E = [-1, -2, 3] - dadt.node_data2elm_data().value
        E = gmsh.ElementData(E, mesh=sphere3_msh)
        E.assign_triangle_values()

        cond = sphere3_msh.elm.tag1
        cond = gmsh.ElementData(cond, mesh=sphere3_msh)

        m = fem.calc_fields(potential, 'vDJEgsej', cond, dadt=dadt,
                            units='m')
        assert np.allclose(m.field['v'].value, potential.value)
        assert np.allclose(m.field['D'].value, dadt.value)
        assert np.allclose(m.field['g'].value, [1, 2, -3])
        assert np.allclose(m.field['E'].value, E.value)
        assert np.allclose(m.field['J'].value, cond.value[:, None] * E.value)
        assert np.allclose(m.field['conductivity'].value, cond.value)
        assert np.allclose(m.field['normE'].value, np.linalg.norm(E.value, axis=1))
        assert np.allclose(m.field['normJ'].value,
                           np.linalg.norm(cond.value[:, None] * E.value, axis=1))

    def test_calc_tensor(self, sphere3_msh):
        phi = sphere3_msh.nodes.node_coord[:, 0] + \
            2 * sphere3_msh.nodes.node_coord[:, 1] + \
            -3 * sphere3_msh.nodes.node_coord[:, 2]
        potential = gmsh.NodeData(phi, mesh=sphere3_msh)

        o = np.ones(sphere3_msh.elm.nr)
        z = np.zeros(sphere3_msh.elm.nr)
        cond = np.vstack([z, o, z, o, z, z, z, z, o]).T
        cond = gmsh.ElementData(cond, mesh=sphere3_msh)
        m = fem.calc_fields(potential, 'vJEgsej', cond, units='m')

        assert np.allclose(m.field['v'].value, potential.value)
        assert np.allclose(m.field['g'].value, [1, 2, -3])
        assert np.allclose(m.field['E'].value, [-1, -2, 3])
        assert np.allclose(m.field['J'].value, [-2, -1, 3])
        assert np.allclose(m.field['normE'].value, np.sqrt(4+1+9))
        assert np.allclose(m.field['normJ'].value, np.sqrt(4+1+9))

class TestTMSGetDP:
    ''' Those are more of an integration test then an unit test '''
    def test_tms_getdp(self, sphere3_msh):
        dipole_pos = np.array([0., 0., 200])
        dipole_moment = np.array([1., 0., 0.])

        didt = 1e6
        r = (sphere3_msh.nodes.node_coord - dipole_pos) * 1e-3
        dAdt = 1e-7 * didt * np.cross(dipole_moment, r) / (np.linalg.norm(r, axis=1)[:, None] ** 3)
        dAdt = gmsh.NodeData(dAdt, mesh=sphere3_msh)

        cond = gmsh.ElementData(np.ones(sphere3_msh.elm.nr), mesh=sphere3_msh)

        phi_sim = fem.tms_getdp(sphere3_msh, cond, dAdt)
        m = fem.calc_fields(phi_sim, 'vE', dadt=dAdt)
        m = m.crop_mesh(elm_type=4)
        E_fem = m.field['E'].value

        pos = m.elements_baricenters().value
        E_analytical = analytical_solutions.tms_E_field(dipole_pos * 1e-3,
                                                        dipole_moment, didt,
                                                        pos * 1e-3)
        m.elmdata.append(gmsh.ElementData(E_analytical))
        #gmsh.write_msh(m, '~/Tests/tms.msh')
        rdm = np.linalg.norm(
                E_fem.reshape(-1)/np.linalg.norm(E_fem) -
                E_analytical.reshape(-1)/np.linalg.norm(E_analytical))
        mag = np.log(np.linalg.norm(E_fem)/np.linalg.norm(E_analytical))
        assert rdm < .2
        assert np.abs(mag) < np.log(1.1)
 
    def test_tms_getdp_aniso(self, sphere3_msh):
        dipole_pos = np.array([0., 0., 200])
        dipole_moment = np.array([0., 1., 0.])

        didt = 1e6
        r = (sphere3_msh.nodes.node_coord - dipole_pos) * 1e-3
        dAdt = 1e-7 * didt * np.cross(dipole_moment, r) / (np.linalg.norm(r, axis=1)[:, None] ** 3)
        dAdt = gmsh.NodeData(dAdt, mesh=sphere3_msh)

        cond = gmsh.ElementData(np.ones((sphere3_msh.elm.nr, 9)), mesh=sphere3_msh)
        cond.value[:] = [1, 0, 0, 0, 1, 0, 0, 0, 1]

        phi_sim = fem.tms_getdp(sphere3_msh, cond, dAdt)
        m = fem.calc_fields(phi_sim, 'vE', dadt=dAdt)
        m = m.crop_mesh(elm_type=4)
        E_fem = m.field['E'].value

        pos = m.elements_baricenters().value
        E_analytical = analytical_solutions.tms_E_field(dipole_pos * 1e-3,
                                                        dipole_moment, didt,
                                                        pos * 1e-3)
        rdm = np.linalg.norm(
                E_fem.reshape(-1)/np.linalg.norm(E_fem) -
                E_analytical.reshape(-1)/np.linalg.norm(E_analytical))
        mag = np.log(np.linalg.norm(E_fem)/np.linalg.norm(E_analytical))
        assert rdm < .2
        assert np.abs(mag) < np.log(1.1)

    def test_tms_getdp_write_first(self, sphere3_msh):
        dipole_pos = np.array([0., 0., 200])
        dipole_moment = np.array([1., 0., 0.])
        didt = 1e6
        r = (sphere3_msh.nodes.node_coord - dipole_pos) * 1e-3
        dAdt = 1e-7 * didt * np.cross(dipole_moment, r) / (np.linalg.norm(r, axis=1)[:, None] ** 3)
        dAdt = gmsh.NodeData(dAdt, mesh=sphere3_msh)
        fn_dadt = tempfile.NamedTemporaryFile(suffix='.msh', delete=False).name
        sphere3_msh.nodedata = [dAdt]
        gmsh.write_msh(sphere3_msh, fn_dadt)
        sphere3_msh.nodedata = []

        cond = gmsh.ElementData(np.ones(sphere3_msh.elm.nr), mesh=sphere3_msh)
        fn_cond = tempfile.NamedTemporaryFile(suffix='.msh', delete=False).name
        sphere3_msh.elmdata = [cond]
        gmsh.write_msh(sphere3_msh, fn_cond)
        sphere3_msh.elmdata = []

        fn_msh = tempfile.NamedTemporaryFile(suffix='.msh', delete=False).name
        gmsh.write_msh(sphere3_msh, fn_msh)

        fn_pro = tempfile.NamedTemporaryFile(suffix='.pro', delete=False).name
        fem.tms_getdp(fn_msh, fn_cond, fn_dadt, fn_pro)
        assert os.path.isfile(fn_dadt)
        assert os.path.isfile(fn_cond)
        assert os.path.isfile(fn_msh)
        assert os.path.isfile(fn_pro)
        os.remove(fn_dadt)
        os.remove(fn_cond)
        os.remove(fn_msh)
        os.remove(fn_pro)

class TestTDCSGetDP:
    def test_calc_flux_electrodes(self, cube_msh):
        v = gmsh.NodeData(cube_msh.nodes.node_coord[:, 1], mesh=cube_msh)
        cond = gmsh.ElementData(np.ones(cube_msh.elm.nr), mesh=cube_msh)
        areas = cube_msh.elements_volumes_and_areas().value
        area_el1 = np.sum(areas[cube_msh.elm.tag1 == 1100]) * 1e-3
        flux = fem._calc_flux_electrodes(v, cond, 500, 1005)
        assert np.isclose(np.abs(flux), area_el1)

    def test_tdcs_getdp_cube(self, cube_msh):
        surface_tags = [1100, 1101]
        currents = [-1, 1]
        cond = gmsh.ElementData(np.ones(cube_msh.elm.nr), mesh=cube_msh)
        cond.value[cube_msh.elm.tag1 != 5] = 1e6
        v = fem.tdcs_getdp(cube_msh, cond, currents, surface_tags)
        m = fem.calc_fields(v, 'vE')
        # Take a section in the middle of the cube
        bar = m.elements_baricenters()
        elements_in_mid = m.elm.elm_number[np.abs(bar.value[:, 1]) < 20]
        m = m.crop_mesh(elements=elements_in_mid)
        E_fem = m.field['E'].value
        #gmsh.write_msh(m, '~/Tests/tdcs.msh')
        E_analytical = np.zeros_like(E_fem)
        E_analytical[:, 1] = 100
        rdm = np.linalg.norm(
                E_fem.reshape(-1)/np.linalg.norm(E_fem) -
                E_analytical.reshape(-1)/np.linalg.norm(E_analytical))
        mag = np.log(np.linalg.norm(E_fem)/np.linalg.norm(E_analytical))
        assert rdm < .2
        assert np.abs(mag) < np.log(1.1)

    def test_tdcs_getdp_sphere(self, sphere_el_msh):
        surface_tags = [1100, 1101]
        currents = [-1, 1]
        cond = gmsh.ElementData(np.ones(sphere_el_msh.elm.nr), mesh=cube_msh)
        cond.value[sphere_el_msh.elm.tag1 == 4] = .01
        v = fem.tdcs_getdp(sphere_el_msh, cond, currents, surface_tags)
        m = copy.deepcopy(sphere_el_msh)
        m.nodedata = [v]
        m = m.crop_mesh(3)
        v_fem = m.nodedata[0].value - m.nodedata[0].value[0]
        v_analytical = analytical_solutions.potential_3layers_surface_electrodes(
            [85, 90, 95], [1., .01, 1.], [0, 0, 95], [0, 0, -95], m.nodes.node_coord)
        v_analytical -= v_analytical[0]
        rdm = np.linalg.norm(
                v_fem/np.linalg.norm(v_fem) -
                v_analytical/np.linalg.norm(v_analytical))
        #m.nodedata.append(gmsh.NodeData(v_analytical))
        #gmsh.write_msh(m, '~/Tests/tdcs.msh')
        assert rdm < .2

    '''
    def test_tdcs_neumann_getdp(self, cube_msh):
        surface_tags = [1100, 1101]
        currents = [-1, 1]
        cond = gmsh.ElementData(np.ones(cube_msh.elm.nr), mesh=cube_msh)
        cond.value[cube_msh.elm.tag1 != 5] = 1e3
        v = fem.tdcs_neumann_getdp(cube_msh, cond, currents, surface_tags)
        m = fem.calc_fields(v, 'vE')
        # Take a section in the middle of the cube
        bar = m.elements_baricenters()
        elements_in_mid = m.elm.elm_number[np.abs(bar.value[:, 1]) < 20]
        m = m.crop_mesh(elements=elements_in_mid)
        E_fem = m.field['E'].value
        #gmsh.write_msh(m, '~/Tests/tdcs.msh')
        E_analytical = np.zeros_like(E_fem)
        E_analytical[:, 1] = 100
        rdm = np.linalg.norm(
                E_fem.reshape(-1)/np.linalg.norm(E_fem) -
                E_analytical.reshape(-1)/np.linalg.norm(E_analytical))
        mag = np.log(np.linalg.norm(E_fem)/np.linalg.norm(E_analytical))
        assert rdm < .2
        assert np.abs(mag) < np.log(1.1)
    '''
