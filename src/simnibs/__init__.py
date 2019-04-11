from .utils.version import __version__ as version
import os
SIMNIBSDIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)),
                 '..', '..', '..'))
from .msh import gmsh_numpy as gmsh
from .simulation import nnav as sim_struct
from .simulation import cond
from .simulation.run_simnibs import run_simulation
