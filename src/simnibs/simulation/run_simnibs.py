import numpy as np
import os
import time
from . import nnav
import logging


def run_simulation(simnibs_struct, cpus=1):
    """Runs a simnnibs problem
    Parameters:
    --------------
    simnibs_struct: sim_struct.SESSION of str
        SESSION of name of '.mat' file defining the simulation
    cpus: int
        Number of processes to run in parallel
    """
    np.set_printoptions(precision=4)

    if isinstance(simnibs_struct, nnav.SESSION):
        p = simnibs_struct
    else:
        p = nnav.import_struct(simnibs_struct)

    # Set-up logger
    time_str = time.strftime("%Y%m%d-%H%M%S")
    # Create the path before setting-up the logger
    if not os.path.isdir(p.pathfem):
        os.mkdir(p.pathfem)
    log_fn = os.path.join(p.pathfem, 'simnibs_simulation_{0}.log'.format(time_str))
    fh = logging.FileHandler(log_fn)
    formatter = logging.Formatter('[ %(name)s - %(asctime)s ]%(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logger = logging.getLogger("simnibs")
    logger.addHandler(fh)

    save_fn = os.path.join(p.pathfem, 'simnibs_simulation_{0}.mat'.format(time_str))
    p.run_simulatons(cpus=cpus, save_fn=save_fn)
    logging.shutdown()
