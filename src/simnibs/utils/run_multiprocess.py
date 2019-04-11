import os
import multiprocessing
import time
import logging

from .simnibs_logger import PicklebleFileHandler


def run_multiprocess(target, args_list, cpus=None, use_slurm=False):
    '''' Run several instances of the function "target" in paralell
    Parameters:
    ---------------
        target: function
            function to be called
        args_list: list of tuples
            list of the argument tupples
        cpus: int or None
            number of processes to be started simultaneously. If None, will start all
            process simultaneously (Default: None)
        use_slurm: True or False
            Wether or not to use SLURM for clusters (Default: False)
    '''

    # remove locks from all loggers
    logger = logging.getLogger('simnibs')
    all_locks = []
    # if using slurm, remove all locks from all loggers
    if use_slurm:
        for k, v in logging.Logger.manager.loggerDict.iteritems():
            try:
                for (j, h) in enumerate(v.handlers):
                    all_locks.append(logging.getLogger(k).handlers[j].lock)
                    logging.getLogger(k).handlers[j].lock = None
            except AttributeError:
                continue
    loggers = []
    for i, h in enumerate(logger.handlers):
        loggers.append([i, h])
    processes = []
    if not use_slurm:
        for args in args_list:
            processes.append(
                multiprocessing.Process(target=target,
                                        args=args))
    else:
        from . import slurm
        for args in args_list:
            processes.append(
                slurm.SlurmProcess(target=target,
                                   args=args))

    if cpus is None:
        cpus = 1e3
    next_to_start = 0
    while next_to_start < len(processes):
        # change name of file loggers
        for i, h in loggers:
            if type(h) == logging.FileHandler:
                new_handle = PicklebleFileHandler(h.baseFilename + str(next_to_start))
                new_handle.lock = None
                logger.handlers[i] = new_handle
        print next_to_start
        if use_slurm:
            time.sleep(3)
        processes[next_to_start].start()
        next_to_start += 1
        if next_to_start < cpus:
            continue
        while True:
            n_completed = sum(p.exitcode is not None for p in processes)
            n_active = sum(p.is_alive() for p in processes) - n_completed
            if n_active < cpus:
                break

    [p.join() for p in processes]
    # join file loggers again
    for i, h in loggers:
        if type(h) == logging.FileHandler:
            with open(h.baseFilename, 'a') as log:
                for pro in range(len(processes)):
                    with open(h.baseFilename + str(pro), 'r') as tmp_log:
                        log.write(tmp_log.read())
                    os.remove(h.baseFilename + str(pro))
        logger.handlers[i] = h

    # put the keys back in the logger
    i = 0
    if use_slurm:
        for k, v in logging.Logger.manager.loggerDict.iteritems():
            try:
                for (j, h) in enumerate(v.handlers):
                    logging.getLogger(k).handlers[j].lock = all_locks[i]
                    i += 1
            except AttributeError:
                continue

    return [p.exitcode for p in processes]
