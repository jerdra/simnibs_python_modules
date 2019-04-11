import logging
import sys
import warnings
import traceback

global logger
logger = logging.getLogger('simnibs')
sh = logging.StreamHandler()
formatter = logging.Formatter('[ %(name)s ]%(levelname)s: %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)
logger.setLevel(logging.INFO)


def log_warnings(message, category, filename, lineno, file=None, line=None):
    logger.warn(warnings.formatwarning(message, category, filename, lineno))


warnings.showwarning = log_warnings

# Redirect the exceptions to the logger
def register_handler(orig_excepthook=sys.excepthook):
    def log_excep(type, value, tback):
        """Log uncaught exceptions. When an exception occurs, sys.exc_info()
        returns a tuple of three variables (exception class, exception value,
        traceback). Setting
            sys.excepthook = log_excep
        will replace the standard way of handling exceptions but that of log_excep.
        log_excep takes the sys.exc_info() as input and prints the exception to 
        "logger" at level error.
        """
        #logger.debug("Traceback:", exc_info=(type, value, tback))
        logger.critical("Unhandled exception:", exc_info=(type, value, tback))
        #orig_excepthook(*exc_info)
    sys.excepthook = log_excep

register_handler()

def strcolour(string, colour, bold=False):
    code = ''
    if bold:
        code = '\033[1m'
    if colour in ['grey', 'r']:
        code += '\033[90m'
    elif colour in ['red', 'r']:
        code += '\033[91m'
    elif colour in ['green', 'g']:
        code += '\033[92m'
    elif colour in ['yellow', 'y']:
        code += '\033[93m'
    elif colour in ['blue', 'b']:
        code += '\033[94m'
    elif colour in ['pink', 'r']:
        code += '\033[95m'
    elif colour in ['cyan', 'r']:
        code += '\033[96m'
    else:
        return string
    return code + string + '\033[0m'


class PicklebleFileHandler(logging.FileHandler):
    def __getstate__(self):
        d = dict(self.__dict__)
        d['stream'] = None
        return d

if __name__ == '__main__':
    warnings.warn('aaaa')
