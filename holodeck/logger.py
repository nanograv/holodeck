"""
"""

# import inspect
import logging
import sys
from logging import DEBUG, INFO, WARNING, ERROR  # noqa


def get_logger(name='holodeck', level_stream=logging.WARNING, tostr=sys.stdout, tofile=None, level_file=logging.DEBUG):
    """Create a standard logger object which logs to file and or stdout stream.

    If logging to output stream (stdout) is enabled, an `_Indent_Formatter` object is used.

    Arguments
    ---------
    name : str,
        Handle for this logger, must be distinct for a distinct logger.
    tostr : bool,
        Log to stdout stream.
    tofile : str or `None`,
        Filename to log to (turned off if `None`).
    level_stream : int,
        Logging level for stream.
    level_file : int,
        Logging level for file.

    Returns
    -------
    logger : ``logging.Logger`` object,
        Logger object to use for logging.

    """
    if (tofile is None) and (not tostr):
        raise ValueError("Must log to something!")

    logger = logging.getLogger(name)
    # Make sure handlers don't get duplicated (ipython issue)
    while len(logger.handlers) > 0:
        logger.handlers.pop()
    # Prevents duplication or something something...
    logger.propagate = 0

    # Logger object must be at minimum level
    # logger.setLevel(int(np.min([level_file, level_stream])))
    logger.setLevel(logging.DEBUG)

    # Log to file
    # -----------
    if tofile not in [None, False]:
        format_date = '%Y/%m/%d %H:%M:%S'
        format_file = "%(asctime)s %(levelname)8.8s [%(filename)20.20s:%(funcName)-20.20s]%(message)s"
        file_formatter = logging.Formatter(format_file, format_date)
        fhandler = logging.FileHandler(tofile, 'w')
        fhandler.setFormatter(file_formatter)
        fhandler.setLevel(level_file)
        logger.addHandler(fhandler)
        # Store output filename to `logger` object
        logger.filename = tofile

    # ---- log To stdout
    if tostr not in [None, False]:
        format_date = '%H:%M:%S'

        format_stream = "%(asctime)s %(levelname)s : %(message)s [%(filename)s:%(funcName)s]"
        stream_formatter = logging.Formatter(format_stream, format_date)
        handler = logging.StreamHandler(tostr)
        handler.setFormatter(stream_formatter)
        handler.setLevel(level_stream)
        logger.addHandler(handler)

    for lvl in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
        setattr(logger, lvl, getattr(logging, lvl))

    return logger
