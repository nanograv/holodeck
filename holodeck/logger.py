"""Logging module.

This module produces a `logging.Logger` instance that can log both to stdout (i.e. using print) and
also to an output file.  This is especially useful for long or parallelized calculations where
more significant diagnostic outputs are required for debugging and/or record-keeping.

"""

from datetime import datetime
from pathlib import Path
import logging
from logging import DEBUG, INFO, WARNING, ERROR  # noqa import these for easier access internally
import sys

from holodeck import LOG_SUFFIX, LOG_FILENAME_WITH_TIME_STAMP


# LOG_SUFFIX = '.log'
# LOG_FILENAME_WITH_TIME_STAMP = False


class RankFilter(logging.Filter):

    def __init__(self, rank):
        super().__init__()
        self.rank = rank

    def filter(self, record):
        record.rank = self.rank
        return True


def get_logger(name=None, level_stream=WARNING, tostr=sys.stdout, tofile=None, level_file=DEBUG):
    """Create a standard logger object which logs to file and or stdout stream.

    Parameters
    ----------
    name : str,
        Handle for this logger, must be distinct for a distinct logger.
    level_stream : int,
        Logging level for stream.
    tostr : bool,
        Log to stdout stream.
    tofile : str or `None`,
        Filename to log to (turned off if `None`).
    level_file : int,
        Logging level for file.

    Returns
    -------
    logger : ``logging.Logger`` object,
        Logger object to use for logging.

    """

    comm_rank = None
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        if comm.size > 1:
            comm_rank = comm.rank
    except ModuleNotFoundError:
        pass

    if name is None:
        name = 'holodeck'
        if (comm_rank is not None):
            name += f"_rank{comm_rank}"

    if (tofile is None) and (not tostr):
        raise ValueError("Must log to something!")

    logger = logging.getLogger(name)
    # Make sure handlers don't get duplicated (ipython issue)
    while len(logger.handlers) > 0:
        logger.handlers.pop()
    # Prevents duplication or something something...
    logger.propagate = 0

    # store information about whether we're running in parallel or not.
    # If `comm_rank` is None, then this is a serial process.  If `comm_rank` is an integer,
    # then that is the rank of the current process.
    logger.comm_rank = comm_rank

    # Logger object must be at minimum level
    logger.setLevel(logging.DEBUG)

    # ---- Log To stdout
    if tostr not in [None, False]:
        format_date = '%H:%M:%S'

        rank_format = "|rank=%(rank)d" if (comm_rank is not None) else ""

        format_stream = f"%(asctime)s %(levelname)s : %(message)s [%(filename)s:%(funcName)s{rank_format}]"
        stream_formatter = logging.Formatter(format_stream, format_date)
        handler = logging.StreamHandler(tostr)
        handler.setFormatter(stream_formatter)
        handler.setLevel(level_stream)
        logger.addHandler(handler)

        if comm_rank is not None:
            rank_filter = RankFilter(comm_rank)
            logger.addFilter(rank_filter)

    # ---- Log to file
    if tofile not in [None, False]:
        log_to_file(logger, file_level=level_file, file_name=tofile)

    # store these values for convenience
    for lvl in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
        setattr(logger, lvl, getattr(logging, lvl))

    # ---- Make sure that the `setLevel` command reaches the stream logger

    # Construct a new function to replace 'setLevel'
    def _set_level(self, lvl):
        for handler in self.handlers:
            if not isinstance(handler, logging.StreamHandler):
                continue
            handler.setLevel(lvl)

        return

    # replace `setLevel` function on the logger
    logger.setLevel = _set_level.__get__(logger)
    logger.setLevel(level_stream)

    return logger


def log_to_file(logger, file_level=DEBUG, file_name=None, base_name='holodeck', path="./logs"):
    """Add a `FileHandler` to the given logger, to log to an output (text) file.

    If `file_name` IS given, then it is used as the output filename.
        If `file_name` is NOT an absolute path, then the file is placed in the `path` directory.
        If `file_name` IS an absolute path, then that's where the file is placed.
    If `file_name` is NOT given, then a filename is constructed based on the `base_name`, the logging level, and
    the processor rank (if this is a parallel job).

    """

    # ---- Construct filename as needed

    comm_rank = logger.comm_rank
    if file_name is None:
        file_name = f"{base_name}"

        if (comm_rank is not None) and (comm_rank > 0):
            file_name = f"_{file_name}_rank-{comm_rank}"

        if LOG_FILENAME_WITH_TIME_STAMP:
            beg = datetime.now()
            str_time = f"{beg.strftime('%Y%m%d-%H%M%S')}"
            # e.g.: "holodeck.librarian.gen_lib__20230918-140722"
            file_name = f"{file_name}_{str_time}"

        file_name += f"_lvl-{file_level}{LOG_SUFFIX}"
        file_name = Path(file_name)
        logger.debug(f"Constructed default file_name='{file_name}'")

    file_name = Path(file_name)
    if not file_name.is_absolute():
        file_name = Path(path).joinpath(file_name)
        file_name.parent.mkdir(exist_ok=True)

    # ---- Setup formatter

    rank_format = "|rank=%(rank)d" if (comm_rank is not None) else ""

    format_file = (
        "%(asctime)s "
        f"[%(filename)15.15s:%(funcName)-25.25s{rank_format}] "
        "%(levelname)8.8s: %(message)s"
    )
    format_date = '%Y/%m/%d %H:%M:%S'
    file_formatter = logging.Formatter(format_file, format_date)
    fhandler = logging.FileHandler(file_name, 'w')
    fhandler.setFormatter(file_formatter)
    fhandler.setLevel(file_level)
    logger.addHandler(fhandler)

    if comm_rank is not None:
        rank_filter = RankFilter(comm_rank)
        logger.addFilter(rank_filter)

    # Store output filename to `logger` object
    logger.filename = file_name
    logger.info(f"Logging to file: '{file_name}'")
    return

