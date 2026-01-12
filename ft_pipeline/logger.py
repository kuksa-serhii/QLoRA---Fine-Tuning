# ft_pipeline/logger.py
import logging
import sys
from typing import Optional
 
_LOGGER_NAME = "ft_pipeline"
 
def setup_logger(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
):
    """
    Call ONCE at notebook start.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
 
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
 
    # avoid duplicate handlers in notebooks
    if not logger.handlers:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
 
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
 
    return logger
 
 
def get_logger():
    return logging.getLogger(_LOGGER_NAME)
 
 