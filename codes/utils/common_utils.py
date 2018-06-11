""" Contains common utility functions
"""
import logging


def setup_logging(path, level=0):

    level = logging.INFO
    if level > 0:
        level = logging.DEBUG

    fmt = '%(name)-12s: %(levelname)-8s %(message)s'
    logging.basicConfig(format=fmt, datefmt='%Y-%m-%d %I:%M:%S %p', filename=path, filemode='w', level=level)

    console = logging.StreamHandler()
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    console.setLevel(level)

    logger = logging.getLogger('aes-lac-2018')
    logger.addHandler(console)
    logger.setLevel(level)