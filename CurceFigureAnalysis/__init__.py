"""
py package for curve figure analysis

"""

import logging

USING_HED = False

SETUP_CONFIG = True

if SETUP_CONFIG:
    logging.basicConfig(format='%(asctime)s [%(levelname)s] : %(message)s', level=logging.DEBUG)
    logging.debug("[cfa] set up basic logging...")
    logging.debug("[cfa] initiating CurveFigureAnalysis")


