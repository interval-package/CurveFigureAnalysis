import matplotlib.pyplot as plt

from utils.dispUtils import *

from LineClasses.LineFigure import LineFigure
from LineClasses.BrokenLineFigure import BrokenLineFigure
from LineClasses.CurveFigure import CurveFigure

# init actions

UsingHed = False

# set basic Set Path

BrokenLineFigure.DataSetPath_BrokenLine_test = "../data/img_test_BrokenLine"

BrokenLineFigure.DataSetPath_BrokenLine_train = "../data/img_train_BrokenLine"

CurveFigure.DataSetPath_Curve_train = "../data/img_train_Curve"

CurveFigure.DataSetPath_Curve_test = "../data/img_test_Curve"
