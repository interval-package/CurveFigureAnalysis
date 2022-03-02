from CurveFigure import *
from BrokenLineFigure import *
from CurveFigure import *


def main():
    id = 13
    for id in range(1000):
        obj = CurveFigure(id, True)
        obj.show()

    pass


if __name__ == '__main__':
    main()
