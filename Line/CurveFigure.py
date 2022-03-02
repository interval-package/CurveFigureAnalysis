from LineFigure import *
import scipy.integrate as spyip


class CurveFigure(LineFigure):
    @staticmethod
    def readCurve(id: int):
        try:
            rawpic = cv2.imread("../../data/img_train_Curve/%d/draw.png" % id)
            maskpic = cv2.imread("../../data/img_train_Curve/%d/draw_mask.png" % id)
            piclable = pd.read_table("../../data/img_train_Curve/%d/db.txt" % id,
                                     engine='python',
                                     delimiter="\n")
            return rawpic, maskpic, piclable
        except Exception as result:
            print('repr(e):\t', repr(result))
            return np.array([]), np.array([]), []

    def __init__(self, id: int, test=False):
        # read in the pic into the obj
        super().__init__("../../data/img_train_Curve/%d" % id, test)
        self.type = "BrokenLine"

    def SmoothCurve(self):
        x, y = self.LinePointDetectCentralize()[2:3]
        xNew = np.linspace(np.min(x), np.max(x), 100)
        # yNew = spyip.spline()
        # y3 = spline(x, y, new_x, order=3, kind='smoothest')
        pass


def main():

    pass

if __name__ == '__main__':
    main()