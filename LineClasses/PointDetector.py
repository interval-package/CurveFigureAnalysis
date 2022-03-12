import cv2
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt


def LinePointDetectCentralize(scrGray):
    """
    @:param scrGray: bin pic
    :return: x, y for the line
    """
    # 由像素检测线上的点
    # 提取线的中点坐标，将中点坐标输出
    if scrGray.ndim > 2:
        # if not gray, change into gray
        scrGray = cv2.cvtColor(scrGray, cv2.COLOR_BGR2GRAY)
    # 再次确认为二进制图片
    scrGray = cv2.threshold(scrGray, 200, 255, cv2.THRESH_BINARY)[1]

    # 这里进行了一次旋转，因为np.where的遍历是沿着行方向进行的
    idx_x, idx_y = np.where(cv2.rotate(scrGray, cv2.ROTATE_90_CLOCKWISE, 90))

    # 对x做一次差分，找出x有递增的点
    dx = np.diff(idx_x)
    # 获取递增点序号，并且在初始处插入一个0
    x_stage = np.insert(np.where(dx), 0, 0)

    # 现在1表示一串相同x的初始点
    x_gap = np.diff(x_stage) + 1
    x_gap = np.floor_divide(x_gap, 2)
    x_gap = np.append(x_gap, 0)
    x_ = x_gap + x_stage

    try:
        idx_x_sep = idx_x[x_]
        idx_y_sep = idx_y[x_]
    except IndexError:
        idx_x_sep = idx_x
        idx_y_sep = idx_y

    return idx_x_sep, idx_y_sep, idx_x, idx_y


def LinePointDetectCentralizeAmplified(pic):
    """
    :param pic: bin pic
    :return:
    using peaks to better fit the output
    对于原来的中间点算法，在强拐点的时候，很容易产生粘合，导致拐点处的中间值过低，这里进行修正
    """
    # 对于原来的中间点算法，在强拐点的时候，很容易产生粘合，导致拐点处的中间值过低，这里进行修正
    if pic.ndims > 2:
        raise ValueError("the input should be a bin Pic")
    x, y, x_all, y_all = LinePointDetectCentralize(pic)
    # 我是没有想到的，scipy的信号处理会有这么大的用处
    # 使用find_peaks找到对应的峰的序号，取反找到低谷
    y_idx_h = find_peaks(y)
    y_idx_l = find_peaks(-y)
    # 将峰值所对应的x提取出来，将其周围一圈替换为原来的最高值或者最低值

    return


def LineLowFreqFilter(y):
    pack = butter(8, 0.02, btype='low', output='ba')
    Data = filtfilt(pack[0], pack[1], y)  # data为要过滤的信号
    return Data


def LinePointsPlot(src, points, color=None, PotType='line'):
    if color is None:
        color = [125, 125, 125]
    if isinstance(points, tuple):
        x, y = points
    elif isinstance(points, np.ndarray):
        x = points[:, 0]
        y = points[:, 1]
    else:
        raise ValueError("unfitted input")
    # 注意，这里图像要flip上下颠倒两次，才会画出正常的图像
    src = cv2.flip(src, 0)
    if PotType == 'line':
        for i in range(0, len(x) - 1):
            cv2.line(src, (x[i], y[i]), (x[i + 1], y[i + 1]), color=color, thickness=5)
    elif PotType == 'circle':
        for i in range(len(x)):
            cv2.circle(src, (x[i], y[i]), 5, color=color)
    elif PotType == 'dot':
        for i in range(len(x)):
            cv2.drawMarker(src, (x[i], y[i]), color=color, markerType=cv2.MARKER_CROSS, markerSize=2)
    else:
        raise ValueError("invalid PotType")
    # 返回时再次颠倒
    return cv2.flip(src, 0)


def DetectPointHarrisMethod(src: np.ndarray, gap=0.01, mask=None, inplace=False):
    src = src.copy()
    if src.ndim > 2:
        img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        img = src.copy()

    # imgBlur = cv2.GaussianBlur(img, (5, 5), 1)  # ADD GAUSSIAN BLUR
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    # img = cv2.erode(imgCanny, kernel, iterations=2)
    # img = cv2.dilate(img, kernel, iterations=2)
    # thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11,
    #                                2)  # 自适应阈值化

    # 设置蒙版
    thresh = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)

    dst = cv2.cornerHarris(src=thresh, blockSize=2, ksize=3, k=0.04)
    # 通过阈值来得到Harris结果的有效点
    ret, dst = cv2.threshold(dst, dst.max() * gap, 255, 0)
    dst = np.uint8(dst)
    y, x = np.where(dst > 0)
    y = y[np.argsort(x)]
    x = x[np.argsort(x)]
    if inplace:
        # src[x, y] = [0, 0, 255]
        for i in range(0, len(x) - 1):
            cv2.line(src, (x[i], y[i]), (x[i + 1], y[i + 1]), color=[0, 0, 255])
    return x, y, src, thresh


def DetectPointGoodFeatureMethod(src: np.ndarray, gap=0.01, maxCount=100, mask=None, inplace=False):
    img = src.copy()
    if src.ndim > 2:
        img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    if mask is not None:
        img = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)
    corners = cv2.goodFeaturesToTrack(img, maxCount, gap, 100)
    corners = np.int0(corners).reshape(corners.shape[0], corners.shape[2])
    corners = corners[np.argsort(corners[:, 0]), :]
    if inplace:
        for i in range(0, corners.shape[0] - 1):
            print((corners[i][1], corners[i][0]), (corners[i + 1][1], corners[i + 1][0]))
            cv2.line(src, (corners[i][0], corners[i][1]), (corners[i + 1][0], corners[i + 1][1]), color=[0, 255, 0])
    return corners


class PointDetector(object):
    def __init__(self, x, y, x_all=None, y_all=None):
        self.x, self.y = x, y
        if x_all is None or y_all is None:
            self.x_all = x
            self.y_all = y
        else:
            self.x_all = x_all
            self.y_all = y_all
        pass

    @classmethod
    def FromBinPic(cls, pic, detectFun=LinePointDetectCentralize):
        if pic.ndims > 2:
            raise ValueError("the input should be a bin Pic")
        obj = detectFun(pic)
        length = len(obj)
        if length == 2:
            return cls(obj[0], obj[1])
        elif length == 4:
            return cls(obj[0], obj[1], obj[2], obj[3])
        else:
            raise ValueError("invalid func with unfitted outputs")

    def PointsTrans(self, x=0, y=0, scale_x=None, scale_y=None):
        """
        parameters
        ==============
        :param x: move the points on the x-axis
        :param y: move the points on the y-axis
        :param scale_x:
        :param scale_y:
        :returns
        ==============
        :return: this function will change the points within, and return the changed
        """
        return

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item: int):
        return self.x[item], self.y[item]


if __name__ == '__main__':
    pass
