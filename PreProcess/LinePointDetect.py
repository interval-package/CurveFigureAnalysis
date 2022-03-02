import cv2
from BrokenLine import *


def LinePointDetect(self):
    # 提取线的中点坐标，将中点坐标输出

    scrGray = cv2.cvtColor(self.binaryPic, cv2.COLOR_BGR2GRAY)
    scrGray = cv2.threshold(scrGray, 200, 255, cv2.THRESH_BINARY)[1]
    # 这里进行了一次旋转，因为np.where的遍历是沿着y行方向进行的
    idx_x, idx_y = np.where(cv2.rotate(scrGray, cv2.ROTATE_90_CLOCKWISE, 90))

    # 对x做一次差分，找出x有递增的点
    dx = np.diff(idx_x)
    # 获取递增点序号，并且在初始处插入一个0
    x_stage = np.insert(np.where(dx), 0, 0)

    # ，现在1表示一串相同x的初始点
    x_gap = np.diff(x_stage)+1
    x_gap = np.floor_divide(x_gap, 2)
    x_gap = np.append(x_gap, 0)
    x_ = x_gap + x_stage

    idx_x_sep = idx_x[x_]
    idx_y_sep = idx_y[x_]

    return idx_x, idx_y, idx_x_sep, idx_y_sep


def GetBrokenPoint():
    """
        转折点查找,使用二阶偏导查找
    """
    pass


def main():
    obj = BrokenLine(568)
    x, y, x_s, y_s = LinePointDetect(obj)
    plt.subplot(1, 2, 1)
    plt.plot(x, y, 'b.')
    plt.plot(x_s, y_s, 'r-')
    plt.subplot(1, 2, 2)
    plt.plot(x_s, y_s, 'r.-')
    plt.show()
    pass


if __name__ == '__main__':
    main()
