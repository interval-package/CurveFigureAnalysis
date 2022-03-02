import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

"""as cite
    read in the Line,and preprocessing
    do the blur and extraction
"""


class BrokenLine:
    @staticmethod
    def readBrokenLine(id: int):
        try:
            rawpic = cv2.imread("../../../data/img_train_BrokenLine/%d/draw.png" % id)
            maskpic = cv2.imread("../../data/img_train_BrokenLine/%d/draw_mask.png" % id)
            piclable = pd.read_table("../../data/img_train_BrokenLine/%d/db.txt" % id,
                                     engine='python',
                                     delimiter="\n")
            return rawpic, maskpic, piclable
        except Exception as result:
            return np.array([]), np.array([]), []

    def __init__(self, id: int):
        # read in the pic into the obj
        self.type = "BrokenLine"
        self.rawPic, self.maskPic, self.picLable = self.readBrokenLine(id)
        self.mask = np.zeros(np.shape(self.rawPic))

    def __int__(self, path: str):
        pass

    def GetMask(self):
        rows, cols = self.rawPic.shape[:2]
        maskArea = np.zeros([rows, cols], dtype=np.uint8)
        maskArea[int(cols / 4):int(cols / 2), int(rows / 4):int(rows / 2)] = 255
        self.mask = maskArea
        return maskArea

    def GetMaskPic(self):
        # 将图片进行Mask处理
        image = cv2.add(self.rawPic, np.zeros(np.shape(self.rawPic), dtype=np.uint8), mask=self.GetMask())
        return image

    def GetBoundry(self):
        # 转换为灰度图像
        src = cv2.cvtColor(self.rawPic, cv2.COLOR_BGR2GRAY)
        # 这里要将histRange设置为[0,256]，才会将空白计算进去，这是一个右开区间，同histSize=[256]
        # 同时，只对图像的主部分进行取样，不对边界进行色域提取，掩膜使用原来定义的掩膜
        hist = cv2.calcHist([src], [0], None, [256], [0, 256])
        histSortedIndex = np.argsort(hist, 0)[::-1]
        # 像素量最大的为背景颜色，其次记为线的颜色
        backgroundColor = histSortedIndex[0]
        lineColor = histSortedIndex[1]

        # 打印统计信息
        # print("""
        # in the Gray type
        # background color value: %d
        # line color value: %d
        # """ % (backgroundColor, lineColor))

        return backgroundColor, lineColor

    def LinePointDetect(self):
        # 提取线的中点坐标，将中点坐标输出
        scrGray = cv2.cvtColor(self.maskPic, cv2.COLOR_BGR2GRAY)
        scrGray = cv2.threshold(scrGray, 200, 255, cv2.THRESH_BINARY)[1]
        # 这里进行了一次旋转，因为np.where的遍历是沿着y行方向进行的
        idx_x, idx_y = np.where(cv2.rotate(scrGray, cv2.ROTATE_90_CLOCKWISE, 90))

        # 对x做一次差分，找出x有递增的点
        dx = np.diff(idx_x)
        # 获取递增点序号，并且在初始处插入一个0
        x_stage = np.insert(np.where(dx), 0, 0)

        # ，现在1表示一串相同x的初始点
        x_gap = np.diff(x_stage) + 1
        x_gap = np.floor_divide(x_gap, 2)
        x_gap = np.append(x_gap, 0)
        x_ = x_gap + x_stage

        idx_x_sep = idx_x[x_]
        idx_y_sep = idx_y[x_]

        return idx_x_sep, idx_y_sep

    @staticmethod
    def SideFilter(image: np.ndarray, boundry: int):
        # image should be gray image
        if image.ndim > 1:
            # if not gray, change into gray
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pos = np.where(image == boundry)
        img = np.zeros(image.shape)
        img[pos] = 255
        return img


def SideFilter(obj: BrokenLine):
    img = BrokenLine.SideFilter(obj.rawPic, obj.GetBoundry()[1])
    kernel = np.ones((2,2),np.uint8)
    img = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel=kernel)
    return img


def CalcFigureColorValue(obj: BrokenLine):
    # 转换为灰度图像
    src = cv2.cvtColor(obj.rawPic, cv2.COLOR_BGR2GRAY)

    # 这里要将histRange设置为[0,256]，才会将空白计算进去，这是一个右开区间，同histSize=[256]
    # 同时，只对图像的主部分进行取样，不对边界进行色域提取，掩膜使用原来定义的掩膜
    hist = cv2.calcHist([src], [0], obj.GetMask(), [256], [0, 256])

    histSortedIndex = np.argsort(hist, 0)[::-1]

    backgroundColor = histSortedIndex[0]
    lineColor = histSortedIndex[1]
    print("""
    in the Gray type
    background color value: %d
    line color value: %d
    """ % (backgroundColor, lineColor))

    return hist, backgroundColor, lineColor


def ReadDetail(obj: BrokenLine):
    src = cv2.cvtColor(obj.rawPic, cv2.COLOR_BGR2RGB)
    detail = pytesseract.image_to_string(src)
    return detail


# 测试
def test():
    obj = BrokenLine(5)
    img = obj.rawPic
    # 保留原图，进行三色道分析
    imgProcessed = img

    # 边缘检测
    lp = cv2.Laplacian(img, cv2.CV_16S, ksize=3).astype(np.uint8)
    # 将边缘检测图像改为
    src = cv2.cvtColor(lp, cv2.COLOR_BGR2GRAY)

    hist, backgroundColor, lineColor = CalcFigureColorValue(obj)
    plt.subplot(4, 4, 1), plt.imshow(src, 'gray'), plt.axis("off"), plt.title("Laplacian gray")
    plt.subplot(4, 4, 5), plt.imshow(lp, 'gray'), plt.axis('off'), plt.title("Laplacian")
    plt.subplot(4, 4, 2), plt.imshow(obj.rawPic), plt.axis('off'), plt.title("raw picture")
    plt.subplot(4, 4, 6), plt.imshow(obj.GetMaskPic(), 'gray'), plt.axis("off"), plt.title("Masked")

    # 三色道拆分，分析色域，取区间[1,255]，也就是不分析黑色0与白色255的value
    hist_0 = cv2.calcHist([imgProcessed], [0], None, [256], [1, 255])
    hist_1 = cv2.calcHist([imgProcessed], [1], None, [256], [1, 255])
    hist_2 = cv2.calcHist([imgProcessed], [2], None, [256], [1, 255])
    rowSlicedCount = 8
    plt.subplot(rowSlicedCount, 2, int(2 * (rowSlicedCount / 2 + 1))), plt.plot(hist_0, 'b'), plt.title("color hist b")
    plt.subplot(rowSlicedCount, 2, int(2 * (rowSlicedCount / 2 + 2))), plt.plot(hist_1, 'g'), plt.title("color hist g")
    plt.subplot(rowSlicedCount, 2, int(2 * (rowSlicedCount / 2 + 3))), plt.plot(hist_2, 'r'), plt.title("color hist r")
    plt.subplot(rowSlicedCount, 2, int(2 * (rowSlicedCount / 2 + 4))), plt.plot(hist), plt.title("gray hist")
    plt.show()
    pass


# 测试1 获取边界
def test_1():
    obj = BrokenLine(5)
    img = obj.rawPic
    imgProcessed = img
    src, hist, lp = CalcFigureColorValue(obj)
    lp_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    contours = cv2.findContours(lp_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(contours)
    plt.subplot(1, 2, 1), plt.imshow(lp, 'gray')
    plt.subplot(1, 2, 2), plt.imshow(lp_gray, 'gray')
    plt.show()
    pass


def test_Filter():
    obj = BrokenLine(5)
    img = SideFilter(obj)
    plt.imshow(img, 'gray')
    plt.show()
    pass


if __name__ == '__main__':
    test_Filter()
