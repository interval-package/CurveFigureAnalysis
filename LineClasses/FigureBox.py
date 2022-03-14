import warnings
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math


def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area


def toPoints(cnt):
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(i, True)
    approx = cv2.approxPolyDP(i, 0.02 * peri, True)
    return approx, peri


def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew


def getRect(cnt):
    [x, y, w, h] = cv2.boundingRect(cnt)


class TransPicGetter(object):
    def __init__(self, pic: np.ndarray):
        self.pic = pic
        gray = pic.copy()
        self.height, self.width = pic.shape[0:2]
        if gray.ndim > 2:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        self.gray = gray
        imgBlur = cv2.GaussianBlur(gray, (5, 5), 1)  # ADD GAUSSIAN BLUR
        thresh = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11,
                                       2)  # 自适应阈值化
        kernel = np.ones((5, 5))
        imgDial = cv2.dilate(thresh, kernel, iterations=2)  # APPLY DILATION
        self.thresh = cv2.erode(imgDial, kernel, iterations=2)  # APPLY EROSION
        pass

    def getPics(self):
        contours, hierarchy = cv2.findContours(self.thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        img = self.pic.copy()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 3000:
                [x, y, w, h] = cv2.boundingRect(cnt)
                tempPic = img.copy()
                cv2.drawContours(tempPic, cnt, -1, [0, 0, 255])
                # tempPic = tempPic[x - 10:x + w + 10, y - 10:y + h + 10, :]
                try:
                    plt.imshow(tempPic)
                    plt.show()
                except:
                    pass
            # pics.append(self.SlicedPic())
        return

    class SlicedPic(object):
        def __init__(self, pic: np.ndarray, pos: tuple):
            self.pic = pic
            self.pos = pos
            pass


class PicWrapper(object):
    def __init__(self, pic: np.ndarray, processFlag=True):
        self.pic = pic
        gray = pic.copy()
        self.height, self.width = pic.shape[0:2]
        if gray.ndim > 2:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        self.gray = gray
        if processFlag:
            imgBlur = cv2.GaussianBlur(gray, (5, 5), 1)  # ADD GAUSSIAN BLUR
            thresh = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11,
                                           2)  # 自适应阈值化
            kernel = np.ones((5, 5))
            imgDial = cv2.dilate(thresh, kernel, iterations=2)  # APPLY DILATION
            self.thresh = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION
        else:
            self.thresh = gray
        pass

    def getBigPic(self):
        contours, hierarchy = cv2.findContours(self.thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        biggest, max_area = biggestContour(contours=contours)
        if biggest.size != 0:
            biggest = reorder(biggest)
            cv2.drawContours(self.pic, biggest, -1, (0, 255, 0), 20)  # DRAW THE BIGGEST CONTOUR
            # imgBigContour = drawRectangle(imgBigContour, biggest, 2)
            pts1 = np.float32(biggest)  # PREPARE POINTS FOR WARP
            pts2 = np.float32(
                [[0, 0], [self.width, 0], [0, self.height], [self.width, self.height]])  # PREPARE POINTS FOR WARP
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarpColored = cv2.warpPerspective(self.pic, matrix, (self.width, self.height))
            return imgWarpColored
        warnings.warn("fail to transform")
        return self.gray


def main(id):
    img = cv2.imread("../../data/img_train_BrokenLine/%d/draw.png" % id)
    src = img.copy()
    src2 = np.zeros(img.shape, np.uint8) * 255
    maskpic_target = cv2.imread("../../data/img_train_BrokenLine/%d/draw_mask.png" % id)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # ADD GAUSSIAN BLUR

    thresh = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)  # 自适应阈值化
    # thresh_2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)  # 自适应阈值化
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    cv2.drawContours(image=src, contours=contours, contourIdx=-1, color=(255, 0, 0), thickness=2)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:  # 使用边缘面积过滤较小边缘框
            [x, y, w, h] = cv2.boundingRect(cnt)
            s = w * h
            if math.log(abs(s - area)) < math.log(area):
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                count += 1
                print("the %dth Contour here is the Properties\nx: %d\ny: %d\nw: %d\nh: %d\nS: %d\nArea: %f"
                      % (count, x, y, w, h, s, area))
                cv2.drawContours(image=src2, contours=cnt, contourIdx=-1,
                                 color=(count % 3 * 125, (count + 1) % 3 * 125, (count + 2) % 3 * 125),
                                 thickness=2)

    plt.figure("图像")

    plt.subplot(2, 2, 1)
    plt.imshow(img), plt.axis("off")
    plt.annotate(count, xy=(2, 1))

    plt.subplot(2, 2, 2)
    plt.imshow(src), plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(thresh, 'gray'), plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(src2, 'gray'), plt.axis("off")
    plt.show()


def main_2(id):
    img = cv2.imread("../../data/img_train_BrokenLine/%d/draw.png" % id)
    src = img.copy()
    src2 = np.zeros(img.shape, np.uint8) * 255
    maskpic_target = cv2.imread("../../data/img_train_BrokenLine/%d/draw_mask.png" % id)
    height, width, covers = src.shape

    # TransPicGetter(src).getPics()

    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    pts2 = np.float32([[0.1 * width, 0.1 * height], [0.7 * width, 0.1 * height], [0.2 * width, 0.7 * height],
                       [0.9 * width, 0.8 * height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(src, matrix, (width, height))
    obj = PicWrapper(imgWarpColored)
    # obj = PicWrapper(obj.getBigPic())
    plt.subplot(2, 2, 1)
    plt.imshow(obj.getBigPic())
    plt.subplot(2, 2, 2)
    plt.imshow(obj.thresh)
    plt.subplot(2, 2, 3)
    plt.imshow(obj.pic)
    plt.show()


def getMaxBox(src: np.ndarray):
    if src.ndim > 2:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        gray = src.copy()
    imgBlur = cv2.GaussianBlur(gray, (5, 5), 1)  # ADD GAUSSIAN BLUR
    # edge_output = cv2.Canny(imgBlur, 50, 150)
    thresh = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11,
                                   2)  # 自适应阈值化

    # 要注意先侵蚀，再膨胀
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    horizontalExtract = cv2.erode(thresh, kernel_h, iterations=3)
    horizontalExtract = cv2.dilate(horizontalExtract, kernel_h, iterations=2)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    verticalExtract = cv2.erode(thresh, kernel_v, iterations=3)
    verticalExtract = cv2.dilate(verticalExtract, kernel_v, iterations=2)
    # verticalExtract = cv2.morphologyEx()

    kernel_rec = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.bitwise_or(horizontalExtract, verticalExtract)
    thresh = cv2.dilate(thresh, kernel_rec, iterations=7)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(src, contours, -1, [0, 0, 225])
    bigCom = 0
    bcnt = None
    for cnt in contours:
        [x, y, w, h] = cv2.boundingRect(cnt)
        if w * h > bigCom:
            bcnt = cnt
    if bcnt is not None:
        [x, y, w, h] = cv2.boundingRect(bcnt)
    else:
        [x, y, w, h] = [0, 0, 0, 0]
    cv2.rectangle(src, [x, y], [x + w, y + h], [0, 255, 0])
    return [x, y, w, h], src


def main_3(id):
    # 使用形态学来提取边框
    img = cv2.imread("../../data/img_train_BrokenLine/%d/draw.png" % id)
    src = img.copy()
    src2 = np.zeros(img.shape, np.uint8) * 255
    maskpic_target = cv2.imread("../../data/img_train_BrokenLine/%d/draw_mask.png" % id)
    height, width, covers = src.shape

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(gray, (5, 5), 1)  # ADD GAUSSIAN BLUR
    # edge_output = cv2.Canny(imgBlur, 50, 150)
    thresh = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11,
                                   2)  # 自适应阈值化

    # 要注意先侵蚀，再膨胀
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    horizontalExtract = cv2.erode(thresh, kernel_h, iterations=3)
    horizontalExtract = cv2.dilate(horizontalExtract, kernel_h, iterations=2)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    verticalExtract = cv2.erode(thresh, kernel_v, iterations=3)
    verticalExtract = cv2.dilate(verticalExtract, kernel_v, iterations=2)
    # verticalExtract = cv2.morphologyEx()

    kernel_rec = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.bitwise_or(horizontalExtract, verticalExtract)
    # thresh = cv2.bitwise_and(horizontalExtract, verticalExtract)
    thresh = cv2.dilate(thresh, kernel_rec, iterations=7)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(src, contours, -1, [0, 0, 225])
    bigCom = 0
    bcnt = None
    for cnt in contours:
        [x, y, w, h] = cv2.boundingRect(cnt)
        if w * h > bigCom:
            bcnt = cnt
    [x, y, w, h] = cv2.boundingRect(bcnt)
    cv2.rectangle(src, [x, y], [x + w, y + h], [0, 255, 0])

    # obj = PicWrapper(thresh, False)

    plt.subplot(2, 2, 1)
    plt.imshow(src, 'gray')
    plt.subplot(2, 2, 2)
    plt.imshow(horizontalExtract, 'gray')
    plt.subplot(2, 2, 3)
    plt.imshow(verticalExtract, 'gray')
    plt.subplot(2, 2, 4)
    plt.imshow(thresh, 'gray')
    plt.show()


if __name__ == '__main__':
    for i in range(25, 50):
        main_2(i)
    pass
