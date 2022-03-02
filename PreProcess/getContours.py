import cv2
import matplotlib.pyplot as plt
import math

from Line.BrokenLineFigure import BrokenLineFigure


class Contours(object):
    def __init__(self):
        pass


def main():
    obj = BrokenLineFigure(5)
    img = obj.rawPic
    src = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)  # 自适应阈值化
    thresh_2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)  # 自适应阈值化
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    print(thresh)

    count = 0
    cv2.drawContours(image=src, contours=contours, contourIdx=-1, color=(1, 1, 1), thickness=2)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:  # 使用边缘面积过滤较小边缘框
            [x, y, w, h] = cv2.boundingRect(cnt)
            s = w * h
            if math.log(abs(s - area)) < math.log(area):
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                count += 1
                print("""
                the %d Contour
                here is the Properties
                x: %d
                y: %d
                w: %d
                h: %d
                S: %d
                Area: %f
                """ % (count, x, y, w, h, s, area))

    plt.figure("图像")

    plt.subplot(2, 2, 1)
    plt.imshow(img), plt.axis("off")
    plt.annotate(count, xy=(2, 1))

    plt.subplot(2, 2, 2)
    plt.imshow(src), plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(thresh), plt.axis("off")

    plt.figure()
    plt.imshow(thresh_2,'gray'), plt.axis("off")
    plt.show()


def main_2():

    pass


if __name__ == '__main__':
    main()
