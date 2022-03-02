import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def train():
    im = cv2.imread('t.png')
    im3 = im.copy()

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # 先转换为灰度图才能够使用图像阈值化

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)  # 自适应阈值化


    ##################      Now finding Contours         ###################
    #

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 边缘查找，找到数字框，但存在误判

    samples = np.empty((0, 900))  # 将每一个识别到的数字所有像素点作为特征，储存到一个30*30的矩阵内
    responses = []  # label
    keys = [i for i in range(48, 58)]  # 48-58为ASCII码
    count = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 80:  # 使用边缘面积过滤较小边缘框
            [x, y, w, h] = cv2.boundingRect(cnt)
            if h > 25 and h < 30:  # 使用高过滤小框和大框
                count += 1
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
                roi = thresh[y:y + h, x:x + w]
                roismall = cv2.resize(roi, (30, 30))
                cv2.imshow('norm', im)
                key = cv2.waitKey(0)
                if key == 27:  # (escape to quit)
                    sys.exit()
                elif key in keys:
                    responses.append(int(chr(key)))
                    sample = roismall.reshape((1, 900))
                    samples = np.append(samples, sample, 0)
                if count == 100:  # 过滤一下过多边缘框，后期可能会尝试极大抑制
                    break
    responses = np.array(responses, np.float32)
    responses = responses.reshape((responses.size, 1))
    print("training complete")

    np.savetxt('generalsamples.data', samples)
    np.savetxt('generalresponses.data', responses)
    #
    cv2.waitKey()
    cv2.destroyAllWindows()





def getNum(src: np.ndarray):
    #######   training part    ###############
    samples = np.loadtxt('generalsamples.data', np.float32)
    responses = np.loadtxt('generalresponses.data', np.float32)
    responses = responses.reshape((responses.size, 1))

    model = cv2.ml.KNearest_create()
    model.train(samples, cv2.ml.ROW_SAMPLE, responses)


    im = src.copy()
    # im = cv2.imread(path)
    out = np.zeros(im.shape, np.uint8)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    gray = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)[1]
    thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    numbers = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 80:
            [x, y, w, h] = cv2.boundingRect(cnt)
            if h > 25:
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi = thresh[y:y + h, x:x + w]
                roismall = cv2.resize(roi, (30, 30))
                roismall = roismall.reshape((1, 900))
                roismall = np.float32(roismall)
                retval, results, neigh_resp, dists = model.findNearest(roismall, k=1)
                string = str(int((results[0][0])))
                numbers.append(int((results[0][0])))
                cv2.putText(out, string, (x, y + h), 0, 1, (0, 255, 0))
                count += 1
        if count == 10:
            break

    plt.figure()
    plt.subplot(1, 3, 1), plt.imshow(im), plt.axis('off')
    plt.subplot(1, 3, 2), plt.imshow(out), plt.axis('off')
    plt.subplot(1, 3, 3), plt.imshow(gray), plt.axis('off')

    plt.show()
    return numbers


if __name__ == '__main__':
    getNum('1.png')
