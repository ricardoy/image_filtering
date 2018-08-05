import sys
import numpy as np
import cv2 as cv


def filter(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    c=70

    lower = np.array([0, c, c])
    upper = np.array([10, 255, 255])
    mask = cv.inRange(hsv, lower, upper)

    lower2 = np.array([160, c, c])
    upper2 = np.array([179, 255, 255])
    mask2 = cv.inRange(hsv, lower2, upper2)

    # lower = np.array([30,150,30])
    # upper = np.array([150,255,150])
    #
    # l = 180
    # lower = np.array([l, l, l])
    # upper = np.array([255,255,255])
    #
    # mask = cv.inRange(img, lower, upper)

    mask = mask | mask2

    res = cv.bitwise_and(img, img, mask=mask)
    gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 50, 255, cv.THRESH_BINARY)

    return res, mask


def cut(img, x0, x1, y0, y1):
    return img[x0:x1, y0:y1]


def main(filename):
    cap = cv.VideoCapture(filename)

    while(cap.isOpened()):
        ret, img = cap.read()
        img = np.rot90(img)
        img = cut(img, 400, 1128, 140, 940)
        img = cv.blur(img, (10, 10))
        res, mask = filter(img)

        cv.imshow('frame', img)

        cv.imshow('res', res)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv[1])
