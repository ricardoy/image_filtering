import sys
import numpy as np
import cv2
import imutils


def filter(img):
    lower = np.array([40,50,50])
    upper = np.array([80,255,255])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    # lower = np.array([30,150,30])
    # upper = np.array([150,255,150])
    #
    # l = 180
    # lower = np.array([l, l, l])
    # upper = np.array([255,255,255])
    #
    # mask = cv2.inRange(img, lower, upper)


    res = cv2.bitwise_and(img, img, mask=mask)
    return res, mask


def main(filename):
    cap = cv2.VideoCapture(filename)

    while(cap.isOpened()):
        ret, img = cap.read()
        img = imutils.rotate(img, 90)

        res, mask = filter(img)

        cv2.imshow('frame', img)
        # cv2.imshow('mask', mask)
        cv2.imshow('res', res)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv[1])
