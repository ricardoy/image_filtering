import sys
import numpy as np
import cv2
import imutils

def color_filter(img):
    target_color = np.array((0, 255, 0), dtype=np.int)
    # blur = cv2.GaussianBlur(img,(5,5),0)
    dist = np.sqrt(np.sum((img-target_color) ** 2, axis=2))
    lower = np.array([0])
    upper = np.array([205])
    mask = cv2.inRange(dist, lower, upper)
    res = cv2.bitwise_and(img, img, mask=mask)
    print('min:', np.min(dist), 'max:', np.max(dist))
    return res, mask


def filter(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    c=60

    lower = np.array([0, c, c])
    upper = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    lower2 = np.array([160, c, c])
    upper2 = np.array([179, 255, 255])
    mask2 = cv2.inRange(hsv, lower2, upper2)

    # lower = np.array([30,150,30])
    # upper = np.array([150,255,150])
    #
    # l = 180
    # lower = np.array([l, l, l])
    # upper = np.array([255,255,255])
    #
    # mask = cv2.inRange(img, lower, upper)

    mask = mask | mask2

    res = cv2.bitwise_and(img, img, mask=mask)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    return res, mask


def main(filename):
    cap = cv2.VideoCapture(filename)

    while(cap.isOpened()):
        ret, img = cap.read()
        # print(img.shape)
        # img = imutils.rotate(img, 90)
        img = np.rot90(img)


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
