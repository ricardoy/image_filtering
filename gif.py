import sys
import numpy as np
import cv2
import imutils

# convert -delay 20 -loop 0 *.jpg myimage.gif


def filter(img):
    lower = np.array([40,40, 40])
    upper = np.array([80,255,255])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    res = cv2.bitwise_and(img, img, mask=mask)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    return binary, mask


def generate_images(filename):
    cap = cv2.VideoCapture(filename)

    c = 0
    while(cap.isOpened()):
        ret, img = cap.read()
        img = imutils.rotate(img, 90)

        res, mask = filter(img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        c += 1
        if c > 100:
            break

        cv2.imwrite('out/%04d.png' % (c), res)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    generate_images(sys.argv[1])
