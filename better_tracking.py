import cv2
import numpy as np
import sys

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')


def rotate_cut(img, x0, x1, y0, y1):
    img = np.rot90(img).astype(np.uint8).copy()
    return img[x0:x1, y0:y1]


if __name__ == '__main__':

    # Set up tracker.
    # Instead of MIL, you can also use

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE']
    tracker_type = tracker_types[0]

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()

    # Read video
    video = cv2.VideoCapture(sys.argv[1])

    # Exit if video not opened.
    if not video.isOpened():
        print
        "Could not open video"
        sys.exit()

    # Read first frame.
    ok, img = video.read()
    if not ok:
        print
        'Cannot read video file'
        sys.exit()

    img = rotate_cut(img, 400, 1128, 140, 940)

    # Define an initial bounding box
    bbox = (287, 23, 86, 320)

    # Uncomment the line below to select a different bounding box
    bbox = cv2.selectROI(img, False)

    x0, y0, x1, y1 = bbox
    print(x0, y0, x1, y1)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(img, bbox)

    while True:
        # Read a new frame
        ok, img = video.read()
        if not ok:
            break

        img = rotate_cut(img, 400, 1128, 140, 940)

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(img)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Draw bounding box
        p1 = (int(x0), int(y0))
        p2 = (int(x0 + x1), int(y0 + y1))
        cv2.rectangle(img, p1, p2, (255, 0, 0), 2, 1)
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(img, p1, p2, (0, 255, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(img, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display tracker type on frame
        cv2.putText(img, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display FPS on frame
        cv2.putText(img, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display X and Y
        cv2.putText(img, "X : " + '{0:+}'.format(int(bbox[0] - x0)), (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.putText(img, "Y : " + '{0:+}'.format(int(bbox[1] - y0)), (100, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display result
        cv2.imshow("Tracking", img)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27: break