import argparse
import datetime
import imutils
import numpy as np
import time
import cv2

def update_gui(window_name, image, background_image, thresholded_image):
    # Convert them all to 3 channel and stack them side by side.
    background_3_channel = cv2.cvtColor(background_image, cv2.COLOR_GRAY2BGR)
    thesholded_3_channel = cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2BGR)
    combined_image = np.hstack((image, background_3_channel, thesholded_3_channel))
    cv2.imshow(window_name, combined_image)

parser = argparse.ArgumentParser()
parser.add_argument('video', help='path to the video file')
parser.add_argument('-a', '--min-area', type=int, default=200, help='minimum area of a change')
parser.add_argument('-t', '--threshold', type=int, default=20, help='pixel difference threshold')
parser.add_argument('-b', '--blur-size', type=int, default=21,
    help='size of guassian blur (must be odd')
args = parser.parse_args()

vs = cv2.VideoCapture(args.video)

window_name = 'Rabbits'

cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)

background_image = None
exit_triggered = False
while not exit_triggered:
    image = vs.read()[1]

    if image is None:
        break

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_image = cv2.GaussianBlur(grayscale_image, (args.blur_size, args.blur_size), 0)

    if background_image is None:
        background_image = grayscale_image
        continue

    diff_image = cv2.absdiff(background_image, grayscale_image)
    thresholded_diff_image = cv2.threshold(diff_image, args.threshold, 255, cv2.THRESH_BINARY)[1]

    threshholded_diff_image = cv2.dilate(thresholded_diff_image, None, iterations=2)
    contours = cv2.findContours(threshholded_diff_image.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    big_enough_contours = [c for c in contours if cv2.contourArea(c) > args.min_area]

    print('Frame stats:')
    print('              Contours : {}'.format(len(contours)))
    print('    Big enough contours : {}'.format(len(big_enough_contours)))

    for c in big_enough_contours:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 4)

    update_gui(window_name, image, background_image, thresholded_diff_image)

    while True:
        key = cv2.waitKey(30)
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print('Window closed - exiting')
            exit_triggered = True
            break
        if key == ord('c'):
            print('Exit key preasssed - exiting')
            exit_triggered = True
            break
        elif key >= 0:
            # We got some non-exit key. Process the next frame.
            break


cv2.destroyAllWindows()
