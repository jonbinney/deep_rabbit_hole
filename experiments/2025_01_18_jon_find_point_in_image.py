import cv2
import matplotlib.pyplot as plt
import numpy as np

image1 = cv2.imread("datasets/water_test_set5/images/2024-10-19/0-1729341601.jpg")
image2 = cv2.imread("datasets/water_test_set5/images/2024-10-29/0-1730203202.jpg")

# Convert images to grayscale
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(image1, None)
kp2, des2 = sift.detectAndCompute(image2, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

if len(good) > 3:
    p1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    p2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

draw_params = dict(
    matchColor=(0, 255, 0),  # draw matches in green color
    singlePointColor=None,
    flags=2,
)

image_siftmatch = cv2.drawMatches(image1, kp1, image2, kp2, good, None, **draw_params)
plt.imshow(image_siftmatch)
plt.show()
