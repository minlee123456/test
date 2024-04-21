import numpy as np
import cv2

image = cv2.imread("../Lenna.png")
if image is None: raise Exception("영상 읽기 오류")

canny = cv2.Canny(image, 100, 150)

result = image.copy()
result[canny != 0] = [0, 0, 0]

data = result.reshape(-1, 3).astype(np.float32)
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 0.001)
retval, bestLabels, centers = cv2.kmeans(data, 10, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
centers = centers.astype(np.uint8)

dst = centers[bestLabels].reshape(result.shape)

cv2.imshow("image", image)
cv2.imshow("dst", dst)
cv2.waitKey()
cv2.destroyAllWindows()
