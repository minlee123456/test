import numpy as np
import cv2

image = cv2.imread("../Lenna.png")
if image is None: raise Exception("영상 읽기 오류")

# 엣지 추출

canny = cv2.Canny(image, 100, 150)

# 이미지에 엣지추가후 result변수에 저장

result = image.copy()
result[canny != 0] = [0, 0, 0]`

# cv2.kmeans를 이용하여 색상을 단순화 하여 색조 단순화를 위한 코드

data = result.reshape(-1, 3).astype(np.float32)
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 0.001)
retval, bestLabels, centers = cv2.kmeans(data, 10, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
centers = centers.astype(np.uint8)

# result변수에 색상 단순화를 적용하여 dst에 저장

dst = centers[bestLabels].reshape(result.shape)

cv2.imshow("image", image)
cv2.imshow("dst", dst)
cv2.waitKey()
cv2.destroyAllWindows()
