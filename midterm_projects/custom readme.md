import numpy as np, cv2

# 랜덤한 잡음을 추가하는 noise 변수를 정의한다

def noise(std, image):
noisy_image = image.copy()
height, width, channels = noisy_image.shape
for i in range(height):
for j in range(width):
for c in range(channels):
noise = np.random.normal()  # 정규 분포에서 랜덤한 노이즈 생성
set_noise = std * noise
noisy_image[i][j][c] = noisy_image[i][j][c] + set_noise
return noisy_image

image = cv2.imread("../Lenna.png")
if image is None: raise Exception("영상 읽기 오류")

# 노이즈 생성

std_dev = 25

# cv2.kmeans를 이용하여 색상을 단순화 하여 색조 단순화를 위한 코드

data = image.reshape(-1, 3).astype(np.float32)
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 0.001)
etval, bestLabels, centers = cv2.kmeans(data, 10, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
centers = centers.astype(np.uint8)

# image변수에 노이즈를 적용하고 dst에 저장

dst = noise(std_dev, image)

# dst변수에 색상 단순화를 적용하여 dst1에 저장

dst1 = centers[bestLabels].reshape(dst.shape)

# dst1변수에 밝기 높고 dst2에 저장

dst2 = cv2.add(dst1, 10)

cv2.imshow("image", image)
cv2.imshow('dst2', dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()
