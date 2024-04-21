![analog.png](..%2F..%2F..%2FOneDrive%2F%BB%E7%C1%F8%2F%BD%BA%C5%A9%B8%B0%BC%A6%2Fanalog.png)

import numpy as np
import cv2

# 랜덤한 잡음을 추가하는 noise 변수를 정의한다

def noise(std, image):
noisy_image = image.copy()
height, width, channels = noisy_image.shape
for i in range(height):
for j in range(width):
for c in range(channels):
noise = np.random.normal()
set_noise = std * noise
noisy_image[i][j][c] = noisy_image[i][j][c] + set_noise
return noisy_image

# 채도와 밝기를 낮추는 saturation_brightness_down 정의한다

def saturation_brightness_down(image, saturation=0, brightness=0):
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
hsv[:, :, 1] -= saturation
hsv[:, :, 2] -= brightness
return cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

# 파장설정

l = 20

# 진폭 설정

amp = 2

image = cv2.imread("../Lenna.png")
if image is None: raise Exception("영상 파일 읽기 오류")

# 노이즈 생성

std_dev = 10

# 얼마나 왜곡할지 값 설정

rows, cols = image.shape[:2]
mapy, mapx = np.indices((rows, cols), dtype=np.float32)
sinx = mapx + amp * np.sin(mapy / l)
cosy = mapy + amp * np.cos(mapx / l)

# image변수에 노이즈를 적용하고 dst에 저장

dst = noise(std_dev, image)

# dst변수에 밝기와 채도를 낮추고 dst1에 저장

dst1 = saturation_brightness_down(dst, saturation=30, brightness=30)

# dst1변수에 왜곡을 적용하고 dst2에 저장

dst2 = cv2.remap(dst1, sinx, cosy, cv2.INTER_LINEAR, \
None, cv2.BORDER_REPLICATE)

cv2.imshow("image", image)
cv2.imshow('dst2', dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()