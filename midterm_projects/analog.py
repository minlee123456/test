import numpy as np
import cv2


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


def saturation_brightness_down(image, saturation=0, brightness=0):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] -= saturation
    hsv[:, :, 2] -= brightness
    return cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


l = 20
amp = 2

image = cv2.imread("../Lenna.png")
if image is None: raise Exception("영상 파일 읽기 오류")


std_dev = 10

rows, cols = image.shape[:2]
mapy, mapx = np.indices((rows, cols), dtype=np.float32)
sinx = mapx + amp * np.sin(mapy / l)
cosy = mapy + amp * np.cos(mapx / l)

dst = noise(std_dev, image)
dst1 = saturation_brightness_down(dst, saturation=30, brightness=30)
dst2 = cv2.remap(dst1, sinx, cosy, cv2.INTER_LINEAR, \
                 None, cv2.BORDER_REPLICATE)

cv2.imshow("image", image)
cv2.imshow('dst2', dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()
