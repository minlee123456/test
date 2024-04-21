import numpy as np, cv2

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

std_dev = 15

sepia_mask = np.asarray([[0.272, 0.534, 0.131],
                         [0.349, 0.684, 0.168],
                         [0.393, 0.769, 0.189]], dtype=np.float32)

dst = cv2.bilateralFilter(image,9,75,75)
dst1 = noise(std_dev, dst)
dst2 = cv2.transform(dst1, sepia_mask)

cv2.imshow("image", image)
cv2.imshow("dst2", dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()
