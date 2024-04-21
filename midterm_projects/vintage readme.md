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

std_dev = 15

# sepia_mask라는 세피아 효과를 적용할 마스크 생성

sepia_mask = np.asarray([[0.272, 0.534, 0.131],
[0.349, 0.684, 0.168],
[0.393, 0.769, 0.189]], dtype=np.float32)

# 이미지에 가우시안 양방향 블러 효과를 적용한 dst변수 생성

dst = cv2.bilateralFilter(image,9,75,75)

# dst변수에 노이즈를 적용하고 dst1에 저장

dst1 = noise(std_dev, dst)

# dst1변수에 세피아 필터를 적용한 후 dst2에 저장

dst2 = cv2.transform(dst1, sepia_mask)

cv2.imshow("image", image)
cv2.imshow("dst2", dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()
