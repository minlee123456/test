import numpy as np, cv2


def saturation_up(image, saturation=0):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] += saturation
    addimage = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return addimage


image = cv2.imread("../Lenna.png")
if image is None: raise Exception("영상 파일 읽기 오류")

dst = cv2.GaussianBlur(image, (0, 0), 3)  # 가우시안 블러
dst1 = cv2.add(dst, 40)  # 밝기 조절
dst2 = saturation_up(dst1, 50)

cv2.imshow("image", image)
cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()
