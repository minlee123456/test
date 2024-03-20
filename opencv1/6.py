import cv2

title1 = "imread Test"
image = cv2.imread('../read_color.jpg')
if image is None:
    print("파일 읽기 오류")

cv2.imshow(title1,image)
cv2.waitKey(8)