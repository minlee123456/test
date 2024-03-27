import numpy as np, cv2

def filter(image, mask):
    rows, cols = image.shape[:2]
    dst = np.zeros((rows, cols), np.float32)
    xcenter, ycenter = mask.shape[1]//2, mask.shape[0]//2

    for i in range(ycenter, rows - ycenter):
        for j in range(xcenter, cols - xcenter):
            y1, y2 = i - ycenter, i + ycenter + 1
            x1, x2 = j - xcenter, j + xcenter + 1
            roi = image[y1:y2, x1:x2].astype("float32")

            tmp = cv2.multiply(roi,mask)
            dst[i, j] = cv2.sumElems(tmp)[0]
    return dst

image = cv2.imread("../filter_sharpen.jpg", cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception("영상 읽기 오류")


data1= [0, -1, 0,
        -1, 5, -1,
        0, -1, 0]
dat2= [[-1,-1,-1],
       [-1,9,-1],
       [-1,-1,-1]]
mask1= np.array(data1, np.float32).reshape(3, 3)
mask2= np.array(dat2, np.float32)

sharpen1 = filter(image, mask1)
sharpen2 = filter(image, mask2)
sharpen1 = cv2.convertScaleAbs(sharpen1)
sharpen2 = cv2.convertScaleAbs(sharpen2)

cv2.imshow("image", image)
cv2.imshow("sharpen1", cv2.convertScaleAbs(sharpen1))
cv2.imshow("sharpen2", cv2.convertScaleAbs(sharpen2))
cv2.waitKey(0)
