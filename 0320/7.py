import cv2

capture = cv2.VideoCapture("../video_file.avi")
if not capture.isOpened(): raise Exception("파일 없음")

frame_rate = capture.get(cv2.CAP_PROP_FPS)
delay = int(1000 / frame_rate)
frame_cnt = 0

while True:
    ret, frame = capture.read()
    if not ret or cv2.waitKey(delay) >= 0: break
    blue, green, red =cv2.split(frame)
    frame_cnt += 1

    if 100 <= frame_cnt < 200: cv2.add(blue, 100, blue)
    elif 200 <= frame_cnt < 300: cv2.add(green, 100, green)
    elif 300 <= frame_cnt < 400: cv2.add(red, 100, red)

    frame = cv2.merge( [blue, green, red] )
    cv2.imshow("Read Video", frame)

capture.release