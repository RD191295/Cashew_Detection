import cv2
import numpy as np
import math

cap = cv2.VideoCapture("Input_Video/Cashew_2.mp4")

cashew_hsv_lower = np.array([19, 70, 128], np.uint8)
cashew_hsv_upper = np.array([50, 255, 255], np.uint8)

# SIZE
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
cashew_count = 0
# size = (frame_width, frame_height)
# result = cv2.VideoWriter('output_detection.avi', cv2.VideoWriter_fourcc(*'MJPG'),10,size)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame = frame[300:750, :]
    frame_blur = cv2.GaussianBlur(frame, (7, 7), 0)

    # converting  BGR to HSV Frame
    hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)

    # finding the range of cashew_hsv color in the image
    cashew_hsv = cv2.inRange(hsv, cashew_hsv_lower, cashew_hsv_upper)

    kernal = np.ones((5, 5), "uint8")

    # dilation of the image ( to remove noise) create mask for cashew_hsv color
    cashew_hsv = cv2.dilate(cashew_hsv, kernal, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=cashew_hsv)

    contours, hierarchy = cv2.findContours(cashew_hsv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if hierarchy[0, i, 3] == -1:
            x, y, w, h = cv2.boundingRect(contour)
            if area > 2500 and h > 100:
                cashew_count = cashew_count + 1
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "cashew", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0))
                cv2.putText(frame, "detected cashew are :" + str(cashew_count), (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0))

    # result.write(frame)
    cv2.imshow("Color", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
