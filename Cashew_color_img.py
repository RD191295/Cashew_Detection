import cv2
import numpy as np

frame = cv2.imread("pic\IMG-20220205-WA0009.jpg")

frame_blur = cv2.GaussianBlur(frame, (7, 7), 0)

# converting  BGR to HSV Frame
hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)

cashew_hsv_lower = np.array([6, 65, 128], np.uint8)
cashew_hsv_upper = np.array([54, 255, 255], np.uint8)

# finding the range of cashew_hsv color in the image
cashew_hsv = cv2.inRange(hsv, cashew_hsv_lower, cashew_hsv_upper)

kernal = np.ones((5, 5), "uint8")

# dilation of the image ( to remove noise) create mask for cashew_hsv color
cashew_hsv = cv2.dilate(cashew_hsv, kernal, iterations = 1)
res = cv2.bitwise_and(frame, frame, mask = cashew_hsv)


contours, hierarchy = cv2.findContours(cashew_hsv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for i, contour in enumerate(contours) :
    area = cv2.contourArea(contour)

    if hierarchy[0, i, 3] == -1 :
        x, y, w, h = cv2.boundingRect(contour)
        if area > 2500 and h > 42:
            print(x, y, w, h)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "cashew", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0))

cv2.imshow("Color", frame)
cv2.imshow("Cashew", hsv)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()