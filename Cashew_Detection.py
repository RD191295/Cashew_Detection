import cv2
import numpy as np
img = cv2.imread("pic/IMG-20220205-WA0009.jpg")

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)


# IMAGE THRESHOLDING :
# IMAGE-9 ---- 45 to 160
ret, thresh_out = cv2.threshold(blur_img, 45, 160, cv2.THRESH_BINARY_INV)
kernel_ip = np.ones((2, 2), np.uint8)
eroded_ip = cv2.erode(thresh_out, kernel_ip, iterations=1)
dilated_ip = cv2.dilate(eroded_ip, kernel_ip, iterations=1)

contours, hierarchy = cv2.findContours(dilated_ip, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for i,cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    if area > 2000:
        if hierarchy[0,i,3] == -1:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(img, "Cashew", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

cv2.imshow("Image", dilated_ip)
cv2.imshow("final", img)

if cv2.waitKey(0)  == ord("q"):
    cv2.destroyAllWindows()
