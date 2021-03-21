import cv2
import imutils
import numpy as np
img=cv2.imread("m.png")
for angle in np.arange(0, 360, 15):
	rotated = imutils.rotate_bound(img, 60)
	cv2.imshow("ff",rotated)
	cv2.imwrite("rotate60.jpg",rotated)
	cv2.waitKey(0)
cv2.destroyAllWindows()
