import cv2
#cap=cv2.VideoCapture("rtsp://admin:Admin@123@192.168.0.146:554")
def onmouse(event,x,y,flags,param):
	if event== cv2.EVENT_LBUTTONDOWN:
		print(x,y)
cap=cv2.VideoCapture("rtsp://admin:Admin@123@192.168.0.146:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif")
while True:
	_,imag=cap.read()
	image=cv2.resize(imag,(1190,700))
	cv2.imshow("",image)
	cv2.setMouseCallback('',onmouse)
	cv2.waitKey(1)