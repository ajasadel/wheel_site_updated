import cv2
import numpy as np
#<<<<========== for increase the brightness=====>>>>>
def add_brightness(image):
	cont=np.ones(image.shape,dtype="uint8")*30
	inc_brightness=cv2.add(image,cont)
	return inc_brightness
#<<<============= for decrease the brightness======>>>
def sub_brightness(image):
	cont=np.ones(image.shape,dtype="uint8")*60
	dec_brightness=cv2.subtract(image,cont)
	return dec_brightness
#<<<<=========== histogram======>>>.
def histogram(image):
	hist,bins = np.histogram(image.flatten(),256,[0,256])

	cdf = hist.cumsum()
	cdf_normalized = cdf * hist.max()/ cdf.max()


	cdf_m = np.ma.masked_equal(cdf,0)
	cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
	cdf = np.ma.filled(cdf_m,0).astype('uint8')
	img2 = cdf[image]
	return img2
#<<<<<<===============gray image converter=====>>>>
def gray_img(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	return gray
#<<<==================clahe model of image========>>>
def clahe(image):
	image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
  
# The declaration of CLAHE  
# clipLimit -> Threshold for contrast limiting 
	clahe = cv2.createCLAHE(clipLimit =2,tileGridSize=(4,4)) 
	final_img = clahe.apply(image_bw) 
	return final_img
#<======image blur techinques====>
def gausian_blur(image):
	gausBlur = cv2.GaussianBlur(image, (5,5),0)  
	return gausBlur
def blur(image):
	avging = cv2.blur(image,(10,10)) 
	return avging
def median_blur(image):
	medBlur = cv2.medianBlur(image,5)
	return medBlur
def bilateral_filtering(image):
	bilFilter = cv2.bilateralFilter(image,9,75,75) 
	return bilFilter
def structingelement(image):
	rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (38, 18)) 
	return rect_kernel
def thresholding(image):
	r, threshold = cv2.threshold(image,140, 255, cv2.THRESH_BINARY)
	return threshold
def adaptive_threshold(image):
	img=cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
	return img
def adaptive_mean_threshold(image):
	cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
	return img

def sharpning(image):
	kernel_sharpening = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])

# Applying the sharpening kernel to the grayscale image & displaying it.
#print("\n\n--- Effects on S&P Noise Image with Probability 0.5 ---\n\n")

# Applying filter on image with salt & pepper noise
	sharpened_img = cv2.filter2D(image, -1, kernel_sharpening)
	return sharpened_img

def edge_detection(image):
	edge_det = cv2.Canny(image,100,200)
	return edge_det
def erosion(image):
	kernel = np.ones((5,5),np.uint8)
	img_erosion = cv2.erode(image, kernel, iterations=1)
	return img_erosion
def dilation(img):
	kernel = np.ones((5,5),np.uint8)

	img_dilation = cv2.dilate(img, kernel, iterations=1)
	return img_dilation
