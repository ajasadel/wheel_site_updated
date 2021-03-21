# import the necessary packages
from imutils.object_detection import non_max_suppression
import imutils
import numpy as np
from copy import copy
import argparse
import time
import cv2
import pytesseract
from matplotlib import pyplot as plt
import tensorflow as tf 
import keras
import preprocess1 as ps
import string 
import random 
import redis
from PIL import Image
import io
import base64
import uuid
#import config
import mysql.connector
import argparse
import importlib

import os

import json
from collections import OrderedDict

import chainer
from pprint import pprint

import chainer.functions as F
import numpy as np

from PIL import Image
from chainer import configuration

from utils.datatypes import Size

ap = argparse.ArgumentParser()
ap.add_argument("model_dir",help="path to directory where model is saved")
ap.add_argument("snapshot_name", help="name of the snapshot to load")
#ap.add_argument("--image_path",default=, help="path to the image that shall be evaluated")
ap.add_argument("char_map", help="path to char map, that maps class id to character")
ap.add_argument("--gpu", type=int, default=-1, help="id of gpu to use [default: use cpu]")
ap.add_argument("-i", "--image", type=str,
	help="path to input image")
ap.add_argument("-east", "--east", type=str,
	help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.1,
	help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=800,
	help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=640,
	help="resized image height (should be multiple of 32)")


args = vars(ap.parse_args())
# config = tf.ConfigProto()
# # config = tf.ConfigProto(device_count={'GPU': 0})
# config.gpu_options.per_process_gpu_memory_fraction = .7
# sess = tf.Session(config=config)

#keras.backend.tensorflow_backend.set_session(sess)
def motion_check(frame1, frame2):
	min_area = 50

	frame1 = imutils.resize(frame1, width=500)
	frame_gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
	frame_gray1 = cv2.GaussianBlur(frame_gray1, (21, 21), 0)

	frame2 = imutils.resize(frame2, width=500)
	frame_gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
	frame_gray2 = cv2.GaussianBlur(frame_gray2, (21, 21), 0)

	frame_delta = cv2.absdiff(frame_gray1, frame_gray2)
	frame_thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
	frame_thresh = cv2.dilate(frame_thresh, None, iterations=2)

	cnts = cv2.findContours(frame_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
	#cnts = cnts[0] if imutils.is_cv2() else cnts[1]

	# loop over the contours
	motion = 0
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < min_area:
			continue
		motion = 1
	return motion

def get_class_and_module(log_data):
	if not isinstance(log_data, list):
		if 'InverseCompositional' in log_data:
			module_name = 'ic_stn.py'
			klass_name = log_data
		else:
			module_name = 'text_recognition.py'
			klass_name = log_data
	else:
		klass_name, module_name = log_data
	return klass_name, module_name


def load_module(module_file):
	module_spec = importlib.util.spec_from_file_location("models.model", module_file)
	module = importlib.util.module_from_spec(module_spec)
	module_spec.loader.exec_module(module)
	return module


def build_recognition_net(recognition_net_class, target_shape, args):
	return recognition_net_class(
		target_shape,
		num_rois=timesteps,
		label_size=52,
	)


def build_localization_net(localization_net_class, target_shape, args):
	return localization_net_class(
		dropout_ratio,
		timesteps,
		0,
		target_shape,
		zoom=1.0,
		do_parameter_refinement=False
	)


def build_fusion_net(fusion_net_class, localization_net, recognition_net):
	return fusion_net_class(localization_net, recognition_net)


def create_network(args, log_data):
	model_dir="model/"
	# Step 1: build network
	localization_net_class_name, localization_module_name = get_class_and_module(log_data['localization_net'])
	module = load_module(os.path.abspath(os.path.join(model_dir, localization_module_name)))
	localization_net_class = eval('module.{}'.format(localization_net_class_name))
	localization_net = build_localization_net(localization_net_class, log_data['target_size'], args)

	recognition_net_class_name, recognition_module_name = get_class_and_module(log_data['recognition_net'])
	module = load_module(os.path.abspath(os.path.join(model_dir, recognition_module_name)))
	recognition_net_class = eval('module.{}'.format(recognition_net_class_name))
	recognition_net = build_recognition_net(recognition_net_class, target_shape, args)

	fusion_net_class_name, fusion_module_name = get_class_and_module(log_data['fusion_net'])
	module = load_module(os.path.abspath(os.path.join(model_dir, fusion_module_name)))
	fusion_net_class = eval('module.{}'.format(fusion_net_class_name))
	net = build_fusion_net(fusion_net_class, localization_net, recognition_net)

	if gpu >= 0:
		net.to_gpu(gpu)

	return net


def load_image(image_file, xp, image_size):
	with Image.open(image_file) as the_image:
		the_image = the_image.convert('L')
		the_image = the_image.resize((image_size.width, image_size.height), Image.LANCZOS)
		image = xp.asarray(the_image, dtype=np.float32)
		image /= 255
		image = xp.broadcast_to(image, (3, image_size.height, image_size.width))
		return image


def strip_prediction(predictions, xp, blank_symbol):
	words = []
	for prediction in predictions:
		blank_symbol_seen = False
		stripped_prediction = xp.full((1,), prediction[0], dtype=xp.int32)
		for char in prediction:
			if char == blank_symbol:
				blank_symbol_seen = True
				continue
			if char == stripped_prediction[-1] and not blank_symbol_seen:
				continue
			blank_symbol_seen = False
			stripped_prediction = xp.hstack((stripped_prediction, char.reshape(1, )))
		words.append(stripped_prediction)
	return words


def extract_bbox(bbox, image_size, target_shape, xp):
	bbox.data[...] = (bbox.data[...] + 1) / 2
	bbox.data[0, :] *= image_size.width
	bbox.data[1, :] *= image_size.height

	x = xp.clip(bbox.data[0, :].reshape(target_shape), 0, image_size.width)
	y = xp.clip(bbox.data[1, :].reshape(target_shape), 0, image_size.height)

	top_left = (float(x[0, 0]), float(y[0, 0]))
	bottom_right = (float(x[-1, -1]), float(y[-1, -1]))

	return top_left, bottom_right

# mydb = mysql.connector.connect(
# 	host= "192.168.1.44",
# 	user= "tejust",
# 	password= "Thirumeni@1947",
# 	database= "wheel"
# )
# mycursor = mydb.cursor()
# N=10
# import tensorflow as tf
# import keras
# from keras.models import load_model
# from skimage.transform import resize
# from skimage import measure
# from skimage.measure import regionprops
# construct the argument parser and parse the arguments
# display=1
# model_char = keras.models.load_model('char2.h5')


# graph = tf.get_default_graph()
# display=1
def img2url(img):
	# OpenCV to PIL
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = Image.fromarray(img)

	# PIL Image to DataURL
	buffered = io.BytesIO()
	img.save(buffered, format="JPEG")
	img_enc = base64.b64encode(buffered.getvalue())
	img_str = 'data:image/jpeg;base64,{}'.format(img_enc.decode('utf-8'))

	return img_str


def straight_image(img):
	
	#
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

## (2) Detect circles
	circles = cv2.HoughCircles(gray, method=cv2.HOUGH_GRADIENT, dp=1, minDist=10, circles=None, param1=200, param2=60, minRadius = 40, maxRadius=0 )

	## make canvas
	canvas = img.copy()

	## (3) Get the mean of centers and do offset
	circles = np.int0(np.array(circles))
	x,y,r = 0,0,0
	for ptx,pty, radius in circles[0]:
		cv2.circle(canvas, (ptx,pty), radius, (0,255,0), 1, 16)
		x += ptx
# 		# Drawing a rectangle on c
		y += pty
		r += radius

	cnt = len(circles[0])
	x = x//cnt
	y = y//cnt
	r = r//cnt
	x+=5
	y-=7

	## (4) Draw the labels in red
	for r in range(100, r, 20):
		cv2.circle(canvas, (x,y), r, (0, 0, 255), 3, cv2.LINE_AA)
	cv2.circle(canvas, (x,y), 3, (0,0,255), -1)

	## (5) Crop the image
	dr = r + 100

	#dr=280
	#print(x,y,dr)
	croped = img[218:576,383:758].copy()
	cropped1 = imutils.rotate_bound(img, 90)

	#croped1=croped[y-dr:y+dr+1, x-dr:x+dr+1].copy()
	cv2.imshow("cropped",croped)
	#croped1 = imutils.rotate_bound(croped, 30)
	w = croped.shape[1]
	h = croped.shape[0]
	angle=0
	# # #rotate matrix


	# #ss=cv2.GetSize(croped1)
	# #w,h,_=croped1.shape
	# #c = (float(w/2.0), float(h/2.0))
	# #print(w,h)
	# rows=w
	# cols=h
	cv2.imshow("",croped)
	dst = cv2.warpPolar(croped, (int(h*3.5),int(w*3)),(191,175),210, flags=(cv2.WARP_POLAR_LOG))

	#img = np.zeros((w,h,3), dtype=np.uint8)
	## (6) logPolar and rotate
	#polar = cv2.logPolar(croped,(287,201),120,1)
	rotated = cv2.rotate(dst, cv2.ROTATE_90_COUNTERCLOCKWISE)
	#cv2,imshow("s",rotated)
	#print("a")
	#rs=cv2.resize(rotated.copy(),None,fx=3,fy=3,interpolation = cv2.INTER_CUBIC)
	angle+=60
	
	out=np.copy(rotated)
	#box,orig=video_text(out)
	return out		

def video_text(image):
	#print("a")

	orig = image.copy() 
	#cv2.resize(orig,(700,400))

	(H, W) = image.shape[:2]
	# set the new width and height and then determine the ratio in change
	# for both the width and height
	(newW, newH) = (args["width"], args["height"])
	rW = W / float(newW)
	rH = H / float(newH)
	# resize the image and grab the new image dimensions
	image = cv2.resize(image, (newW, newH))
	# display=1
	(H, W) = image.shape[:2]
	# define the two output layer names for the EAST detector model that
	# we are interested -- the first is the output probabilities and the
	# second can be used to derive the bounding box coordinates of text
	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]
	# load the pre-trained EAST text detector
	#print("[INFO] loading EAST text detector...")
	net = cv2.dnn.readNet(args["east"])
	# construct a blob from the image and then perform a forward pass of
	# the model to obtain the two output layer sets
	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	start = time.time()
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)
	end = time.time()
	# show timing information on text prediction
	#print("[INFO] text detection took {:.6f} seconds".format(end - start))
	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []
	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the geometrical
		# data used to derive potential bounding box coordinates that
		# surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]
		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability, ignore it
			if scoresData[x] < args["min_confidence"]:
				continue
			# compute the offset factor as our resulting feature maps will
			# be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)
			# extract the rotation angle for the prediction and then
			# compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)
			# use the geometry volume to derive the width and height of
			# the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]
			# compute both the starting and ending (x, y)-coordinates for
			# the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)
			# add the bounding box coordinates and probability score to
			# our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])
			# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	boxes = non_max_suppression(np.array(rects), probs=confidences)
	# loop over the bounding boxes
	results = []
	number=""
	char_scores = []
	lic_num_set=[]
	box=[]
	a=[]
	b=[]
	c=[]
	d=[]
	for (startX, startY, endX, endY) in boxes:

		# scale the bounding box coordinates based on the respective
		# ratios
		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)
		#if startY>200:
		# draw the bounding box on the image
		###cv2.rectangle(orig, (startX-20, startY), (endX, endY+30), (0, 255, 0), 2)
		box=[startX,startY,endX,endY]
		#print(box)
		a.append(startX)
		b.append(startY)
		c.append(endX)
		d.append(endY)
		#if startY>200:
		e=min(a)
		f=min(b)
		g=max(c)
		h=max(d)



		# char_images, image =char_segment(orig, box)

		# print("",char_images)

		# #cv2.imshow("",char_images)
		# if char_images is not None:
		# 	for char_image in char_images:
		# 		[char_score, pred_label] = char_recog(char_image)
		# 		char_scores.append(char_score)
		# 		number += pred_label
		# 		lic_num_set.append(number)
		# 		print(lic_num_set)
		r = orig[startY:endY+30, startX-20:endX]
			#cv2.imshow("",r)

			#r1=orig[548:628+30,669-20:925]
			#cv2.imwrite("r1.jpg",r1)
			#gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
			#imagetotext(r)


	# 	#configuration setting to convert image to string.  
	# 	target = pytesseract.image_to_string(r,config='--psm 11--oem 3 -c tessedit_char_whitelist=0123456789')
	# 	print(target)

	# 	# append bbox coordinate and associated text to the list of results 
	# 	results.append(((startX, startY, endX, endY), target))
	# 	orig_image = orig.copy()

	# #Moving over the results and display on the image
	# for ((start_X, start_Y, end_X, end_Y), text) in results:
	# 	# display the text detected by Tesseract
	# 	print("{}\n".format(text))

	# 	# Displaying text
	# 	text = "".join([x if ord(x) < 128 else "" for x in text]).strip()
	# 	cv2.rectangle(orig_image, (start_X, start_Y), (end_X, end_Y),
	# 		(0, 0, 255), 2)
	# 	cv2.putText(orig_image, text, (start_X, start_Y - 30),
	# 		cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0, 255), 2)

	#cv2.imshow("",orig_image)
	#cv2.title('Output')
	#plt.show()
	#Display the image with bounding box and recognized text

	# show the output image

	
	cv2.rectangle(orig, (g-250, h-30), (g+30,h+20), (0, 255, 0), 2)
	cv2.imshow("Text Detection", orig)
	box=[g-250,h-30,g+30,h+20]
	crop=orig[h-30:h+20,g-250:g+30]


		#cv2.waitKey(1)
	if box:
		return crop

# def imagetotext(image):
# 	img=ps.clahe(image)
# 	#text =  pytesseract.image_to_string(cropped,config='--psm 11--oem 8 -c tessedit_char_whitelist=0123456789',timeout=2) 
# 	#print(text)
# 	#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
  
# # Performing OTSU threshold 
# 	ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) 
	  
# 	# Specify structure shape and kernel size.  
# 	# Kernel size increases or decreases the area  
# 	# of the rectangle to be detected. 
# 	# A smaller value like (10, 10) will detect  
# 	# each word instead of a sentence. 

# 	rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18)) 
	  
# 	# Appplying dilation on the threshold image 
# 	dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1) 
	  
# 	# Finding contours python 
# 	_,contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,  
# 													 cv2.CHAIN_APPROX_NONE) 
	  
# 	# Creating a copy of image 
# 	im2 = img.copy() 
	  
# 	# A text file is created and flushed 
# 	file = open("recognized.txt", "a+") 
# 	file.write("") 
# 	file.close() 
	  
# 	# Looping through the identified contours 
# 	# Then rectangular part is cropped and passed on 
# 	# to pytesseract for extracting text from it 
# 	# Extracted text is then written into the text file 
# 	for cnt in contours: 
# 		x, y, w, h = cv2.boundingRect(cnt) 
		  
# 		# Drawing a rectangle on copied image 
# 		rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2) 
		  
# 		# Cropping the text block for giving input to OCR 
# 		cropped = im2[y:y + h, x:x + w] 
		  
# 		# Open the file in append mode 
# 		file = open("recognized.txt", "a") 
# 	   # print("a")
		  
# 		# Apply OCR on the cropped image 
# 		text =  pytesseract.image_to_string(cropped,config='--psm 11--oem 8 -c tessedit_char_whitelist=0123456789',timeout=2) 
# 		print(text)
# 		word=text.split()
# 		length=len(word)

# 		for i in word :
# 			if len(i)==4:
# 				print(i)
		  
# 		# Appending the text into file 
# 		file.write(text) 
# 		file.write("\n") 
		  
# 		# Close the file 
# 		file.close 
# 		return cropped

# load the input image and grab the image dimensions
img=cv2.VideoCapture(args['image'])
frame_prev=None
crp_img=cv2.imread("new1.jpg")
crp_url=img2url(crp_img)
start_frame_number=1
img.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
skip=10

while True:
	for i in range(skip + 1):
		#img = cam.read()[1]
		image=img.read()[1]
		image=cv2.resize(image,(1190,700))
	cv2.imshow("",image)
	cv2.waitKey(1)
	# if image is None:
	# 	break
	# if frame_prev is None:
	# 	frame_prev = image
	# 	continue
	# frame_prev = copy(image)
	# motion=motion_check(image,frame_prev)
	#start_frame_number+=50

	#copy=np.float32(image)


	try:
		straight=straight_image(image)
		crop=video_text(straight)
		#bright=ps.sharpning(crop)
		#new2=cv2.resize(crop,(400,300))
		new=ps.clahe(crop)
		#edges = cv2.Canny(crop,200,300)
		#cv2.imshow("",edges)
		cv2.imwrite("new1.jpg",crop)


		#text =  pytesseract.image_to_string(new,config='--psm 11--oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ') 
		#print("d",text)
		cv2.imshow("crop",crop)
		#if box:
		#cropped=orig[box[1]:box[3],box[0]:box[2]]
		log_name = 'model/log'
		dropout_ratio = 0.5
		blank_symbol = 0
		# max number of text regions in the image
		timesteps = 23
		# max number of characters per word
		num_labels = 1
		gpu=-1
		model_dir="model/"
		snapshot_name="model_190000.npz"
		char_map="small_dataset/ctc_char_map.json"

		# open log and extract meta information
		with open(log_name) as the_log:
			log_data = json.load(the_log)[0]

		target_shape = Size._make(log_data['target_size'])
		image_size = Size._make(log_data['image_size'])

		xp = chainer.cuda.cupy if gpu >= 0 else np
		network = create_network(args, log_data)

		# load weights
		with np.load(os.path.join(model_dir,snapshot_name)) as f:
			chainer.serializers.NpzDeserializer(f).load(network)

		# load char map
		with open(char_map) as the_map:
			char_map = json.load(the_map)

		# load image
		image = load_image("new1.jpg", xp, image_size)
		with configuration.using_config('train', False):
			predictions, crops, grids = network(image[xp.newaxis, ...])

		# extract class scores for each word
		words = OrderedDict({})

		predictions = F.concat([F.expand_dims(prediction, axis=0) for prediction in predictions], axis=0)

		classification = F.softmax(predictions, axis=2)
		classification = classification.data
		classification = xp.argmax(classification, axis=2)
		classification = xp.transpose(classification, (1, 0))

		word = strip_prediction(classification, xp,blank_symbol)[0]

		word = "".join(map(lambda x: chr(char_map[str(x)]), word))

		bboxes = []
		for bbox in grids[0]:
			bbox = extract_bbox(bbox, image_size, target_shape, xp)
			bboxes.append(OrderedDict({
				'top_left': bbox[0],
				'bottom_right': bbox[1]
			}))
		words[word] = bboxes

		print(word)

	except Exception as e:
		print(e)
		pass
			#result=imagetotext(r)

			#result_str = ''.join(random.sample(string.ascii_lowercase, 8))
			#r = redis.Redis(host='192.168.1.39', port=6379, db=0)
			#p = r.pubsub()
			#ID=uuid.uuid1()
			#ID=str(ID)
			#print(ID)
			#full_img_path="d.png"
			#msg = {'id':ID,'number': result_str, 'crop_image': crp_url, 'full_image': full_img_path}
		#sql = "INSERT INTO Wheels (ID,WheelNo,Hardness,CropImg,FullImg,TestID ) VALUES (%s, %s,%s,%s,%s,%s)"
		#val=(ID,result_str,"0",crp_url,full_img_path,"343c0b3f-b9a6-470b-8d53-d097925e2ec5")
		#mycursor.execute(sql, val)

		#mydb.commit()
		#r.publish('msg', str(msg))
		#print(msg)



		#print("a")


