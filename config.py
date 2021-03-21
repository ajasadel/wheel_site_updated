params = {
    'x1': 432,# manual rectangle box x1,y1,x2,y2 for check wheel size
    'x2':770,
    'y1': 180,
    'y2': 519,
    'cx': 604,# manual circle center of big wheel
    'cy': 360,
    'radius': 180,
    
}
params2= {
	'x1': 383,
    'x2':758,
    'y1': 218,
    'y2': 576,
    'cx': 191,# manual center& radius of bg wheel
    'cy': 175,
    'radius': 210,
}
cam= {
	'cam1':'dsp-dir-3.mp4',# video link
}
display={
	'orignal_feed':True,
    'rotated_feed': True,# show window
    'crop_images': True,
    'crop_text': True,
}
padding={
    'x1_shift':250,#detect box shifting(padding)
    'y1_shift':30,
    'x2_shift':30,
    'y2_shift':20,
}
