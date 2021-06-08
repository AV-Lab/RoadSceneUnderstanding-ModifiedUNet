import cv2
import os 
import matplotlib.pyplot as plt
import numpy as np 

from parameters import *

# Identify number of classes

def vehicleExtractor(source, dist):
	list_imgs = os.listdir(source)
	
	os.mkdir(dist)
	
	for img_name in list_imgs:
		out = cv2.cvtColor(cv2.imread(source + img_name), cv2.COLOR_BGR2RGB)
		# Extract vehicles
		mask = np.zeros((out.shape[0], out.shape[1]))
		mask[out[:,:,2] == 142 ] = 255
		mask[out[:,:,2] == 230 ] = 255
		# Write
		cv2.imwrite(dist + img_name, mask)
	

vehicleExtractor(source = TRAIN_MASK_IMG_DIR, dist= TRAIN_MASK_VEHICLE_DIR)
vehicleExtractor(source = VAL_MASK_IMG_DIR, dist = VAL_MASK_VEHICLE_DIR)