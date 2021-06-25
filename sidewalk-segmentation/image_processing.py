import cv2
import os 
import matplotlib.pyplot as plt
import numpy as np 

from parameters import *
from tqdm import tqdm

def sidewalkExtractor(source, dist):
	list_imgs = os.listdir(source)

	os.mkdir(dist)
	looper = tqdm(list_imgs, leave=False)
	for img_name in looper:
		out = cv2.cvtColor(cv2.imread(source + img_name), cv2.COLOR_BGR2RGB)

		# Extract sidewalk
		mask = np.zeros((out.shape[0], out.shape[1]))
		mask[np.sum(out == np.array(SIDEWALK_COLOR), 2) == 3] = 255.0

		# Write
		cv2.imwrite(dist + img_name, mask)


def roadExtractor(source, dist):
	list_imgs = os.listdir(source)

	os.mkdir(dist)
	looper = tqdm(list_imgs, leave=False)
	for img_name in looper:
		out = cv2.cvtColor(cv2.imread(source + img_name), cv2.COLOR_BGR2RGB)

		# Extract sidewalk
		mask = np.zeros((out.shape[0], out.shape[1]))
		mask[np.sum(out == np.array(ROAD_COLOR), 2) == 3] = 255.0

		# Write
		cv2.imwrite(dist + img_name, mask)


def lanelineExtractor(source, dist):
	list_imgs = os.listdir(source)

	os.mkdir(dist)
	looper = tqdm(list_imgs, leave=False)
	for img_name in looper:
		out = cv2.cvtColor(cv2.imread(source + img_name), cv2.COLOR_BGR2RGB)

		# Extract sidewalk
		mask = np.zeros((out.shape[0], out.shape[1]))
		mask[np.sum(out == np.array(LANELINE_COLOR), 2) == 3] = 255.0

		# Write
		cv2.imwrite(dist + img_name, mask)

def lanepointExtractor(source, dist):
	list_imgs = os.listdir(source)

	os.mkdir(dist)
	looper = tqdm(list_imgs, leave=False)
	for img_name in looper:
		out = cv2.cvtColor(cv2.imread(source + img_name), cv2.COLOR_BGR2RGB)

		# Extract sidewalk
		mask = np.zeros((out.shape[0], out.shape[1]))
		mask[np.sum(out == np.array(LANELINE_COLOR_MARKERS), 2) == 3] = 255.0

		# Write
		cv2.imwrite(dist + img_name, mask)

def trafficExtractor(source, dist):
	list_imgs = os.listdir(source)

	os.mkdir(dist)
	looper = tqdm(list_imgs, leave=False)
	for img_name in looper:
		out = cv2.cvtColor(cv2.imread(source + img_name), cv2.COLOR_BGR2RGB)

		# Extract sidewalk
		mask = np.zeros((out.shape[0], out.shape[1]))
		mask[np.sum(out == np.array(TRAFFIC_LIGHT), 2) == 3] = 255.0

		# Write
		cv2.imwrite(dist + img_name, mask)


def bicycleExtractor(source, dist):
	list_imgs = os.listdir(source)

	os.mkdir(dist)
	looper = tqdm(list_imgs, leave=False)
	for img_name in looper:
		out = cv2.cvtColor(cv2.imread(source + img_name), cv2.COLOR_BGR2RGB)

		# Extract sidewalk
		mask = np.zeros((out.shape[0], out.shape[1]))
		mask[np.sum(out == np.array(ROAD_BICYCLE), 2) == 3] = 255.0

		# Write
		cv2.imwrite(dist + img_name, mask)


def roadSuddenIssueExtractor(source, dist):
	list_imgs = os.listdir(source)

	os.mkdir(dist)
	looper = tqdm(list_imgs, leave=False)
	for img_name in looper:
		out = cv2.cvtColor(cv2.imread(source + img_name), cv2.COLOR_BGR2RGB)

		# Extract sidewalk
		mask = np.zeros((out.shape[0], out.shape[1]))
		mask[np.sum(out == np.array(ROAD_SUDDEN_ISSUE), 2) == 3] = 255.0

		# Write
		cv2.imwrite(dist + img_name, mask)


def parkingExtractor(source, dist):
	list_imgs = os.listdir(source)

	os.mkdir(dist)
	looper = tqdm(list_imgs, leave=False)
	for img_name in looper:
		out = cv2.cvtColor(cv2.imread(source + img_name), cv2.COLOR_BGR2RGB)

		# Extract sidewalk
		mask = np.zeros((out.shape[0], out.shape[1]))
		mask[np.sum(out == np.array(PARKING), 2) == 3] = 255.0

		# Write
		cv2.imwrite(dist + img_name, mask)

def streetSinkExtractor(source, dist):
	list_imgs = os.listdir(source)

	os.mkdir(dist)
	looper = tqdm(list_imgs, leave=False)
	for img_name in looper:
		out = cv2.cvtColor(cv2.imread(source + img_name), cv2.COLOR_BGR2RGB)

		# Extract sidewalk
		mask = np.zeros((out.shape[0], out.shape[1]))
		mask[np.sum(out == np.array(STREET_SINK), 2) == 3] = 255.0

		# Write
		cv2.imwrite(dist + img_name, mask)

def kerbExtractor(source, dist):
	list_imgs = os.listdir(source)

	os.mkdir(dist)
	looper = tqdm(list_imgs, leave=False)
	for img_name in looper:
		out = cv2.cvtColor(cv2.imread(source + img_name), cv2.COLOR_BGR2RGB)

		# Extract sidewalk
		mask = np.zeros((out.shape[0], out.shape[1]))
		mask[np.sum((out == np.array(KERB_COLOR_1)) | (out == np.array(KERB_COLOR_2)), 2) == 3] = 255.0

		# Write
		cv2.imwrite(dist + img_name, mask)


def crosswalkExtractor(source, dist):
	list_imgs = os.listdir(source)

	os.mkdir(dist)
	looper = tqdm(list_imgs, leave=False)
	for img_name in looper:
		out = cv2.cvtColor(cv2.imread(source + img_name), cv2.COLOR_BGR2RGB)

		# Extract sidewalk
		mask = np.zeros((out.shape[0], out.shape[1]))
		mask[np.sum(((out == np.array(CROSSWALK_1)) | (out == np.array(CROSSWALK_2))), 2) == 3] = 255.0

		# Write
		cv2.imwrite(dist + img_name, mask)


# Sidewalk

sidewalkExtractor(source= TRAIN_MASK_IMG_DIR, dist = TRAIN_MASK_SIDEWALK_DIR)
sidewalkExtractor(source= VAL_MASK_IMG_DIR, dist = VAL_MASK_SIDEWALK_DIR)
# Laneline
lanelineExtractor(source= TRAIN_MASK_IMG_DIR, dist = TRAIN_MASK_LANELINE_DIR)
lanelineExtractor(source= VAL_MASK_IMG_DIR, dist = VAL_MASK_LANELINE_DIR)
lanepointExtractor(source= TRAIN_MASK_IMG_DIR, dist = TRAIN_MASK_LANEPOINT_DIR)
lanepointExtractor(source= VAL_MASK_IMG_DIR, dist = VAL_MASK_LANEPOINT_DIR)
# Kerb
kerbExtractor(source= TRAIN_MASK_IMG_DIR, dist = TRAIN_MASK_BOUNDARY_DIR)
kerbExtractor(source= VAL_MASK_IMG_DIR, dist = VAL_MASK_BOUNDARY_DIR)
# Traffic
trafficExtractor(source=TRAIN_MASK_IMG_DIR, dist= TRAIN_MASK_TRAFFIC_DIR)
trafficExtractor(source=VAL_MASK_IMG_DIR, dist=VAL_MASK_TRAFFIC_DIR)
# Road
roadExtractor(source= TRAIN_MASK_IMG_DIR, dist = TRAIN_MASK_ROAD_DIR)
roadExtractor(source= VAL_MASK_IMG_DIR, dist = VAL_MASK_ROAD_DIR)
# Parking
parkingExtractor(source= TRAIN_MASK_IMG_DIR, dist = TRAIN_MASK_ROAD_PARKING)
parkingExtractor(source= VAL_MASK_IMG_DIR, dist = VAL_MASK_ROAD_PARKING)
# Sudden Issue
roadSuddenIssueExtractor(source= TRAIN_MASK_IMG_DIR, dist = TRAIN_MASK_ROAD_SUDDEN_DIR)
roadSuddenIssueExtractor(source= VAL_MASK_IMG_DIR, dist = VAL_MASK_ROAD_SUDDEN_DIR)
# bicycle
bicycleExtractor(source= TRAIN_MASK_IMG_DIR, dist = TRAIN_MASK_ROAD_BICYCLE_DIR)
bicycleExtractor(source= VAL_MASK_IMG_DIR, dist = VAL_MASK_ROAD_BICYCLE_DIR)
# Crosswalk
crosswalkExtractor(source=TRAIN_MASK_IMG_DIR, dist= TRAIN_MASK_CROSSWALK)
crosswalkExtractor(source= VAL_MASK_IMG_DIR, dist = VAL_MASK_CROSSWALK)
# Street sink
streetSinkExtractor(source=TRAIN_MASK_IMG_DIR, dist= TRAIN_MASK_STREET_SINK)
streetSinkExtractor(source= VAL_MASK_IMG_DIR, dist = VAL_MASK_STREET_SINK)
