import matplotlib.pyplot as plt 
import numpy as np
import torch
import os
import cv2
import albumentations as Aug

from utils import (loadParameters)
from parameters import *
from models import UNET
from albumentations.pytorch import ToTensorV2
from moviepy.video.io.VideoFileClip import VideoFileClip

def cos(angle): return np.round(np.cos(np.deg2rad(angle)), decimals=5)
def sin(angle): return np.round(np.sin(np.deg2rad(angle)), decimals=5)

def globalModel():
	global model1, model2
	model1, model2 = loadModel()

def doSegmentation(image):

	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	height, width = image.shape[:2]

	transform = Aug.Compose([
		Aug.Resize(height=IMAGE_SIZE, width= IMAGE_SIZE),
		Aug.Normalize(
			mean=[0.0, 0.0, 0.0],
			std= [1.0, 1.0, 1.0],
			max_pixel_value = 255.0
			),
			ToTensorV2()
		])

	dtransform = Aug.Compose([Aug.Resize(height=height, width= width)])

	transform_out = transform(image= image)

	with torch.no_grad():
		# Prepare Image
		pre_image = transform_out['image'].unsqueeze(0).cuda()

		preds1 = model1(pre_image)
		preds2 = model2(pre_image)

		preds1 = torch.softmax(preds1, dim=1)
		preds1 = torch.argmax(preds1, dim=1).squeeze(0).detach().cpu().numpy()
		preds2 = torch.softmax(preds2, dim=1)
		preds2 = torch.argmax(preds2, dim=1).squeeze(0).detach().cpu().numpy()

	# Mask formulation
	road_mask_beam = np.zeros((pre_image.shape[2], pre_image.shape[3], 3))
	road_mask      = np.zeros((pre_image.shape[2], pre_image.shape[3], 3))
	sidewalk_mask  = np.zeros((pre_image.shape[2], pre_image.shape[3], 3))
	curb_maks      = np.zeros((pre_image.shape[2], pre_image.shape[3], 3))
	laneline_mask  = np.zeros((pre_image.shape[2], pre_image.shape[3], 3))

	# Assign values
	curb_maks[(preds2 == 1) & (preds1 ==0)]    = [125,0,125]
	sidewalk_mask[(preds2 == 2) & (preds1 == 0)] = [125, 125, 0]
	road_mask[preds1 == 2]     = [0, 0, 255]
	laneline_mask[preds1 == 1] = [0, 255, 0]
	road_mask_beam[preds1 == 1] = [255, 0, 0]
	road_mask_beam[preds1 == 2] = [255, 0, 0]

	center_point_x, center_point_y = road_mask.shape[1] // 2, road_mask.shape[0]

	beam_mask = centerSignalsBeam(angles=np.array(range(-100, 100, 5)),
								  image=road_mask_beam,
								  beam_length=1,
								  center=(center_point_x, center_point_y))

	mask = road_mask + laneline_mask + beam_mask +sidewalk_mask + curb_maks


	dtransform_out = dtransform(image = mask)

	out = cv2.addWeighted(image,0.7,dtransform_out['image'],0.3, 0, dtype=cv2.CV_64F)
	img =  cv2.cvtColor(out.astype('uint8'), cv2.COLOR_RGB2BGR)

	return img



def loadModel():
	model1     = UNET(in_channels= COLOR_CHANNEL, out_channels= N_CLASSES)
	model2     = UNET(in_channels= COLOR_CHANNEL, out_channels= N_CLASSES)

	model1 = loadParameters(model = model1, optimizer = None, name= 'laneroad_model')
	model2 = loadParameters(model = model2, optimizer = None, name= 'sidewalk_curb_model')


	return model1.cuda(), model2.cuda()

def readImage(path):
	image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
	print(image.shape)
	transform = Aug.Compose([
		Aug.Resize(height=IMAGE_SIZE, width= IMAGE_SIZE),
		Aug.Normalize(
			mean=[0.0, 0.0, 0.0],
			std= [1.0, 1.0, 1.0],
			max_pixel_value = 255.0
			),
			ToTensorV2()
		])
	out = transform(image = image)

	return out['image'], image.shape[0], image.shape[1]

def addMask(image, mask, width, height):
	transform = Aug.Compose([Aug.Resize(height=height, width= width)])
	out = transform(image= image, mask= mask)

	print(out['image'].shape)
	print(out['mask'].shape)
	image_out =  cv2.addWeighted(out['image'],1,out['mask'],0.4, 0,dtype=cv2.CV_64F)
	print(out['mask'])
	plt.imshow(image_out)
	plt.show()


def doVideo(path, output):
	white_output = output

	clip1 = VideoFileClip(path)
	white_clip = clip1.fl_image(doSegmentation)
	white_clip.write_videofile(white_output, audio=False)

# This function draw beams to locate objects
def centerSignalsBeam(angles, image, beam_length, center, thershold = 2):

	out        = np.zeros(image.shape)
	used_angle = np.array([thershold] * angles.shape[0])
	x_point = []
	y_point = []

	for y in range(beam_length, center[1]):
		av_angles = (used_angle > 0)
		if av_angles.sum() == 0: break

		x_points = y * sin(angles[av_angles])/cos(angles[av_angles])
		x_points[np.abs(x_points) == np.inf] = 0
		x_points = center[0] + x_points


		indx_remove = []
		for i, x in enumerate(x_points[(x_points < image.shape[1]) & (x_points > 0)]):

			if image[center[1] - y, int(x), 0] == 0:
				# Counter reachs 1, then the current state is object.
				if used_angle[i] == 1:
					indx_remove.append(i)
					cv2.line(out, (int(x), center[1]-y), (center[0], center[1]), color=[255,255,0])

				# Counter to ensure there is an object at the given angle.
				else:
					used_angle[i] -=1
			# Update the threshold for incorrect mask.
			elif image[int(x), center[1] - y, 0] == 0 and used_angle[i] > 0 and used_angle[i] < thershold :
				used_angle[i] = thershold

		# Remove lines that reached to after finding object
		for j in reversed(indx_remove):
			used_angle = np.delete(used_angle, j)
			angles = np.delete(angles, j)

	return out

'''
# Load Model
model = loadModel()

# Prepare Image

img_path_list = os.listdir(VAL_IMG_DIR)

image_path = VAL_IMG_DIR + img_path_list[255]
print(image_path)
image, width, height = readImage(image_path)
#label= cv2.imread(VAL_MASK_IMG_DIR + img_path_list[19].replace('.jpg','.png'))
#label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
# Prediction

preds  = model(image.unsqueeze(0).float().cuda())
print(preds.shape)
preds  = torch.softmax(preds, dim=1)
preds  = torch.argmax(preds, dim = 1).squeeze(0).detach().cpu().numpy()

# Mask formulation
road_mask      = np.zeros((image.shape[1], image.shape[2], 3))
road_mask_beam = np.zeros((image.shape[1], image.shape[2], 3))
lane_line_mask = np.zeros((image.shape[1], image.shape[2], 3))

# Assign values
road_mask_beam[preds == 2] 	=  [255,0,0]
road_mask_beam[preds == 1] 	=  [255,0,0]
center_point_x, center_point_y = road_mask.shape[1]//2, road_mask.shape[0]

beam_mask = centerSignalsBeam(angles=np.array(range(-100,100,5)), image = road_mask, beam_length= 1, center=(center_point_x, center_point_y))
'''
'''
output_channel = pred.detach().cpu().numpy()
print(pred.argmax(dim=2))

out = np.zeros(output_channel.shape)
print(label)
out[:,:,output_channel.argmax(axis=2)] = 255.
print(image.permute(1,2,0).detach().cpu().numpy().shape)
plt.subplot(1,5,1)
plt.imshow(output_channel[:,:,0], cmap='gray')
plt.subplot(1,5,2)
plt.imshow(output_channel[:,:,1], cmap='gray')
plt.subplot(1,5,3)
plt.imshow(output_channel[:,:,2], cmap='gray')
plt.subplot(1,5,4)
plt.imshow(output_channel[:,:,3], cmap='gray')
plt.subplot(1,5,5)
plt.imshow(label)
plt.show()
'''
'''
# Load Model
globalModel()

# Read Image

img_path_list = os.listdir(VAL_IMG_DIR)

image_path = VAL_IMG_DIR + img_path_list[840]

image = cv2.imread(image_path)

# Do Segmentation

out = doSegmentation(image = image)

cv2.imwrite('t.png',out)

'''

globalModel()

doVideo(path=VIDEO_DIR+ 'loss ang.mp4', output= SAVE_VIDEO_DIR +'los ang.mp4')
