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
from moviepy.editor import VideoFileClip


def globalModel():
	global model
	model = loadModel()

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

		preds = model(pre_image)
		preds = torch.softmax(preds, dim=1)
		preds = torch.argmax(preds, dim=1).squeeze(0).detach().cpu().numpy()

	# Mask formulation
	side_walk_mask = np.zeros((pre_image.shape[2], pre_image.shape[3], 3))
	curb_mask = np.zeros((pre_image.shape[2], pre_image.shape[3], 3))
	road_mask = np.zeros((pre_image.shape[2], pre_image.shape[3], 3))
	laneline_mask = np.zeros((pre_image.shape[2], pre_image.shape[3], 3))
	# Assign values
	side_walk_mask[preds == 3] = [255, 255, 0]
	curb_mask[preds == 1] = [255, 0, 0]
	road_mask[preds == 4] = [0, 0, 255]
	laneline_mask[preds == 2] = [0, 255, 0]

	mask = side_walk_mask + curb_mask + road_mask + laneline_mask

	dtransform_out = dtransform(image = mask)

	out = cv2.addWeighted(image,1,dtransform_out['image'],0.4, 0, dtype=cv2.CV_64F)
	return cv2.cvtColor(out.astype('uint8'), cv2.COLOR_RGB2BGR)



def loadModel():
	model     = UNET(in_channels= COLOR_CHANNEL, out_channels= N_CLASSES)

	model = loadParameters(model = model, optimizer = None, name= 'side_walk_bg')

	return model.cuda()

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
'''

# Load Model
model = loadModel()

# Prepare Image

img_path_list = os.listdir(VAL_IMG_DIR)

image_path = VAL_IMG_DIR + img_path_list[1450]
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
side_walk_mask = np.zeros((image.shape[1], image.shape[2], 3))
curb_mask      = np.zeros((image.shape[1], image.shape[2], 3))
road_mask      = np.zeros((image.shape[1], image.shape[2], 3))
laneline_mask  = np.zeros((image.shape[1], image.shape[2], 3))
print(preds)
# Assign values
side_walk_mask[preds == 3] 	=  [255,255,0]
curb_mask[preds == 1] 		= [255,0,0]
road_mask[preds == 4] 		= [0,0,255]
laneline_mask[preds == 2] 	= [0, 255, 0]

mask = side_walk_mask + curb_mask + road_mask + laneline_mask

addMask(image = image.permute(1,2,0).numpy(), mask = mask, width=height, height= width)
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

doVideo(path=VIDEO_DIR+ 'test.mp4', output= SAVE_VIDEO_DIR +'test_out_2.mp4')

