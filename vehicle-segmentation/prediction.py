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

		pred = model(pre_image).squeeze(0).squeeze(0).detach().cpu()

	mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))
	mask[:,:, 0][pred >= 0.5] = 255.0

	dtransform_out = dtransform(image = mask)

	out = cv2.addWeighted(image,1,dtransform_out['image'].astype('uint8'),0.7, 0, dtype=cv2.CV_64F)
	return cv2.cvtColor(out.astype('uint8'), cv2.COLOR_RGB2BGR)



def loadModel():
	model     = UNET(in_channels= COLOR_CHANNEL, out_channels= N_CLASSES)

	model = loadParameters(model = model, optimizer = None, name= 'side_walk')

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

def addMask(image, mask, size):
	transform = Aug.Compose([Aug.Resize(height=IMAGE_SIZE*2, width= IMAGE_SIZE *2)])
	out = transform(image= image, mask= mask)

	
	image_out =  cv2.addWeighted(out['image'].astype(np.uint8),1,out['mask'].astype(np.uint8),0.7, 0)
	cv2.imshow('combined.png', image_out)
	cv2.waitKey()


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

image_path = VAL_IMG_DIR + img_path_list[19]

image, width, height = readImage(image_path)
label= cv2.imread(VAL_MASK_IMG_DIR + img_path_list[19].replace('.jpg','.png'))
label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
# Prediction

pred = torch.softmax(model(image.unsqueeze(0).float().cuda()), dim=1).squeeze(0).permute(1,2,0)
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
'''
globalModel()

doVideo(path='test.mp4', output='test_out.mp4')

'''