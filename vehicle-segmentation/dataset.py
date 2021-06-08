import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class MapillaryDataset(Dataset):
	def __init__(self, image_dir, mask_dir, transform = None, multiple_classes = False):
		self.image_dir 			= image_dir
		self.mask_dir  			= mask_dir
		self.transform 			= transform
		self.multiple_classes 	= multiple_classes
		self.images    			= os.listdir(self.image_dir)


	def __len__(self):
		return len(self.images)

	def __getitem__(self, index):

		mask_path = ""
		mask      = None

		img_path = os.path.join(self.image_dir, self.images[index])
		image = cv2.imread(img_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		if self.transform is not None:
			augmentation = self.transform(image = image, mask=None)
			image = augmentation['image']

		if not self.multiple_classes:
			mask_path = os.path.join(self.mask_dir, self.images[index]).replace('.jpg', '.png')
			mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
			mask = mask / 255.

			mask = augmentation['mask']

			if self.transform is not None:
				augmentation = self.transform(image=mask, mask=None)
				mask = augmentation['image']

			return image, mask.unsqueeze(0)
		else:
			mask_channel = None
			for mask_path in self.mask_dir:
				mask_path_item = os.path.join(mask_path, self.images[index]).replace('.jpg', '.png')
				mask = cv2.imread(mask_path_item, cv2.IMREAD_GRAYSCALE)
				# Process
				mask = mask / 255. # binary image
				mask = mask.reshape((mask.shape[0], mask.shape[1])) # Add gray channel
				# Combination
				mask_channel = mask if mask_channel is None else np.concatenate((mask_channel, mask), axis = 2)

			if self.transform is not None:
				augmentation = self.transform(image=mask_channel, mask=None)
				mask_channel = augmentation['image']

			return image, mask_channel

