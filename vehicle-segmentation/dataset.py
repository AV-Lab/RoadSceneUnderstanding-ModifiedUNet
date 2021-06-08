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
		img_path = os.path.join(self.image_dir, self.images[index])
		image = cv2.imread(img_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		if not self.multiple_classes:
			mask_path = os.path.join(self.mask_dir, self.images[index]).replace('.jpg', '.png')
			mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
			mask /= 255.

			if self.transform is not None:
				augmentation = self.transform(image=image, mask=mask)
				mask = augmentation['mask']
				image = augmentation['image']

			return image, mask.unsqueeze(0)
		else:
			mask_channel = None
			for mask_path in self.mask_dir:
				mask_path_item = os.path.join(mask_path, self.images[index]).replace('.jpg', '.png')
				mask = cv2.imread(mask_path_item, cv2.IMREAD_GRAYSCALE)
				# Process
				mask = mask / 255. # binary image
				mask = mask.reshape((mask.shape[0], mask.shape[1], 1)) # Add gray channel
				# Combination
				mask_channel = mask if mask_channel is None else np.concatenate((mask_channel, mask), axis = 2)

			if self.transform is not None:
				augmentation = self.transform(image= image, mask=mask_channel)
				mask_channel = augmentation['mask']
				image = augmentation['image']

			return image, mask_channel

import torch

img = torch.from_numpy(np.array([np.array([[1,2,3],[4,3,7],[10,23,423]]),
					np.array([[23, 23,122],[232,423,12],[2323,43,32]])]).astype(np.float32))
print(img.size())
print(torch.softmax(img, dim=2))
print(torch.softmax(img, dim=2).argmax(dim=2))